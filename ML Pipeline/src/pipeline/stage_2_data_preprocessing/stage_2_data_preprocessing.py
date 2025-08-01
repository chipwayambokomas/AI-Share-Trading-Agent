# src/pipeline/stage_2_data_preprocessing.py
"""
This module provides a unified data preprocessing pipeline that can operate in
two distinct modes, controlled by config.PREDICTION_MODE:

1.  "POINT" Mode: Prepares multivariate data for price point prediction.
    - For TCN/MLP, it creates individual sequences per stock.
    - For Graph-based models, it creates graph-structured sequences.
2.  "TREND" Mode: Prepares data for trend line prediction using a
    bottom-up piecewise segmentation algorithm (`ruptures`).
"""
import os
import time
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import ruptures as rpt

from src.utils import print_header, create_adjacency_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# GENERAL HELPER FUNCTIONS
# =============================================================================

def _clean_data(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Cleans the raw dataframe by handling missing and erroneous values.
    """
    logger.info("Starting data cleaning process...")
    cleaned_stocks = []
    
    for col in config.FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] <= 0, col] = np.nan
    
    initial_stock_count = df['StockID'].nunique()
    
    for stock_id, stock_df in df.groupby('StockID'):
        original_len = len(stock_df)
        
        missing_count = stock_df[config.TARGET_COLUMN].isnull().sum()
        missing_pct = missing_count / original_len if original_len > 0 else 0
        
        if missing_pct > 0.10:
            logger.warning(f"Excluding stock '{stock_id}': {missing_pct:.1%} missing target data ({missing_count}/{original_len} rows).")
            continue
            
        cleaned_stock_df = stock_df.copy()
        cleaned_stock_df[config.FEATURE_COLUMNS] = cleaned_stock_df[config.FEATURE_COLUMNS].ffill()
        cleaned_stock_df.dropna(subset=config.FEATURE_COLUMNS, inplace=True)
        
        if cleaned_stock_df.empty:
            logger.warning(f"Excluding stock '{stock_id}': Empty after cleaning.")
            continue
            
        cleaned_stocks.append(cleaned_stock_df)
        
    if not cleaned_stocks:
        msg = "No stocks remained after data cleaning. Check data quality."
        logger.error(msg)
        raise ValueError(msg)
        
    final_df = pd.concat(cleaned_stocks, ignore_index=True)
    final_stock_count = final_df['StockID'].nunique()
    
    logger.info(f"Data cleaning complete. Kept {final_stock_count}/{initial_stock_count} stocks.")
    return final_df

# =============================================================================
# HELPER FUNCTIONS FOR "TREND" MODE (USING BOTTOM-UP SEGMENTATION)
# =============================================================================

def _segment_one_stock_bottom_up(args):
    """
    Worker function to segment a SINGLE stock's time series into trends
    using the Bottom-Up algorithm.
    """
    stock_id, price_values, penalty = args
    
    if len(price_values) < 2:
        return None
        
    try:
        algo = rpt.BottomUp(model="l2").fit(price_values)
        breakpoints = algo.predict(pen=penalty)
        
        all_bps = [0] + breakpoints
        
        slopes, durations = [], []
        for start_idx, end_idx in zip(all_bps[:-1], all_bps[1:]):
            segment_len = end_idx - start_idx
            if segment_len > 1:
                y_segment = price_values[start_idx:end_idx]
                x_segment = np.arange(segment_len)
                slope = np.polyfit(x_segment, y_segment, 1)[0]
                slopes.append(slope)
                durations.append(segment_len)

        if not slopes: return None
        return (stock_id, np.array(slopes), np.array(durations))
        
    except Exception as e:
        logger.error(f"Stock {stock_id}: Error during bottom-up segmentation: {str(e)}")
        return None

def _process_stock_trend_sequences(args):
    """Process trend sequences for a single stock to create ML inputs."""
    stock_id, trends_data, slope_scaler, duration_scaler, window_size = args
    try:
        slopes = np.array(trends_data['slopes']).reshape(-1, 1)
        durations = np.array(trends_data['durations']).reshape(-1, 1)
        
        slope_scaled = slope_scaler.transform(slopes)
        duration_scaled = duration_scaler.transform(durations)
        scaled_trends = np.column_stack([slope_scaled.flatten(), duration_scaled.flatten()])
        
        X, y = [], []
        if len(scaled_trends) > window_size:
            for i in range(len(scaled_trends) - window_size):
                X.append(scaled_trends[i:(i + window_size)])
                y.append(scaled_trends[i + window_size])
        
        if len(X) > 0:
            return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        else:
            return stock_id, None, None
            
    except Exception as e:
        logger.error(f"Stock {stock_id}: Error processing sequences: {str(e)}")
        return stock_id, None, None

def _segment_one_stock_with_dates_bottom_up(args):
    """Worker function to segment a stock's time series using Bottom-Up and return trends with dates."""
    stock_id, stock_df, target_column, penalty = args
    price_values = stock_df[target_column].values
    dates = stock_df['Date'].values

    if len(price_values) < 2:
        return stock_id, []

    try:
        algo = rpt.BottomUp(model="l2").fit(price_values)
        breakpoints = algo.predict(pen=penalty)
        
        all_bps = [0] + breakpoints + [len(price_values)]
        all_bps = sorted(list(set(all_bps))) # Ensure uniqueness and order

        trends = []
        for start_idx, end_idx in zip(all_bps[:-1], all_bps[1:]):
            if end_idx - start_idx > 1:
                y_segment = price_values[start_idx:end_idx]
                x_segment = np.arange(len(y_segment))
                slope = np.polyfit(x_segment, y_segment, 1)[0]
                
                trends.append({
                    'start_date': dates[start_idx],
                    'end_date': dates[end_idx - 1],
                    'slope': slope,
                    'duration': len(y_segment)
                })
        return stock_id, trends
    except Exception as e:
        logger.error(f"Error segmenting stock {stock_id} with dates: {e}")
        return stock_id, []

# =============================================================================
# HELPER FUNCTIONS FOR "POINT" MODE
# =============================================================================

def _create_sequences_for_stock(args):
    """Worker function for creating sequences for a single stock (for TCN/MLP)."""
    stock_id, stock_df, scalers, feature_cols, target_col, in_win, out_win = args
    
    scaler = scalers.get(stock_id)
    if scaler is None:
        return stock_id, None, None

    # Scale the data for the current stock
    scaled_data = scaler.transform(stock_df[feature_cols])

    # Create sequences
    X, y = [], []
    total_len = in_win + out_win

    if len(scaled_data) >= total_len:
        for i in range(len(scaled_data) - total_len + 1):
            X.append(scaled_data[i : i + in_win, :])
            # The target is the specified target column over the output window
            target_col_idx = feature_cols.index(target_col)
            y.append(scaled_data[i + in_win : i + total_len, target_col_idx])
    
    if not X:
        return stock_id, None, None
        
    return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# =============================================================================
# MODE-SPECIFIC PREPROCESSING LOGIC
# =============================================================================

def _run_trend_prediction_for_graph(df, config):
    """Prepares trend data in a graph-compatible format using Bottom-Up segmentation."""
    print_header("Stage 2: Data Preprocessing (TREND PREDICTION for Graph Models - Bottom-Up)")
    cleaned_df = _clean_data(df, config)
    
    logger.info("Segmenting all stocks with Bottom-Up to extract trend data with dates...")
    task_args = [
        (sid, sdf, config.TARGET_COLUMN, config.SEGMENTATION_PENALTY)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(_segment_one_stock_with_dates_bottom_up, task_args), total=len(task_args), desc="Segmenting Trends"))
    
    trends_by_stock = {sid: trends for sid, trends in results if trends}
    if not trends_by_stock:
        raise ValueError("Trend segmentation failed for all stocks.")

    logger.info("Creating daily trend state matrix for all stocks...")
    all_dates = pd.to_datetime(np.unique(cleaned_df['Date'])).sort_values()
    stock_ids = sorted(trends_by_stock.keys())
    
    daily_slopes = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)
    daily_durations = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)

    for stock_id, trends in trends_by_stock.items():
        for trend in trends:
            daily_slopes.loc[trend['start_date']:trend['end_date'], stock_id] = trend['slope']
            daily_durations.loc[trend['start_date']:trend['end_date'], stock_id] = trend['duration']
    
    daily_slopes.ffill(inplace=True); daily_slopes.bfill(inplace=True)
    daily_durations.ffill(inplace=True); daily_durations.bfill(inplace=True)

    logger.info("Creating adjacency matrix from trend slope correlations...")
    correlation_matrix = daily_slopes.corr()
    adj_matrix = (correlation_matrix.abs() >= 0.5).astype(int)
    np.fill_diagonal(adj_matrix.values, 0)
    adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float32)

    logger.info("Scaling daily trend data...")
    train_end_idx = int(len(all_dates) * config.TRAIN_SPLIT)
    train_dates = all_dates[:train_end_idx]
    
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(daily_slopes.loc[train_dates])
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(daily_durations.loc[train_dates])
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}

    scaled_slopes = slope_scaler.transform(daily_slopes)
    scaled_durations = duration_scaler.transform(daily_durations)
    graph_data = np.stack([scaled_slopes, scaled_durations], axis=-1)

    logger.info("Creating graph-structured sequences from trend data...")
    X, y = [], []
    in_win, out_win = config.TREND_INPUT_WINDOW_SIZE, 1
    total_len = in_win + out_win

    for i in range(len(graph_data) - total_len + 1):
        X.append(graph_data[i : i + in_win, :, :])
        y.append(graph_data[i + in_win, :, :])
    
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)
    
    return X_np, y_np, stock_ids, scalers, adj_matrix_tensor

def _run_trend_prediction_parallel(df, config):
    """Executes the parallelized Bottom-Up preprocessing pipeline for non-graph models."""
    print_header("Stage 2: Data Preprocessing (TREND PREDICTION - Bottom-Up Parallel)")
    
    cleaned_df = _clean_data(df, config)
    logger.info(f"Target for segmentation: {config.TARGET_COLUMN}, Penalty: {config.SEGMENTATION_PENALTY}, Window: {config.TREND_INPUT_WINDOW_SIZE}")

    try:
        num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        num_cores = cpu_count() or 1

    logger.info("Segmenting full dataset using Bottom-Up approach...")
    full_task_args = [
        (sid, sdf[config.TARGET_COLUMN].values.astype(np.float32), config.SEGMENTATION_PENALTY)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_segment_one_stock_bottom_up, full_task_args), total=len(full_task_args), desc="Segmenting All Stocks")
        full_results = list(pbar)
    
    all_slopes, all_durations, trends_by_stock = [], [], {}
    for res in full_results:
        if res:
            stock_id, slopes, durations = res
            all_slopes.extend(slopes)
            all_durations.extend(durations)
            trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations}

    if not trends_by_stock:
        raise ValueError("Bottom-Up segmentation failed to produce any trends.")

    logger.info("Fitting global scalers for slope and duration...")
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(all_slopes).reshape(-1, 1))
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(all_durations).reshape(-1, 1))
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}

    logger.info("Creating sequences from trends...")
    seq_task_args = [
        (sid, data, scalers['slope'], scalers['duration'], config.TREND_INPUT_WINDOW_SIZE) 
        for sid, data in trends_by_stock.items()
    ]
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_process_stock_trend_sequences, seq_task_args), total=len(seq_task_args), desc="Creating Sequences")
        seq_results = list(pbar)

    all_X, all_y, all_stock_ids = [], [], []
    for stock_id, X_stock, y_stock in seq_results:
        if X_stock is not None and y_stock is not None:
            all_X.append(X_stock)
            all_y.append(y_stock)
            all_stock_ids.extend([stock_id] * len(X_stock))
    
    if not all_X:
        raise ValueError("Sequencing failed: No trend sequences could be created.")
    
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_stock_ids = np.array(all_stock_ids)
    
    return final_X, final_y, final_stock_ids, scalers, None

def _run_point_prediction(df, config):
    """Process multivariate data for point prediction mode."""
    print_header("Stage 2: Data Preprocessing (POINT PREDICTION - MULTIVARIATE)")
    
    cleaned_df = _clean_data(df, config)
    adj_matrix = None
    
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        # --- GRAPH-BASED POINT PREDICTION ---
        logger.info("Processing data for graph-based point prediction...")
        
        # 1. Pivot data to (dates, stocks, features)
        pivoted_df = cleaned_df.pivot(index='Date', columns='StockID', values=config.FEATURE_COLUMNS)
        pivoted_df.ffill(inplace=True)
        pivoted_df.bfill(inplace=True)
        
        stock_ids = pivoted_df.columns.get_level_values('StockID').unique().tolist()
        
        # 2. Create Adjacency Matrix
        adj_matrix = create_adjacency_matrix(cleaned_df, config.TARGET_COLUMN)
        
        # 3. Scale features per stock
        scalers = {}
        scaled_data_dict = {}
        
        train_end_date = pivoted_df.index[int(len(pivoted_df) * config.TRAIN_SPLIT)]
        
        for stock_id in tqdm(stock_ids, desc="Scaling features per stock"):
            stock_features_df = pivoted_df.xs(stock_id, level='StockID', axis=1)
            scaler = MinMaxScaler()
            scaler.fit(stock_features_df.loc[:train_end_date])
            scalers[stock_id] = scaler
            scaled_data_dict[stock_id] = scaler.transform(stock_features_df)
        
        # 4. Combine scaled data into a single numpy array
        num_dates = len(pivoted_df)
        num_nodes = len(stock_ids)
        num_features = len(config.FEATURE_COLUMNS)
        
        scaled_array = np.zeros((num_dates, num_nodes, num_features))
        for i, stock_id in enumerate(stock_ids):
            scaled_array[:, i, :] = scaled_data_dict[stock_id]

        # 5. Create sequences
        X, y = [], []
        in_win, out_win = config.POINT_INPUT_WINDOW_SIZE, config.POINT_OUTPUT_WINDOW_SIZE
        total_len = in_win + out_win
        
        for i in tqdm(range(len(scaled_array) - total_len + 1), desc="Creating graph sequences"):
            X.append(scaled_array[i : i + in_win, :, :])
            target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
            y.append(scaled_array[i + in_win : i + total_len, :, target_col_idx:target_col_idx+1])

        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

        return X_np, y_np, stock_ids, scalers, adj_matrix

    else:
        # --- STANDARD (TCN/MLP) POINT PREDICTION ---
        logger.info("Processing data for standard point prediction (TCN/MLP)...")
        
        # 1. Fit scalers on the training portion of each stock's data
        scalers = {}
        logger.info("Fitting scalers on training data for each stock...")
        for stock_id, stock_df in cleaned_df.groupby('StockID'):
            train_size = int(len(stock_df) * config.TRAIN_SPLIT)
            train_data = stock_df.head(train_size)
            if not train_data.empty:
                scaler = MinMaxScaler()
                scalers[stock_id] = scaler.fit(train_data[config.FEATURE_COLUMNS])

        # 2. Prepare arguments for the worker function, passing specific config values
        task_args = [
            (stock_id, stock_df, scalers, 
             config.FEATURE_COLUMNS, config.TARGET_COLUMN,
             config.POINT_INPUT_WINDOW_SIZE, config.POINT_OUTPUT_WINDOW_SIZE) 
            for stock_id, stock_df in cleaned_df.groupby('StockID')
        ]

        all_X, all_y, all_stock_ids = [], [], []
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(_create_sequences_for_stock, task_args), total=len(task_args), desc="Creating sequences per stock"))
        
        for stock_id, X_stock, y_stock in results:
            if X_stock is not None and y_stock is not None:
                all_X.append(X_stock)
                all_y.append(y_stock)
                all_stock_ids.extend([stock_id] * len(X_stock))

        if not all_X:
            raise ValueError("No sequences could be created from the data.")
            
        final_X = np.concatenate(all_X, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        
        return final_X, final_y, np.array(all_stock_ids), scalers, None


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def run(df, config):
    """Main entry point for data preprocessing."""
    logger.info(f"Starting preprocessing with mode: {config.PREDICTION_MODE}")
    logger.info(f"Using model type: {config.MODEL_TYPE}")
    
    if config.PREDICTION_MODE == "TREND" and not hasattr(config, 'SEGMENTATION_PENALTY'):
        raise AttributeError("Configuration must include 'SEGMENTATION_PENALTY' for TREND mode.")

    if config.PREDICTION_MODE == "TREND":
        if config.MODEL_TYPE in ["GraphWaveNet", "DSTAGNN"]:
            X, y, stock_ids, scalers, adj_matrix = _run_trend_prediction_for_graph(df, config)
        else:
            X, y, stock_ids, scalers, adj_matrix = _run_trend_prediction_parallel(df, config)
    elif config.PREDICTION_MODE == "POINT":
        X, y, stock_ids, scalers, adj_matrix = _run_point_prediction(df, config)
    else:
        raise ValueError(f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'.")
    
    if X.size > 0:
        logger.info("=== PREPROCESSING SUMMARY ===")
        logger.info(f"Total sequences generated: {len(X)}")
        logger.info(f"Input sequences shape (X): {X.shape}")
        logger.info(f"Target sequences shape (y): {y.shape}")
        unique_ids = stock_ids if isinstance(stock_ids, list) else np.unique(stock_ids)
        logger.info(f"Unique stocks in final dataset: {len(unique_ids)}")
    else:
        logger.warning("Preprocessing finished, but no data was generated.")
        
    return X, y, stock_ids, scalers, adj_matrix