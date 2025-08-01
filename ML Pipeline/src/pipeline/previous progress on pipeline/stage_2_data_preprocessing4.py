# src/pipeline/stage_2_data_preprocessing.py
"""
This module provides a unified data preprocessing pipeline that can operate in
two distinct modes, controlled by config.PREDICTION_MODE:

1.  "POINT" Mode: Prepares multivariate data for price point prediction.
    - For TCN/MLP, it creates individual sequences per stock.
    - For Graph-based models, it creates graph-structured sequences.
2.  "TREND" Mode: Prepares data for trend line prediction using a highly
    efficient, parallelized segmentation algorithm (`ruptures`).
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
# HELPER FUNCTIONS FOR "TREND" MODE
# =============================================================================
def _segment_one_stock(args):
    stock_id, price_values, avg_points, window_size = args
    num_data_points = len(price_values)
    min_required_points = (window_size + 1) * avg_points
    if num_data_points < min_required_points: return None
    num_segments = int(num_data_points / avg_points)
    try:
        algo = rpt.Dynp(model="l2", min_size=1).fit(price_values)
        breakpoints = algo.predict(n_bkps=num_segments - 1)
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
        return (stock_id, np.array(slopes), np.array(durations))
    except Exception: return None

def _process_stock_trend_sequences(args):
    stock_id, trends_data, slope_scaler, duration_scaler, window_size = args
    try:
        slopes = np.array(trends_data['slopes']).reshape(-1, 1)
        durations = np.array(trends_data['durations']).reshape(-1, 1)
        slope_scaled = slope_scaler.transform(slopes)
        duration_scaled = duration_scaler.transform(durations)
        scaled_trends = np.column_stack([slope_scaled.flatten(), duration_scaled.flatten()])
        X, y = [], []
        for i in range(len(scaled_trends) - window_size):
            X.append(scaled_trends[i:(i + window_size)])
            y.append(scaled_trends[i + window_size])
        if len(X) > 0: return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        else: return stock_id, None, None
    except Exception: return stock_id, None, None

def _segment_one_stock_with_dates(args):
    """Worker function to segment a stock's time series and return trends with dates."""
    stock_id, stock_df, target_column, avg_points = args
    price_values = stock_df[target_column].values
    dates = stock_df['Date'].values
    num_data_points = len(price_values)
    
    if num_data_points < avg_points * 2:
        return stock_id, []

    num_segments = int(num_data_points / avg_points)
    if num_segments < 1:
        return stock_id, []

    try:
        algo = rpt.Dynp(model="l2", min_size=1).fit(price_values)
        breakpoints = algo.predict(n_bkps=num_segments - 1)
        all_bps = [0] + breakpoints + [num_data_points]

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
        logger.error(f"Error segmenting stock {stock_id}: {e}")
        return stock_id, []

# =============================================================================
# MODE-SPECIFIC PREPROCESSING LOGIC
# =============================================================================

def _run_trend_prediction_for_graph(df, config):
    """Prepares trend data in a graph-compatible format for GraphWaveNet."""
    print_header("Stage 2: Data Preprocessing (TREND PREDICTION for GraphWaveNet)")
    cleaned_df = _clean_data(df, config)
    
    logger.info("Segmenting all stocks to extract trend data with dates...")
    task_args = [
        (sid, sdf, config.TARGET_COLUMN, config.AVG_POINTS_PER_TREND)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(_segment_one_stock_with_dates, task_args), total=len(task_args), desc="Segmenting Trends"))
    
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
    logger.info(f"Trend adjacency matrix created with shape: {adj_matrix_tensor.shape}")

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
    in_win = config.TREND_INPUT_WINDOW_SIZE
    out_win = 1 
    total_len = in_win + out_win

    for i in range(len(graph_data) - total_len + 1):
        X.append(graph_data[i : i + in_win, :, :])
        y.append(graph_data[i + in_win, :, :])
    
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)
    
    logger.info(f"Graph-based trend data created. X shape: {X_np.shape}, y shape: {y_np.shape}")
    
    return X_np, y_np, stock_ids, scalers, adj_matrix_tensor

def _run_trend_prediction_parallel(df, config):
    """Executes the parallelized preprocessing pipeline for trend prediction for non-graph models."""
    print_header("Stage 2: Data Preprocessing (TREND PREDICTION - PARALLEL MODE)")
    
    total_start_time = time.time()
    logger.info("=== STARTING TREND PREDICTION PREPROCESSING ===")
    
    cleaned_df = _clean_data(df, config)
    logger.info(f"Cleaned dataframe shape: {cleaned_df.shape}, Unique stocks: {cleaned_df['StockID'].nunique()}")
    logger.info(f"Target for segmentation: {config.TARGET_COLUMN}, Avg points/trend: {config.AVG_POINTS_PER_TREND}, Window: {config.TREND_INPUT_WINDOW_SIZE}")

    logger.info("Preparing training data subset for scaler fitting...")
    train_data_for_scalers_list = []
    for _, stock_df in cleaned_df.groupby('StockID'):
        train_size = int(len(stock_df) * config.TRAIN_SPLIT)
        if train_size > config.AVG_POINTS_PER_TREND:
            train_data_for_scalers_list.append(stock_df.iloc[:train_size])
    
    if not train_data_for_scalers_list:
        raise ValueError("Not enough data in any stock to fit trend scalers after cleaning and splitting.")
        
    train_df_for_scalers = pd.concat(train_data_for_scalers_list)

    try:
        num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        num_cores = cpu_count() or 1
    logger.info(f"Using {num_cores} CPU cores for parallel processing.")

    train_task_args = [
        (sid, sdf[config.TARGET_COLUMN].values.astype(np.float32), config.AVG_POINTS_PER_TREND, config.TREND_INPUT_WINDOW_SIZE)
        for sid, sdf in train_df_for_scalers.groupby('StockID')
    ]

    logger.info(f"Segmenting training data portion ({len(train_task_args)} stocks) to fit scalers...")
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_segment_one_stock, train_task_args), total=len(train_task_args), desc="Fitting Scalers")
        train_results = list(pbar)

    train_slopes, train_durations = [], []
    for res in train_results:
        if res:
            _, slopes, durations = res
            train_slopes.extend(slopes)
            train_durations.extend(durations)

    if not train_slopes:
        raise ValueError("Could not generate any trends from the training data portion to fit scalers.")

    logger.info("Fitting global scalers for slope and duration on training data...")
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(train_slopes).reshape(-1, 1))
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(train_durations).reshape(-1, 1))
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}
    logger.info("Scalers created successfully from training data.")

    logger.info("Segmenting full dataset and creating sequences...")
    
    full_task_args = [
        (sid, sdf[config.TARGET_COLUMN].values.astype(np.float32), config.AVG_POINTS_PER_TREND, config.TREND_INPUT_WINDOW_SIZE)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_segment_one_stock, full_task_args), total=len(full_task_args), desc="Segmenting Stocks")
        full_results = list(pbar)
    
    trends_by_stock = {}
    for res in full_results:
        if res:
            stock_id, slopes, durations = res
            trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations}
    
    if not trends_by_stock:
        raise ValueError("Segmentation of the full dataset failed to produce any trends.")

    seq_task_args = [(sid, data, scalers['slope'], scalers['duration'], config.TREND_INPUT_WINDOW_SIZE) 
                     for sid, data in trends_by_stock.items()]
    
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_process_stock_trend_sequences, seq_task_args), 
                   total=len(seq_task_args), desc="Creating Sequences")
        seq_results = list(pbar)

    all_X, all_y, all_stock_ids = [], [], []
    for stock_id, X_stock, y_stock in seq_results:
        if X_stock is not None and y_stock is not None:
            all_X.append(X_stock)
            all_y.append(y_stock)
            all_stock_ids.extend([stock_id] * len(X_stock))
    
    if not all_X:
        logger.error("No trend sequences could be created.")
        raise ValueError("Sequencing failed: No trend sequences could be created.")
    
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_stock_ids = np.array(all_stock_ids)
    
    total_time = time.time() - total_start_time
    logger.info("=== TREND PREDICTION PREPROCESSING COMPLETE ===")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Final X shape: {final_X.shape}, Final y shape: {final_y.shape}")
    
    return final_X, final_y, final_stock_ids, scalers, None


def _preprocess_for_graph_model(cleaned_df, config):
    """
    Pivots, scales, and creates sequences from the data for graph-based models.
    """
    logger.info("Starting graph-specific data preprocessing...")

    stock_ids_ordered = sorted(cleaned_df['StockID'].unique())
    num_nodes = len(stock_ids_ordered)
    logger.info(f"Processing data for {num_nodes} nodes in a fixed order.")

    scaled_df = cleaned_df.copy()
    scalers = {}
    for stock_id in stock_ids_ordered:
        stock_mask = scaled_df['StockID'] == stock_id
        stock_data = scaled_df.loc[stock_mask, config.FEATURE_COLUMNS]
        
        train_size = int(len(stock_data) * config.TRAIN_SPLIT)
        if train_size < 2:
            raise ValueError(f"Stock '{stock_id}' has insufficient data ({len(stock_data)} rows) to create a training split for the scaler.")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(stock_data.iloc[:train_size])
        scalers[stock_id] = scaler
        
        scaled_df.loc[stock_mask, config.FEATURE_COLUMNS] = scaler.transform(stock_data)

    pivoted_df = scaled_df.pivot(index='Date', columns='StockID', values=config.FEATURE_COLUMNS)
    
    pivoted_df = pivoted_df.reindex(columns=pd.MultiIndex.from_product([config.FEATURE_COLUMNS, stock_ids_ordered]))
    pivoted_df.ffill(inplace=True)
    pivoted_df.bfill(inplace=True)

    if pivoted_df.isnull().values.any():
        raise ValueError("Data still contains NaNs after pivoting and filling. Check data quality.")

    num_timesteps = len(pivoted_df)
    num_features = len(config.FEATURE_COLUMNS)
    graph_data = np.zeros((num_timesteps, num_nodes, num_features))

    for i, feature in enumerate(config.FEATURE_COLUMNS):
        graph_data[:, :, i] = pivoted_df[feature].values

    X, y = [], []
    in_win, out_win = config.POINT_INPUT_WINDOW_SIZE, config.POINT_OUTPUT_WINDOW_SIZE
    total_len = in_win + out_win
    target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)

    for i in range(len(graph_data) - total_len + 1):
        X.append(graph_data[i : i + in_win, :, :])
        y_start = i + in_win
        y_end = y_start + out_win
        y.append(graph_data[y_start:y_end, :, target_col_idx])
    
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)

    if config.MODEL_TYPE != 'DSTAGNN':
        y_np = np.expand_dims(y_np, axis=-1)

    logger.info(f"Graph data shapes created: X={X_np.shape}, y={y_np.shape}")
    
    return X_np, y_np, stock_ids_ordered, scalers

def _run_point_prediction(df, config):
    """Process multivariate data for point prediction mode."""
    print_header("Stage 2: Data Preprocessing (POINT PREDICTION - MULTIVARIATE)")
    
    cleaned_df = _clean_data(df, config)
    adj_matrix = None
    
    # --- START OF FIX ---
    # Check if the model is a graph model and run the appropriate preprocessing
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        X, y, stock_ids, scalers = _preprocess_for_graph_model(cleaned_df, config)
        adj_matrix = create_adjacency_matrix(cleaned_df, config.TARGET_COLUMN, threshold=0.75)
    else:
        # Original logic for TCN, MLP
        logger.info("Using standard sequence creation for TCN/MLP model.")
        all_X, all_y, all_stock_ids, scalers = [], [], [], {}
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
        
        grouped = cleaned_df.groupby('StockID')
        for stock_id, stock_df in tqdm(grouped, desc="Processing Stocks for Point Prediction"):
            train_size = int(len(stock_df) * config.TRAIN_SPLIT)
            if train_size < 2: continue
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(stock_df.iloc[:train_size][config.FEATURE_COLUMNS])
            scalers[stock_id] = scaler
            
            scaled_data = scaler.transform(stock_df[config.FEATURE_COLUMNS])
            
            X_stock, y_stock = [], []
            total_len = config.POINT_INPUT_WINDOW_SIZE + config.POINT_OUTPUT_WINDOW_SIZE
            for i in range(len(scaled_data) - total_len + 1):
                X_stock.append(scaled_data[i : i + config.POINT_INPUT_WINDOW_SIZE, :])
                y_start, y_end = i + config.POINT_INPUT_WINDOW_SIZE, i + total_len
                y_stock.append(scaled_data[y_start:y_end, target_col_idx])

            if len(X_stock) > 0:
                all_X.append(np.array(X_stock))
                y_stock_np = np.array(y_stock)
                all_y.append(y_stock_np.reshape(y_stock_np.shape[0], y_stock_np.shape[1], 1))
                all_stock_ids.extend([stock_id] * len(X_stock))
        
        if not all_X: raise ValueError("No sequences could be created.")
        
        X = np.concatenate(all_X, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.float32)
        stock_ids = np.array(all_stock_ids)
        logger.info(f"Standard data shapes created: X={X.shape}, y={y.shape}")
    # --- END OF FIX ---

    return X, y, stock_ids, scalers, adj_matrix

# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def run(df, config):
    """Main entry point for data preprocessing."""
    logger.info(f"Starting preprocessing with mode: {config.PREDICTION_MODE}")
    
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
        logger.info(f"Unique stocks in final dataset: {len(np.unique(stock_ids))}")
    else:
        logger.warning("Preprocessing finished, but no data was generated.")
        
    return X, y, stock_ids, scalers, adj_matrix
