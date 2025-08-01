# src/pipeline/stage_2_data_preprocessing.py
"""
This module provides a unified data preprocessing pipeline that can operate in
two distinct modes, controlled by `config.PREDICTION_MODE`:

1.  "POINT" Mode: Prepares multivariate data for price point prediction.

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
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import ruptures as rpt

from src.utils import print_header

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
    - Excludes stocks with >10% missing data in the target column.
    - Fills minor missing data in all feature columns using forward-fill.
    - Treats non-positive values in all feature columns as erroneous (NaN).
    """
    logger.info("Starting data cleaning process...")
    cleaned_stocks = []
    
    # --- FIX: Apply cleaning to all feature columns ---
    for col in config.FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] <= 0, col] = np.nan
    
    initial_stock_count = df['StockID'].nunique()
    
    for stock_id, stock_df in df.groupby('StockID'):
        original_len = len(stock_df)
        
        # Calculate missing data percentage based on the target column
        missing_count = stock_df[config.TARGET_COLUMN].isnull().sum()
        missing_pct = missing_count / original_len if original_len > 0 else 0
        
        if missing_pct > 0.10:
            logger.warning(f"Excluding stock '{stock_id}': {missing_pct:.1%} missing target data ({missing_count}/{original_len} rows).")
            continue
            
        cleaned_stock_df = stock_df.copy()
        # --- FIX: Forward-fill all feature columns ---
        cleaned_stock_df[config.FEATURE_COLUMNS] = cleaned_stock_df[config.FEATURE_COLUMNS].fillna(method='ffill')
        
        # After ffill, drop any rows that still have NaNs (e.g., at the start)
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
    """
    Worker function to segment a SINGLE stock's time series into trends.
    This version uses the 'ruptures' library for efficient changepoint detection.
    """
    stock_id, price_values, avg_points, window_size = args
    
    start_time = time.time()
    worker_pid = os.getpid()
    
    logger.debug(f"[Worker PID: {worker_pid}] Starting segmentation for stock: {stock_id} ({len(price_values)} points)")

    num_data_points = len(price_values)
    min_required_points = (window_size + 1) * avg_points
    
    if num_data_points < min_required_points:
        logger.debug(f"[Worker PID: {worker_pid}] Stock {stock_id}: Insufficient data ({num_data_points} < {min_required_points})")
        return None
    
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

        processing_time = time.time() - start_time
        logger.debug(f"[Worker PID: {worker_pid}] Completed stock: {stock_id} in {processing_time:.2f}s. Found {len(slopes)} trends.")
        
        return (stock_id, np.array(slopes), np.array(durations))
        
    except Exception as e:
        logger.error(f"[Worker PID: {worker_pid}] Stock {stock_id}: Error during segmentation: {str(e)}")
        return None

def _process_stock_trend_sequences(args):
    """Process trend sequences for a single stock."""
    stock_id, trends_data, slope_scaler, duration_scaler, window_size = args
    
    start_time = time.time()
    
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
        
        processing_time = time.time() - start_time
        
        if len(X) > 0:
            logger.debug(f"Stock {stock_id}: Generated {len(X)} sequences in {processing_time:.2f}s")
            return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        else:
            logger.debug(f"Stock {stock_id}: No sequences generated (insufficient trend data)")
            return stock_id, None, None
            
    except Exception as e:
        logger.error(f"Stock {stock_id}: Error processing sequences: {str(e)}")
        return stock_id, None, None

# =============================================================================
# MODE-SPECIFIC PREPROCESSING LOGIC
# =============================================================================
def _run_trend_prediction_parallel(df, config):
    """Executes the parallelized preprocessing pipeline for trend prediction."""
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
    
    return final_X, final_y, final_stock_ids, scalers

def _run_point_prediction(df, config):
    """Process multivariate data for point prediction mode."""
    print_header("Stage 2: Data Preprocessing (POINT PREDICTION - MULTIVARIATE)")
    
    total_start_time = time.time()
    logger.info("=== STARTING POINT PREDICTION PREPROCESSING ===")
    
    cleaned_df = _clean_data(df, config)
    logger.info(f"Cleaned dataframe shape: {cleaned_df.shape}, Unique stocks: {cleaned_df['StockID'].nunique()}")
    logger.info(f"Features: {config.FEATURE_COLUMNS}")
    logger.info(f"Window In/Out: {config.POINT_INPUT_WINDOW_SIZE}/{config.POINT_OUTPUT_WINDOW_SIZE}")
    
    all_X, all_y, all_stock_ids, scalers = [], [], [], {}
    
    # Get the index of the target column within the feature list
    try:
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    except ValueError:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in FEATURE_COLUMNS list in config.")

    grouped = cleaned_df.groupby('StockID')
    for stock_id, stock_df in tqdm(grouped, desc="Processing Stocks for Point Prediction"):
        
        train_size = int(len(stock_df) * config.TRAIN_SPLIT)
        if train_size < 2:
            logger.warning(f"Skipping stock '{stock_id}': Not enough data for training scaler ({train_size} points).")
            continue
        train_data_for_scaler = stock_df.iloc[:train_size]

        # Fit scaler on all feature columns of the training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data_for_scaler[config.FEATURE_COLUMNS])
        scalers[stock_id] = scaler
        
        # Transform the entire stock's data
        scaled_data = scaler.transform(stock_df[config.FEATURE_COLUMNS])
        
        # --- FIX: Create multivariate sequences using a standard loop ---
        X_stock, y_stock = [], []
        total_len = config.POINT_INPUT_WINDOW_SIZE + config.POINT_OUTPUT_WINDOW_SIZE
        for i in range(len(scaled_data) - total_len + 1):
            # Input sequence (X) contains all features
            X_stock.append(scaled_data[i : i + config.POINT_INPUT_WINDOW_SIZE, :])
            
            # Output sequence (y) contains only the target feature
            y_start_idx = i + config.POINT_INPUT_WINDOW_SIZE
            y_end_idx = y_start_idx + config.POINT_OUTPUT_WINDOW_SIZE
            y_stock.append(scaled_data[y_start_idx:y_end_idx, target_col_idx])

        if len(X_stock) > 0:
            # Convert to numpy arrays
            X_stock_np = np.array(X_stock)
            y_stock_np = np.array(y_stock)
            
            all_X.append(X_stock_np)
            # Reshape y to be (samples, output_window, 1) for consistency
            all_y.append(y_stock_np.reshape(y_stock_np.shape[0], y_stock_np.shape[1], 1))
            all_stock_ids.extend([stock_id] * len(X_stock))
        else:
            logger.warning(f"Stock {stock_id}: No sequences generated (insufficient data for windowing)")
    
    if not all_X:
        logger.error("No sequences could be created for point prediction.")
        raise ValueError("Preprocessing failed: No sequences could be created for point prediction.")
    
    final_X = np.concatenate(all_X, axis=0).astype(np.float32)
    final_y = np.concatenate(all_y, axis=0).astype(np.float32)
    final_stock_ids = np.array(all_stock_ids)
    
    total_time = time.time() - total_start_time
    logger.info("=== POINT PREDICTION PREPROCESSING COMPLETE ===")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Final X shape: {final_X.shape}, Final y shape: {final_y.shape}")
    
    return final_X, final_y, final_stock_ids, scalers

# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def run(df, config):
    """Main entry point for data preprocessing."""
    logger.info(f"Starting preprocessing with mode: {config.PREDICTION_MODE}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if config.PREDICTION_MODE == "TREND":
        logger.info("Running parallel trend prediction preprocessing")
        X, y, stock_ids, scalers = _run_trend_prediction_parallel(df, config)

    elif config.PREDICTION_MODE == "POINT":
        X, y, stock_ids, scalers = _run_point_prediction(df, config)
    else:
        msg = f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'. Must be 'POINT' or 'TREND'."
        logger.error(msg)
        raise ValueError(msg)
    
    if X.size > 0:
        logger.info("=== PREPROCESSING SUMMARY ===")
        logger.info(f"Total sequences generated: {len(X)}")
        logger.info(f"Input sequences shape (X): {X.shape}")
        logger.info(f"Target sequences shape (y): {y.shape}")
        logger.info(f"Unique stocks with data: {len(np.unique(stock_ids))}")
        logger.info("Preprocessing completed successfully!")
    else:
        logger.warning("Preprocessing finished, but no data was generated. Check logs for warnings.")
        
    return X, y, stock_ids, scalers