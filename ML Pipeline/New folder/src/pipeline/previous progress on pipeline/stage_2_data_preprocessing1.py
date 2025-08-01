# src/pipeline/stage_2_data_preprocessing.py
"""
This module provides a unified data preprocessing pipeline that can operate in
two distinct modes, controlled by `config.PREDICTION_MODE`:

1.  "POINT" Mode: Prepares data for price point prediction. This version is
    optimized with vectorized sequence creation.

2.  "TREND" Mode: Prepares data for trend line prediction using a highly
    efficient, parallelized segmentation algorithm (`ruptures`).
    - Includes a sequential version for debugging purposes.
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

# The 'ruptures' library is a fast and robust alternative to 'pwlf' for this task.
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

def create_sequences_vectorized(data: np.ndarray, input_len: int, output_len: int):
    """
    Creates overlapping sequences using NumPy stride tricks for maximum efficiency.
    This is significantly faster than a Python for-loop.

    Args:
        data (np.ndarray): The 1D array of scaled time-series data.
        input_len (int): The length of the input window (X).
        output_len (int): The length of the output window (y).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the X and y arrays.
    """
    from numpy.lib.stride_tricks import as_strided

    total_len = input_len + output_len
    num_sequences = len(data) - total_len + 1

    if num_sequences <= 0:
        return np.array([]), np.array([])

    itemsize = data.itemsize
    # Create the input sequences (X)
    X_strided = as_strided(data, shape=(num_sequences, input_len), strides=(itemsize, itemsize))
    
    # Create the output sequences (y) by starting from the end of the first input window
    y_start_index = input_len
    y_data = data[y_start_index:]
    y_strided = as_strided(y_data, shape=(num_sequences, output_len), strides=(itemsize, itemsize))
    
    # Return copies to ensure memory safety, as strided arrays are just views
    return X_strided.copy(), y_strided.copy()


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
    
    logger.info(f"[Worker PID: {worker_pid}] Starting segmentation for stock: {stock_id} ({len(price_values)} points)")

    num_data_points = len(price_values)
    # Minimum points needed to create at least one full window of trends
    min_required_points = (window_size + 1) * avg_points
    
    if num_data_points < min_required_points:
        logger.warning(f"[Worker PID: {worker_pid}] Stock {stock_id}: Insufficient data ({num_data_points} < {min_required_points})")
        return None
    
    num_segments = int(num_data_points / avg_points)
    
    try:
        # 1. Use ruptures for fast and robust segmentation.
        # Dynp = Dynamic Programming. "l2" is a standard least-squares cost function.
        algo = rpt.Dynp(model="l2", min_size=1).fit(price_values)
        # Predict the locations of n-1 breakpoints to get n segments.
        breakpoints = algo.predict(n_bkps=num_segments - 1)
        
        # Add the start and end of the series to the list of breakpoints
        all_bps = [0] + breakpoints

        # 2. Efficiently calculate slopes and durations for each segment
        slopes = []
        durations = []
        for start_idx, end_idx in zip(all_bps[:-1], all_bps[1:]):
            segment_len = end_idx - start_idx
            if segment_len > 1:
                # Use NumPy's polyfit for a fast linear regression to find the slope
                y_segment = price_values[start_idx:end_idx]
                x_segment = np.arange(segment_len)
                slope = np.polyfit(x_segment, y_segment, 1)[0]
                slopes.append(slope)
                durations.append(segment_len)

        processing_time = time.time() - start_time
        logger.info(f"[Worker PID: {worker_pid}] Completed stock: {stock_id} in {processing_time:.2f}s. Found {len(slopes)} trends.")
        
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
            logger.info(f"Stock {stock_id}: Generated {len(X)} sequences in {processing_time:.2f}s")
            return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        else:
            logger.warning(f"Stock {stock_id}: No sequences generated (insufficient trend data)")
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
    logger.info(f"Input dataframe shape: {df.shape}, Unique stocks: {df['StockID'].nunique()}")
    logger.info(f"Target: {config.TARGET_COLUMN}, Avg points/trend: {config.AVG_POINTS_PER_TREND}, Window: {config.TREND_INPUT_WINDOW_SIZE}")
    
    # --- OPTIMIZED CORE DETECTION ---
    # Correctly detect cores allocated by SLURM, otherwise use all available.
    try:
        num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        logger.info(f"SLURM environment detected (cpus-per-task). Using {num_cores} allocated CPU cores.")
    except KeyError:
        try:
            num_cores = int(os.environ['SLURM_NTASKS'])
            logger.info(f"SLURM environment detected (ntasks). Using {num_cores} allocated CPU cores.")
        except KeyError:
            num_cores = cpu_count() or 1
            logger.info(f"No SLURM environment detected. Using {num_cores} available CPU cores.")

    # Prepare data for parallel processing
    grouped = df.groupby('StockID')
    task_args = [
        (sid, sdf[config.TARGET_COLUMN].values.astype(np.float32), config.AVG_POINTS_PER_TREND, config.TREND_INPUT_WINDOW_SIZE)
        for sid, sdf in grouped
    ]
    logger.info(f"Prepared {len(task_args)} stocks for parallel segmentation.")

    # Parallel segmentation
    segmentation_start = time.time()
    logger.info(f"Starting parallel segmentation on {num_cores} cores...")
    
    results = []
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_segment_one_stock, task_args), total=len(task_args), desc="Segmenting Stocks")
        results = list(pbar)
    
    segmentation_time = time.time() - segmentation_start
    logger.info(f"Parallel segmentation completed in {segmentation_time:.2f}s")

    # Process results
    all_slopes, all_durations, trends_by_stock = [], [], {}
    for res in results:
        if res:
            stock_id, slopes, durations = res
            all_slopes.extend(slopes)
            all_durations.extend(durations)
            trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations}
    
    successful_stocks = len(trends_by_stock)
    logger.info(f"Successfully segmented {successful_stocks}/{len(results)} stocks.")
    
    if not trends_by_stock:
        logger.error("No stocks had enough data for trend segmentation.")
        raise ValueError("Preprocessing failed: No stocks had enough data for trend segmentation.")
    
    # Fit scalers
    logger.info("Fitting global scalers for slope and duration...")
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(all_slopes).reshape(-1, 1))
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(all_durations).reshape(-1, 1))
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}
    logger.info("Scalers created successfully.")

    # Create sequences in parallel
    logger.info("Creating input/output sequences from trends in parallel...")
    seq_task_args = [(sid, data, scalers['slope'], scalers['duration'], config.TREND_INPUT_WINDOW_SIZE) 
                     for sid, data in trends_by_stock.items()]
    
    with Pool(processes=num_cores) as pool:
        pbar = tqdm(pool.imap(_process_stock_trend_sequences, seq_task_args), 
                   total=len(seq_task_args), desc="Creating Sequences")
        seq_results = list(pbar)

    # Combine results
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

def _run_trend_prediction_sequential(df, config):
    """Executes a non-parallel version for debugging."""
    print_header("Stage 2: Data Preprocessing (TREND PREDICTION - SEQUENTIAL DEBUG MODE)")
    logger.info("=== STARTING SEQUENTIAL DEBUG MODE ===")
    
    grouped = df.groupby('StockID')
    for stock_id, stock_df in tqdm(grouped, desc="Segmenting Stocks Sequentially"):
        args = (stock_id, stock_df[config.TARGET_COLUMN].values, config.AVG_POINTS_PER_TREND, config.TREND_INPUT_WINDOW_SIZE)
        _segment_one_stock(args) # We only care about the logs, not the result
        
    logger.warning("Sequential debug run finished. This mode does not produce output data.")
    return np.array([]), np.array([]), np.array([]), {}

def _run_point_prediction(df, config):
    """Process data for point prediction mode using vectorized operations."""
    print_header("Stage 2: Data Preprocessing (POINT PREDICTION MODE)")
    
    total_start_time = time.time()
    logger.info("=== STARTING POINT PREDICTION PREPROCESSING ===")
    logger.info(f"Input dataframe shape: {df.shape}, Unique stocks: {df['StockID'].nunique()}")
    logger.info(f"Window In/Out: {config.POINT_INPUT_WINDOW_SIZE}/{config.POINT_OUTPUT_WINDOW_SIZE}")
    
    all_X, all_y, all_stock_ids, scalers = [], [], [], {}
    
    grouped = df.groupby('StockID')
    for stock_id, stock_df in tqdm(grouped, desc="Processing Stocks for Point Prediction"):
        stock_df = stock_df.copy()
        
        # Handle missing values
        if stock_df[config.TARGET_COLUMN].isnull().any():
            stock_df.fillna(method='ffill', inplace=True)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_df[[config.TARGET_COLUMN]])
        scalers[stock_id] = scaler
        
        # --- VECTORIZED SEQUENCE CREATION ---
        X_stock, y_stock = create_sequences_vectorized(
            scaled_data.flatten(), 
            config.POINT_INPUT_WINDOW_SIZE, 
            config.POINT_OUTPUT_WINDOW_SIZE
        )
        
        if X_stock.size > 0:
            # Reshape to (samples, timesteps, features) for typical deep learning models
            all_X.append(X_stock.reshape(X_stock.shape[0], X_stock.shape[1], 1))
            all_y.append(y_stock.reshape(y_stock.shape[0], y_stock.shape[1], 1))
            all_stock_ids.extend([stock_id] * len(X_stock))
        else:
            logger.warning(f"Stock {stock_id}: No sequences generated (insufficient data)")
    
    if not all_X:
        logger.error("No sequences could be created for point prediction.")
        raise ValueError("Preprocessing failed: No sequences could be created for point prediction.")
    
    # Concatenate and set data type for efficiency
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
        # --- FIX: Use getattr for optional config attributes ---
        # This safely checks if 'DEBUG_SEQUENTIAL' exists in the config module.
        if getattr(config, 'DEBUG_SEQUENTIAL', False):
            logger.info("Running sequential trend prediction preprocessing (DEBUG)")
            X, y, stock_ids, scalers = _run_trend_prediction_sequential(df, config)
        else:
            logger.info("Running parallel trend prediction preprocessing")
            X, y, stock_ids, scalers = _run_trend_prediction_parallel(df, config)

    elif config.PREDICTION_MODE == "POINT":
        X, y, stock_ids, scalers = _run_point_prediction(df, config)
    else:
        msg = f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'. Must be 'POINT' or 'TREND'."
        logger.error(msg)
        raise ValueError(msg)
    
    # Final summary
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
