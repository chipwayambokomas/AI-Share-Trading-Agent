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
# DATA SPLITTING AND CLEANING (MODIFIED LOGIC)
# =============================================================================

# --- MODIFICATION START: New function for chronological splitting ---
def _chronological_split(df, config):
    """
    Splits the dataframe chronologically into train, validation, and test sets
    for each stock.
    """
    #logger.info("Performing 60:20:20 chronological split for each stock...")
    train_dfs, val_dfs, test_dfs = [], [], []

    for stock_id, stock_df in df.groupby('StockID'):
        if len(stock_df) < 5: # Ensure there's enough data to split
            #logger.warning(f"Skipping stock '{stock_id}': Insufficient data for splitting (has {len(stock_df)} rows).")
            continue

        n = len(stock_df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_dfs.append(stock_df.iloc[:train_end])
        val_dfs.append(stock_df.iloc[train_end:val_end])
        test_dfs.append(stock_df.iloc[val_end:])

    if not train_dfs:
        raise ValueError("No data available after attempting to split. Check data lengths.")

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    #logger.info(f"Split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} rows.")
    return train_df, val_df, test_df
# --- MODIFICATION END ---

def _clean_data(df: pd.DataFrame, config, set_name: str) -> pd.DataFrame:
    """
    Cleans the raw dataframe by handling missing and erroneous values.
    """
    #logger.info(f"Starting data cleaning process for {set_name} set...")
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
            #logger.warning(f"({set_name}) Excluding stock '{stock_id}': {missing_pct:.1%} missing target data.")
            continue
            
        cleaned_stock_df = stock_df.copy()
        cleaned_stock_df[config.FEATURE_COLUMNS] = cleaned_stock_df[config.FEATURE_COLUMNS].fillna(method='ffill')
        cleaned_stock_df.dropna(subset=config.FEATURE_COLUMNS, inplace=True)
        
        if cleaned_stock_df.empty:
            #logger.warning(f"({set_name}) Excluding stock '{stock_id}': Empty after cleaning.")
            continue
            
        cleaned_stocks.append(cleaned_stock_df)
        
    if not cleaned_stocks:
        #logger.warning(f"No stocks remained for the {set_name} set after data cleaning.")
        return pd.DataFrame()
        
    final_df = pd.concat(cleaned_stocks, ignore_index=True)
    final_stock_count = final_df['StockID'].nunique()
    
    #logger.info(f"Data cleaning for {set_name} complete. Kept {final_stock_count}/{initial_stock_count} stocks.")
    return final_df

# =============================================================================
# HELPER FUNCTIONS FOR "TREND" MODE (Unchanged)
# =============================================================================
def _segment_one_stock(args):
    """
    Worker function to segment a SINGLE stock's time series into trends.
    This version uses the 'ruptures' library for efficient changepoint detection.
    """
    stock_id, price_values, avg_points, window_size = args
    worker_pid = os.getpid()
    num_data_points = len(price_values)
    min_required_points = (window_size + 1) * avg_points
    if num_data_points < min_required_points:
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
        return (stock_id, np.array(slopes), np.array(durations))
    except Exception as e:
        #logger.error(f"[Worker PID: {worker_pid}] Stock {stock_id}: Error during segmentation: {str(e)}")
        return None

def _process_stock_trend_sequences(args):
    """Process trend sequences for a single stock."""
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
        if len(X) > 0:
            return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        else:
            return stock_id, None, None
    except Exception as e:
        #logger.error(f"Stock {stock_id}: Error processing sequences: {str(e)}")
        return stock_id, None, None

# --- MODIFICATION START: Entire point prediction logic is refactored ---
# =============================================================================
# MODE-SPECIFIC PREPROCESSING LOGIC (MODIFIED)
# =============================================================================
def _create_sequences(df, scaler, config, target_col_idx):
    """Helper function to create sequences from a scaled dataframe."""
    all_X, all_y, all_stock_ids = [], [], []
    
    scaled_data = scaler.transform(df[config.FEATURE_COLUMNS])
    # Re-attach StockID to the scaled data for grouping
    scaled_df = pd.DataFrame(scaled_data, columns=config.FEATURE_COLUMNS)
    scaled_df['StockID'] = df['StockID'].values

    grouped = scaled_df.groupby('StockID')
    for stock_id, stock_df_scaled in tqdm(grouped, desc="Creating Sequences", leave=False):
        # --- MODIFICATION START: Show which stock is in the sliding window ---
        #logger.info(f"Creating sequences (sliding window) for stock: '{stock_id}'")
        # --- MODIFICATION END ---
        
        current_scaled_data = stock_df_scaled[config.FEATURE_COLUMNS].values
        X_stock, y_stock = [], []
        total_len = config.POINT_INPUT_WINDOW_SIZE + config.POINT_OUTPUT_WINDOW_SIZE
        for i in range(len(current_scaled_data) - total_len + 1):
            X_stock.append(current_scaled_data[i : i + config.POINT_INPUT_WINDOW_SIZE, :])
            y_start_idx = i + config.POINT_INPUT_WINDOW_SIZE
            y_end_idx = y_start_idx + config.POINT_OUTPUT_WINDOW_SIZE
            y_stock.append(current_scaled_data[y_start_idx:y_end_idx, target_col_idx])

        if len(X_stock) > 0:
            X_stock_np = np.array(X_stock)
            y_stock_np = np.array(y_stock)
            all_X.append(X_stock_np)
            all_y.append(y_stock_np.reshape(y_stock_np.shape[0], y_stock_np.shape[1], 1))
            all_stock_ids.extend([stock_id] * len(X_stock))

    if not all_X:
        return np.array([]), np.array([]), np.array([])
        
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0), np.array(all_stock_ids)


def _run_point_prediction(train_df, val_df, test_df, config):
    """Process multivariate data for point prediction mode with new splitting and scaling."""
    #print_header("Stage 2: Data Preprocessing (POINT PREDICTION - CHRONOLOGICAL SPLIT)")
    total_start_time = time.time()

    try:
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    except ValueError:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in FEATURE_COLUMNS.")

    # 1. Fit scaler on TRAINING data only
    #logger.info("Fitting scaler on training data...")
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaler.fit(train_df[config.FEATURE_COLUMNS])
    
    # 2. Create training sequences using the fitted scaler
    X_train, y_train, stock_ids_train = _create_sequences(train_df, train_scaler, config, target_col_idx)

    # 3. Create validation sequences using the SAME scaler
    #logger.info("Transforming validation data and creating sequences...")
    X_val, y_val, stock_ids_val = _create_sequences(val_df, train_scaler, config, target_col_idx)

    # 4. Fit a SEPARATE scaler for test data and create sequences
    #logger.info("Fitting separate scaler on test data and creating sequences...")
    test_scaler = MinMaxScaler(feature_range=(0, 1))
    test_scaler.fit(test_df[config.FEATURE_COLUMNS])
    X_test, y_test, stock_ids_test = _create_sequences(test_df, test_scaler, config, target_col_idx)
    
    scalers = {'train_val': train_scaler, 'test': test_scaler}

    total_time = time.time() - total_start_time
    logger.info("=== POINT PREDICTION PREPROCESSING COMPLETE ===")
    #logger.info(f"Total processing time: {total_time:.2f}s")
    #logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    #logger.info(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    #logger.info(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    
    # Note: The return signature is changed to reflect the new partitioned structure.
    # The next pipeline stage (stage_3) would need to be adapted to handle this.
    return (X_train, y_train, stock_ids_train), \
           (X_val, y_val, stock_ids_val), \
           (X_test, y_test, stock_ids_test), \
           scalers
# --- MODIFICATION END ---

def _run_trend_prediction_parallel(df, config):
    """
    This function is left largely intact as the user's scaling request is not
    directly applicable to the trend mode's methodology (which scales slopes
    and durations, not raw features). It will, however, use the new chronological
    split for defining its scaler training set.
    """
    print_header("Stage 2: Data Preprocessing (TREND PREDICTION - PARALLEL MODE)")
    # This function would also need to be refactored to use the pre-split data
    # For now, it will use the original logic but only fit scalers on the train split.
    # A full refactor would be similar in complexity to the point prediction one.
    
    # For demonstration, we will show the original logic but highlight where
    # the pre-split `train_df` would be used.
    
    # 1. Clean the full dataset (as it was originally)
    cleaned_df = _clean_data(df, config, "FULL")
    
    # 2. Define the training set portion for fitting scalers
    logger.info("Preparing training data subset for scaler fitting...")
    # --- MODIFICATION: In a full refactor, you would use the `train_df` from the chronological split here.
    train_df_for_scalers_list = []
    for _, stock_df in cleaned_df.groupby('StockID'):
        train_size = int(len(stock_df) * 0.6) # Using 60% split ratio
        if train_size > config.AVG_POINTS_PER_TREND:
            train_df_for_scalers_list.append(stock_df.iloc[:train_size])
    train_df_for_scalers = pd.concat(train_df_for_scalers_list)
    # The rest of the function would proceed as original, but this section
    # shows how the conceptual change would be integrated.
    logger.warning("Trend prediction mode has not been fully refactored. It follows its original logic for now.")
    
    # ... (rest of the original _run_trend_prediction_parallel function would follow)
    # This is a placeholder to prevent breaking the script. A full implementation
    # would require rewriting the parallel processing logic to handle three separate dataframes.
    raise NotImplementedError("The 'TREND' mode has not been fully refactored for the new data splitting and scaling logic.")


# =============================================================================
# PUBLIC ENTRY POINT (MODIFIED)
# =============================================================================
def run(df, config):
    """Main entry point for data preprocessing."""
    logger.info(f"Starting preprocessing with mode: {config.PREDICTION_MODE}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- MODIFICATION START: Call splitting and cleaning at the start ---
    # 1. Chronologically split the raw data
    train_df_raw, val_df_raw, test_df_raw = _chronological_split(df, config)
    
    # 2. Clean each dataset separately
    train_df = _clean_data(train_df_raw, config, "Train")
    val_df = _clean_data(val_df_raw, config, "Validation")
    test_df = _clean_data(test_df_raw, config, "Test")
    # --- MODIFICATION END ---

    if config.PREDICTION_MODE == "TREND":
        # Note: The trend function is not fully implemented with the new logic.
        return _run_trend_prediction_parallel(df, config)

    elif config.PREDICTION_MODE == "POINT":
        # --- MODIFICATION START: Call point prediction with split data ---
        (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test), scalers = \
            _run_point_prediction(train_df, val_df, test_df, config)
        
        logger.info("=== PREPROCESSING SUMMARY ===")
        logger.info(f"Total sequences generated: {len(X_train) + len(X_val) + len(X_test)}")
        logger.info(f"Unique stocks in Train: {len(np.unique(s_train))}, Val: {len(np.unique(s_val))}, Test: {len(np.unique(s_test))}")
        logger.info("Preprocessing completed successfully!")
        
        # This new return signature will require changes in stage_3_data_partitioning
        return (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test), scalers
        # --- MODIFICATION END ---
        
    else:
        msg = f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'. Must be 'POINT' or 'TREND'."
        logger.error(msg)
        raise ValueError(msg)