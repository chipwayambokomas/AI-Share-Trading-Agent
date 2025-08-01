# src/preprocessing/dnn_trendline_prediction.py

import os
import logging
import numpy as np
import pandas as pd
# import ruptures as rpt # No longer needed for this implementation
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils import print_header

# Configure logging
logger = logging.getLogger(__name__)

# --- Data Cleaning (Identical to previous version) ---
def _clean_data(df: pd.DataFrame, config) -> pd.DataFrame:
    """(Identical to the _clean_data function in dnn_point_prediction.py)"""
    logger.info("Starting data cleaning for DNN trend prediction...")
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
            logger.warning(f"Excluding stock '{stock_id}': {missing_pct:.1%} missing target data.")
            continue
            
        cleaned_stock_df = stock_df.copy()
        cleaned_stock_df[config.FEATURE_COLUMNS] = cleaned_stock_df[config.FEATURE_COLUMNS].ffill()
        cleaned_stock_df.dropna(subset=config.FEATURE_COLUMNS, inplace=True)
        
        if cleaned_stock_df.empty:
            logger.warning(f"Excluding stock '{stock_id}': Empty after cleaning.")
            continue
            
        cleaned_stocks.append(cleaned_stock_df)
        
    final_df = pd.concat(cleaned_stocks, ignore_index=True)
    logger.info(f"Data cleaning complete. Kept {final_df['StockID'].nunique()}/{initial_stock_count} stocks.")
    return final_df

# --- Custom Segmentation Algorithms and Helpers ---

def _create_segment_least_squares(sequence, seq_range):
    """
    Creates a line segment using linear regression (least squares).
    The segment is represented as a tuple: (start_index, intercept, slope, end_index).
    """
    start, end = seq_range
    sub_sequence = sequence[start:end+1]
    if len(sub_sequence) < 2:
        # For a single point, slope is 0, intercept is the point's value
        return (start, sub_sequence[0], 0.0, end)
    
    x = np.arange(len(sub_sequence))
    slope, intercept = np.polyfit(x, sub_sequence, 1)
    return (start, intercept, slope, end)

def _compute_sum_of_squared_error(sequence, segment):
    """
    Computes the Sum of Squared Errors (SSE) for a given segment against the sequence.
    This error metric is consistent with the L2-norm cost function.
    """
    start, intercept, slope, end = segment
    
    # Ensure indices are integers for slicing
    start, end = int(start), int(end)
    
    if start >= end:
        return 0.0
        
    sub_sequence = sequence[start:end+1]
    x = np.arange(len(sub_sequence))
    predicted_sequence = slope * x + intercept
    return np.sum((sub_sequence - predicted_sequence) ** 2)
    
def bottomupsegment1(sequence, create_segment, compute_error, max_error):
    """
    Return a list of line segments that approximate the sequence.
    The list is computed using the bottom-up technique.
    """
    # Start with the finest possible segmentation (one segment for each pair of points)
    segments = [create_segment(sequence, seq_range) for seq_range in zip(range(len(sequence)), range(1, len(sequence)))]

    if not segments:
        return []

    # Calculate the cost of merging adjacent segments
    mergesegments = [create_segment(sequence, (seg1[0], seg2[3])) for seg1, seg2 in zip(segments[:-1], segments[1:])]
    mergecosts = [compute_error(sequence, segment) for segment in mergesegments]

    while mergecosts and min(mergecosts) < max_error:
        idx = mergecosts.index(min(mergecosts))
        
        # Merge the two segments with the lowest merge cost
        segments[idx] = mergesegments[idx]
        del segments[idx+1]
        del mergesegments[idx]
        del mergecosts[idx]

        # Update the merge cost for the segment to the left of the new merged segment
        if idx > 0:
            new_mergesegment = create_segment(sequence, (segments[idx-1][0], segments[idx][3]))
            mergesegments[idx-1] = new_mergesegment
            mergecosts[idx-1] = compute_error(sequence, new_mergesegment)

        # Update the merge cost for the segment to the right of the new merged segment
        if idx < len(segments) - 1:
            new_mergesegment = create_segment(sequence, (segments[idx][0], segments[idx+1][3]))
            mergesegments.insert(idx, new_mergesegment) # Use insert instead of direct assignment
            mergecosts.insert(idx, compute_error(sequence, new_mergesegment))
            
    return segments

def bottomupsegment(sequence, create_segment, compute_error, max_error):
    """
    Return a list of line segments that approximate the sequence.
    The list is computed using the bottom-up technique.
    """
    # Start with the finest possible segmentation (one segment for each pair of points)
    segments = [create_segment(sequence, seq_range) for seq_range in zip(range(len(sequence)), range(1, len(sequence)))]

    if not segments:
        return []

    # Calculate the initial cost of merging adjacent segments
    mergesegments = [create_segment(sequence, (seg1[0], seg2[3])) for seg1, seg2 in zip(segments[:-1], segments[1:])]
    mergecosts = [compute_error(sequence, segment) for segment in mergesegments]

    while mergecosts and min(mergecosts) < max_error:
        idx = mergecosts.index(min(mergecosts))
        
        # Merge the two segments with the lowest merge cost
        segments[idx] = mergesegments[idx]
        del segments[idx+1]

        # --- START OF FIX ---
        # The previous index-based update logic was flawed. For robustness, we now
        # fully recalculate the merge costs in each iteration. This is simpler
        # and guarantees correctness, avoiding index out-of-bounds errors.
        if len(segments) < 2:
            # No more segments can be merged
            break

        mergesegments = [create_segment(sequence, (seg1[0], seg2[3])) for seg1, seg2 in zip(segments[:-1], segments[1:])]
        mergecosts = [compute_error(sequence, segment) for segment in mergesegments]
        # --- END OF FIX ---
            
    return segments

# --- Worker Function for Custom Segmentation ---

def _segment_one_stock_custom_bottom_up(args):
    """Worker function to segment a SINGLE stock's time series using the custom bottom-up algorithm."""
    stock_id, price_values, max_error = args
    if len(price_values) < 2: return None
    try:
        # Call the custom bottom-up segmentation function
        segments = bottomupsegment(
            price_values, 
            _create_segment_least_squares, 
            _compute_sum_of_squared_error, 
            max_error
        )
        
        if not segments: return None

        # Extract slopes and durations from the resulting segments
        slopes = [seg[2] for seg in segments] # slope is the 3rd element
        durations = [seg[3] - seg[0] for seg in segments] # duration = end_idx - start_idx
        
        return (stock_id, np.array(slopes), np.array(durations))
    except Exception as e:
        logger.error(f"Stock {stock_id}: Error during custom segmentation: {str(e)}")
        return None

# --- Sequence Processing (Identical to previous version) ---
def _process_stock_trend_sequences(args):
    """Process trend sequences for a single stock to create ML inputs."""
    stock_id, trends_data, slope_scaler, duration_scaler, window_size = args
    try:
        slopes = np.array(trends_data['slopes']).reshape(-1, 1)
        durations = np.array(trends_data['durations']).reshape(-1, 1)
        scaled_trends = np.column_stack([slope_scaler.transform(slopes), duration_scaler.transform(durations)])
        
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

# --- Main Execution Pipeline ---
def run(df, config):
    """Executes the parallelized preprocessing pipeline for non-graph trend models."""
    print_header("Stage 2: Preprocessing (DNN TREND PREDICTION with Custom Bottom-Up)")
    cleaned_df = _clean_data(df, config)

    logger.info("Segmenting full dataset using Custom Bottom-Up approach...")
    # NOTE: The config must now have MAX_SEGMENTATION_ERROR instead of SEGMENTATION_PENALTY
    task_args = [
        (sid, sdf[config.TARGET_COLUMN].values, config.MAX_SEGMENTATION_ERROR)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=cpu_count()) as pool:
        # Call the new worker function
        results = list(tqdm(pool.imap(_segment_one_stock_custom_bottom_up, task_args), total=len(task_args), desc="Segmenting All Stocks (Custom)"))
    
    all_slopes, all_durations, trends_by_stock = [], [], {}
    for res in results:
        if res:
            stock_id, slopes, durations = res
            all_slopes.extend(slopes)
            all_durations.extend(durations)
            trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations}

    if not trends_by_stock: raise ValueError("Segmentation failed to produce any trends.")

    logger.info("Fitting global scalers for slope and duration...")
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(all_slopes).reshape(-1, 1))
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(all_durations).reshape(-1, 1))
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}

    logger.info("Creating sequences from trends...")
    seq_args = [
        (sid, data, scalers['slope'], scalers['duration'], config.TREND_INPUT_WINDOW_SIZE) 
        for sid, data in trends_by_stock.items()
    ]
    with Pool(processes=cpu_count()) as pool:
        seq_results = list(tqdm(pool.imap(_process_stock_trend_sequences, seq_args), total=len(seq_args), desc="Creating Sequences"))

    all_X, all_y, all_stock_ids = [], [], []
    for stock_id, X_stock, y_stock in seq_results:
        if X_stock is not None and y_stock is not None:
            all_X.append(X_stock)
            all_y.append(y_stock)
            all_stock_ids.extend([stock_id] * len(X_stock))
    
    if not all_X: raise ValueError("Sequencing failed: No trend sequences could be created.")
    
    logger.info("DNN trendline prediction preprocessing complete.")
    return np.concatenate(all_X), np.concatenate(all_y), np.array(all_stock_ids), scalers, None