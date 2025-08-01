# src/preprocessing/dnn_trendline_prediction.py

import os
import logging
import numpy as np
import pandas as pd
import ruptures as rpt
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils import print_header

# Configure logging
logger = logging.getLogger(__name__)

def _clean_data(df: pd.DataFrame, config) -> pd.DataFrame:
    # (Identical to the _clean_data function in dnn_point_prediction.py)
    # This function would be duplicated here or placed in a shared utils file.
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


def _segment_one_stock_bottom_up(args):
    """Worker function to segment a SINGLE stock's time series into trends."""
    stock_id, price_values, penalty = args
    if len(price_values) < 2: return None
    try:
        algo = rpt.BottomUp(model="l2").fit(price_values)
        breakpoints = algo.predict(pen=penalty)
        all_bps = [0] + breakpoints
        slopes, durations = [], []
        for start_idx, end_idx in zip(all_bps[:-1], all_bps[1:]):
            segment_len = end_idx - start_idx
            if segment_len > 1:
                y_segment = price_values[start_idx:end_idx]
                slope = np.polyfit(np.arange(segment_len), y_segment, 1)[0]
                slopes.append(slope)
                durations.append(segment_len)
        if not slopes: return None
        return (stock_id, np.array(slopes), np.array(durations))
    except Exception as e:
        logger.error(f"Stock {stock_id}: Error during segmentation: {str(e)}")
        return None

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

def run(df, config):
    """Executes the parallelized preprocessing pipeline for non-graph trend models."""
    print_header("Stage 2: Preprocessing (DNN TREND PREDICTION)")
    cleaned_df = _clean_data(df, config)

    logger.info("Segmenting full dataset using Bottom-Up approach...")
    task_args = [
        (sid, sdf[config.TARGET_COLUMN].values, config.SEGMENTATION_PENALTY)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(_segment_one_stock_bottom_up, task_args), total=len(task_args), desc="Segmenting All Stocks"))
    
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