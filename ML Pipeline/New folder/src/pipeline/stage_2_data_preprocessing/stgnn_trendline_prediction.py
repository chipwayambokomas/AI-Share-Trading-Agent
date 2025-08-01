# src/pipeline/stage_2_data_preprocessing/stgnn_trendline_prediction.py

import logging
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils import print_header, create_adjacency_matrix

# --- Import the exact same segmentation functions from the DNN version for consistency ---
from .dnn_trendline_prediction import (
    _clean_data, 
    _segment_one_stock_custom_bottom_up,
    bottomupsegment,
    _create_segment_least_squares,
    _compute_sum_of_squared_error
)

logger = logging.getLogger(__name__)

def run(df, config):
    """
    Prepares trend data in a spatio-temporal graph-compatible format, including
    the time-based features required by models like HSDGNN.
    """
    print_header("Stage 2: Preprocessing (STGNN TREND PREDICTION)")
    
    # 1. Clean data
    cleaned_df = _clean_data(df, config)
    
    # 2. Segment trends
    logger.info("Segmenting trends for each stock using custom bottom-up algorithm...")
    task_args = [
        (sid, sdf.reset_index(drop=True), config.MAX_SEGMENTATION_ERROR, config.TARGET_COLUMN)
        for sid, sdf in cleaned_df.groupby('StockID')
    ]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(_segment_one_stock_custom_bottom_up, task_args), total=len(task_args), desc="Segmenting All Stocks"))
    
    trends_by_stock = {}
    for res in results:
        if res:
            stock_id, slopes, durations, dates = res
            trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations, 'dates': dates}

    if not trends_by_stock:
        raise ValueError("Trend segmentation failed for all stocks.")
    
     # --- START: ADD THIS CODE BLOCK TO PRINT TREND STATISTICS ---
    num_stocks_with_trends = len(trends_by_stock)
    all_durations_list = [duration for stock_data in trends_by_stock.values() for duration in stock_data['durations']]
    
    if not all_durations_list:
        print("\nWARNING: Segmentation ran but produced zero valid trends.\n")
        total_trends_formed = 0
        avg_trends_per_stock = 0
        avg_trend_duration = 0
    else:
        total_trends_formed = len(all_durations_list)
        avg_trends_per_stock = total_trends_formed / num_stocks_with_trends
        avg_trend_duration = sum(all_durations_list) / total_trends_formed

    print("\n" + "="*50)
    print("--- Trend Segmentation Summary ---")
    print(f"Parameter `MAX_SEGMENTATION_ERROR`: {config.MAX_SEGMENTATION_ERROR}")
    print(f"Successfully generated trends for {num_stocks_with_trends} stocks.")
    print(f"Total number of trends formed: {total_trends_formed}")
    print(f"Average trends per stock: {avg_trends_per_stock:.2f}")
    print(f"Average trend duration (days): {avg_trend_duration:.2f}")
    print("="*50 + "\n")
    # --- END OF ADDED CODE BLOCK ---

    # 3. Create graph-structured daily dataframes
    logger.info("Creating graph-structured daily dataframes...")
    all_dates = pd.to_datetime(np.unique(cleaned_df['Date'])).sort_values()
    stock_ids = sorted(trends_by_stock.keys())
    
    daily_slopes = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)
    daily_durations = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)

    for stock_id, trends in tqdm(trends_by_stock.items(), desc="Populating daily data"):
        for i in range(len(trends['slopes'])):
            start_date = trends['dates'][i]
            duration = trends['durations'][i]
            end_date = start_date + pd.to_timedelta(duration - 1, unit='D')
            
            daily_slopes.loc[start_date:end_date, stock_id] = trends['slopes'][i]
            daily_durations.loc[start_date:end_date, stock_id] = trends['durations'][i]
    
    daily_slopes.ffill(inplace=True); daily_slopes.bfill(inplace=True)
    daily_durations.ffill(inplace=True); daily_durations.bfill(inplace=True)

    # 4. Create the adjacency matrix
    slopes_long_format = daily_slopes.reset_index().melt(
        id_vars=['index'], value_vars=stock_ids, var_name='StockID', value_name='slope'
    )
    slopes_long_format.rename(columns={'index': 'Date'}, inplace=True)
    adj_matrix = create_adjacency_matrix(
        slopes_long_format, target_column='slope', threshold=0.5
    )

    # 5. Scale the trend data
    logger.info("Scaling trend data...")
    train_end_idx = int(len(all_dates) * config.TRAIN_SPLIT)
    train_dates = all_dates[:train_end_idx]
    
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(daily_slopes.loc[train_dates])
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(daily_durations.loc[train_dates])
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}

    scaled_slopes = slope_scaler.transform(daily_slopes)
    scaled_durations = duration_scaler.transform(daily_durations)
    
    # --- START OF FIX: Create and add time-based features ---
    logger.info("Creating time-based features for HSDGNN...")
    # Day of the week (0=Monday, 6=Sunday), normalized to [0, 1]
    day_of_week = all_dates.dayofweek / 6.0
    
    # Time of day, normalized to [0, 1). For daily data, this will be 0.
    # The model requires this feature, even if it's constant for daily data.
    time_of_day = (all_dates.hour * 60 + all_dates.minute) / (24 * 60)
    
    # Reshape time features to be broadcastable to the main graph_data shape
    # Shape becomes (num_timesteps, 1, 1)
    day_of_week_reshaped = day_of_week.values.reshape(-1, 1, 1)
    time_of_day_reshaped = time_of_day.values.reshape(-1, 1, 1)
    
    # Broadcast to (num_timesteps, num_nodes, 1)
    num_nodes = len(stock_ids)
    day_of_week_broadcast = np.broadcast_to(day_of_week_reshaped, (len(all_dates), num_nodes, 1))
    time_of_day_broadcast = np.broadcast_to(time_of_day_reshaped, (len(all_dates), num_nodes, 1))

    # Stack all features together: slopes, durations, day_of_week, time_of_day
    # The final feature dimension will be 4.
    graph_data = np.stack([
        scaled_slopes, 
        scaled_durations, 
        day_of_week_broadcast[..., 0], # Squeeze the last dimension back
        time_of_day_broadcast[..., 0]
    ], axis=-1)
    # --- END OF FIX ---

    # 6. Create sequences (X, y) and the corresponding dates array
    logger.info("Creating STGNN sequences and dates...")
    X, y, dates_for_seq = [], [], []
    in_win = config.TREND_INPUT_WINDOW_SIZE
    total_len = in_win + 1 

    for i in range(len(graph_data) - total_len + 1):
        # Input X should contain all features (trends + time)
        X.append(graph_data[i : i + in_win, :, :])
        # Target y should only contain the features to be predicted (trends)
        y.append(graph_data[i + in_win, :, :2]) # Slice to get only slope and duration
        
        dates_for_seq.append(all_dates[i + in_win])
    
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64)
    dates_np = np.array(dates_for_seq)
    
    logger.info("STGNN trendline prediction preprocessing complete.")
    logger.info(f"  - Generated X shape: {X_np.shape}")
    logger.info(f"  - Generated y shape: {y_np.shape}")
    logger.info(f"  - Generated dates shape: {dates_np.shape}")
    
    # 7. Return all 6 required components for a graph model
    return X_np, y_np, stock_ids, dates_np, scalers, adj_matrix