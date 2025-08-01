# src/pipeline/stage_2_data_preprocessing/gnn_trendline_prediction.py

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
    Prepares trend data in a graph-compatible format using the same custom 
    Bottom-Up segmentation as the DNN pipeline.
    """
    print_header("Stage 2: Preprocessing (GNN TREND PREDICTION)")
    
    # 1. Clean data using the shared function
    cleaned_df = _clean_data(df, config)
    
    # 2. Segment trends for each stock using the exact same parallelized worker
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

    # 3. Create graph-structured daily dataframes (Dates x Stocks)
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

    # --- START OF FIX ---
    # 4. Prepare the data for the adjacency matrix function in the correct format.
    # First, unpivot the data from wide (Dates x Stocks) to long format.
    slopes_long_format = daily_slopes.reset_index().melt(
        id_vars=['index'], 
        value_vars=stock_ids, 
        var_name='StockID', 
        value_name='slope'
    )
    
    # Second, rename the 'index' column to 'Date' to match the function's requirement.
    slopes_long_format.rename(columns={'index': 'Date'}, inplace=True)
    
    # Now, create the adjacency matrix using the correctly formatted DataFrame.
    adj_matrix = create_adjacency_matrix(
        slopes_long_format, 
        target_column='slope', 
        threshold=0.5
    )
    # --- END OF FIX ---

    # 5. Scale the data based on the training set
    logger.info("Scaling trend data...")
    train_end_idx = int(len(all_dates) * config.TRAIN_SPLIT)
    train_dates = all_dates[:train_end_idx]
    
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(daily_slopes.loc[train_dates])
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(daily_durations.loc[train_dates])
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}

    scaled_slopes = slope_scaler.transform(daily_slopes)
    scaled_durations = duration_scaler.transform(daily_durations)
    
    graph_data = np.stack([scaled_slopes, scaled_durations], axis=-1)

    # 6. Create sequences (X, y) and the corresponding dates array
    logger.info("Creating GNN sequences and dates...")
    X, y, dates_for_seq = [], [], []
    in_win = config.TREND_INPUT_WINDOW_SIZE
    total_len = in_win + 1 

    for i in range(len(graph_data) - total_len + 1):
        X.append(graph_data[i : i + in_win, :, :])
        y.append(graph_data[i + in_win, :, :])
        dates_for_seq.append(all_dates[i + in_win])
    
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64)
    dates_np = np.array(dates_for_seq)
    
    logger.info("GNN trendline prediction preprocessing complete.")
    logger.info(f"  - Generated X shape: {X_np.shape}")
    logger.info(f"  - Generated y shape: {y_np.shape}")
    logger.info(f"  - Generated dates shape: {dates_np.shape}")
    
    # 7. Return all 6 required components for a graph model
    return X_np, y_np, stock_ids, dates_np, scalers, adj_matrix