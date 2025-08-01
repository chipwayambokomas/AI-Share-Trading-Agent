# src/preprocessing/gnn_point_prediction.py

import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils import print_header, create_adjacency_matrix

# Configure logging
logger = logging.getLogger(__name__)

def _clean_data(df: pd.DataFrame, config) -> pd.DataFrame:
    # (This function remains unchanged)
    logger.info("Starting data cleaning for GNN point prediction...")
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


def run(df, config):
    """Prepares multivariate data for graph-based point prediction."""
    print_header("Stage 2: Preprocessing (GNN POINT PREDICTION)")
    cleaned_df = _clean_data(df, config)
    
    logger.info("Pivoting data to (dates, stocks, features) format...")
    pivoted_df = cleaned_df.pivot(index='Date', columns='StockID', values=config.FEATURE_COLUMNS)
    pivoted_df.ffill(inplace=True); pivoted_df.bfill(inplace=True)
    stock_ids = pivoted_df.columns.get_level_values('StockID').unique().tolist()
    
    adj_matrix = create_adjacency_matrix(cleaned_df, config.TARGET_COLUMN)
    
    scalers = {}
    scaled_data_dict = {}
    train_end_date = pivoted_df.index[int(len(pivoted_df) * config.TRAIN_SPLIT)]
    
    for stock_id in tqdm(stock_ids, desc="Scaling features per stock"):
        stock_features_df = pivoted_df.xs(stock_id, level='StockID', axis=1)
        scaler = MinMaxScaler().fit(stock_features_df.loc[:train_end_date])
        scalers[stock_id] = scaler
        scaled_data_dict[stock_id] = scaler.transform(stock_features_df)
    
    num_dates, num_nodes, num_features = len(pivoted_df), len(stock_ids), len(config.FEATURE_COLUMNS)
    scaled_array = np.zeros((num_dates, num_nodes, num_features))
    for i, stock_id in enumerate(stock_ids):
        scaled_array[:, i, :] = scaled_data_dict[stock_id]

    X, y = [], []
    in_win, out_win = config.POINT_INPUT_WINDOW_SIZE, config.POINT_OUTPUT_WINDOW_SIZE
    total_len = in_win + out_win
    
    for i in tqdm(range(len(scaled_array) - total_len + 1), desc="Creating graph sequences"):
        X.append(scaled_array[i : i + in_win, :, :])
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
        y.append(scaled_array[i + in_win : i + total_len, :, target_col_idx:target_col_idx+1])

    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64)
    
    # --- START OF FIX ---
    # Create the dates array corresponding to the target (y) values.
    # For each sequence, we capture the 'horizon' number of dates from the original index.
    num_sequences = len(X_np)
    dates_list = []
    for i in range(num_sequences):
        # The target dates start immediately after the input window ends.
        start_idx = i + in_win
        end_idx = start_idx + out_win
        dates_list.append(pivoted_df.index[start_idx:end_idx].to_numpy())

    # Convert the list of arrays into a single 2D numpy array of shape (num_samples, horizon).
    dates_np = np.array(dates_list)
    # --- END OF FIX ---
    
    logger.info("GNN point prediction preprocessing complete.")
    logger.info(f"Generated X shape: {X_np.shape}, y shape: {y_np.shape}, dates shape: {dates_np.shape}")
    
    return X_np, y_np, stock_ids, dates_np, scalers, adj_matrix