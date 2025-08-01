# src/preprocessing/dnn_point_prediction.py

import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils import print_header

# Configure logging
logger = logging.getLogger(__name__)

def _clean_data(df: pd.DataFrame, config) -> pd.DataFrame:
    """Cleans the raw dataframe by handling missing and erroneous values."""
    logger.info("Starting data cleaning for DNN point prediction...")
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

def _create_sequences_for_stock(args):
    """Worker function for creating sequences for a single stock."""
    stock_id, stock_df, scalers, feature_cols, target_col, in_win, out_win = args
    scaler = scalers.get(stock_id)
    if not scaler:
        return stock_id, None, None

    scaled_data = scaler.transform(stock_df[feature_cols])
    X, y = [], []
    total_len = in_win + out_win

    if len(scaled_data) >= total_len:
        for i in range(len(scaled_data) - total_len + 1):
            X.append(scaled_data[i : i + in_win, :])
            target_col_idx = feature_cols.index(target_col)
            y.append(scaled_data[i + in_win : i + total_len, target_col_idx])
    
    if not X:
        return stock_id, None, None
        
    return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def run(df, config):
    """Prepares multivariate data for point prediction using TCN/MLP models."""
    print_header("Stage 2: Preprocessing (DNN POINT PREDICTION)")
    cleaned_df = _clean_data(df, config)
    
    scalers = {}
    logger.info("Fitting scalers on training data for each stock...")
    for stock_id, stock_df in cleaned_df.groupby('StockID'):
        train_size = int(len(stock_df) * config.TRAIN_SPLIT)
        train_data = stock_df.head(train_size)
        if not train_data.empty:
            scaler = MinMaxScaler()
            scalers[stock_id] = scaler.fit(train_data[config.FEATURE_COLUMNS])

    task_args = [
        (stock_id, stock_df, scalers, config.FEATURE_COLUMNS, config.TARGET_COLUMN,
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
    
    logger.info("DNN point prediction preprocessing complete.")
    return final_X, final_y, np.array(all_stock_ids), scalers, None