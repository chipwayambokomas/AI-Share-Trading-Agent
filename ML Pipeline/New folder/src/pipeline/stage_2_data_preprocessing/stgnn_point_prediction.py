# src/pipeline/stage_2_data_preprocessing/stgnn_point_prediction.py

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
    """
    Performs initial data cleaning by handling non-numeric values, NaNs,
    and filtering out stocks with excessive missing data. This function is
    common to all STGNN-based models.
    """
    logger.info("Starting data cleaning for STGNN point prediction...")
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

    if not cleaned_stocks:
        raise ValueError("No stocks remained after the cleaning process. Check data quality or cleaning thresholds.")

    final_df = pd.concat(cleaned_stocks, ignore_index=True)
    logger.info(f"Data cleaning complete. Kept {final_df['StockID'].nunique()}/{initial_stock_count} stocks.")
    return final_df

def run(df, config):
    """
    Prepares multivariate data for spatio-temporal graph-based point prediction.
    This function routes to the correct logic based on the MODEL_TYPE.
    """
    cleaned_df = _clean_data(df, config)
    
    if config.MODEL_TYPE == 'HSDGNN':
        print_header("Stage 2: Preprocessing (HSDGNN POINT PREDICTION)")
        
        logger.info("Adding time-based features for HSDGNN...")
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
        # Use set_index on a copy to avoid SettingWithCopyWarning
        cleaned_df_indexed = cleaned_df.set_index('Date')
        cleaned_df_indexed['time_of_day'] = (cleaned_df_indexed.index.hour * 60 + cleaned_df_indexed.index.minute) / (24 * 60)
        cleaned_df_indexed['day_of_week'] = cleaned_df_indexed.index.dayofweek
        
        all_feature_cols = config.FEATURE_COLUMNS + ['time_of_day', 'day_of_week']
        
        logger.info("Pivoting data to (dates, stocks, features) format...")
        pivoted_df = cleaned_df_indexed.pivot(columns='StockID', values=all_feature_cols)
        pivoted_df.ffill(inplace=True); pivoted_df.bfill(inplace=True)
        stock_ids = pivoted_df.columns.get_level_values('StockID').unique().tolist()

        scalers = {}
        scaled_data_list = []
        train_end_idx = int(len(pivoted_df) * config.TRAIN_SPLIT)
        train_end_date = pivoted_df.index[train_end_idx]

        for stock_id in tqdm(stock_ids, desc="Scaling features per stock for HSDGNN"):
            stock_df = pivoted_df.xs(stock_id, level='StockID', axis=1)
            original_features_df = stock_df[config.FEATURE_COLUMNS]
            time_features_df = stock_df[['time_of_day', 'day_of_week']]

            scaler = MinMaxScaler().fit(original_features_df.loc[:train_end_date])
            scalers[stock_id] = scaler
            
            scaled_original_features = scaler.transform(original_features_df)
            combined_features = np.hstack([scaled_original_features, time_features_df.values])
            scaled_data_list.append(combined_features)

        scaled_array = np.stack(scaled_data_list, axis=1)
        adj_matrix = None
        logger.info("Skipping static adjacency matrix creation for HSDGNN (uses dynamic graph).")

    else: # Original logic for other STGNNs (e.g., DSTAGNN)
        print_header(f"Stage 2: Preprocessing ({config.MODEL_TYPE} POINT PREDICTION)")
        
        logger.info("Pivoting data to (dates, stocks, features) format...")
        pivoted_df = cleaned_df.pivot(index='Date', columns='StockID', values=config.FEATURE_COLUMNS)
        pivoted_df.ffill(inplace=True); pivoted_df.bfill(inplace=True)
        stock_ids = pivoted_df.columns.get_level_values('StockID').unique().tolist()
        
        adj_matrix = create_adjacency_matrix(cleaned_df, config.TARGET_COLUMN)
        
        scalers = {}
        scaled_data_dict = {}
        train_end_date = pivoted_df.index[int(len(pivoted_df) * config.TRAIN_SPLIT)]
        
        for stock_id in tqdm(stock_ids, desc=f"Scaling features per stock for {config.MODEL_TYPE}"):
            stock_features_df = pivoted_df.xs(stock_id, level='StockID', axis=1)
            scaler = MinMaxScaler().fit(stock_features_df.loc[:train_end_date])
            scalers[stock_id] = scaler
            scaled_data_dict[stock_id] = scaler.transform(stock_features_df)
        
        num_dates, num_nodes, num_features = len(pivoted_df), len(stock_ids), len(config.FEATURE_COLUMNS)
        scaled_array = np.zeros((num_dates, num_nodes, num_features))
        for i, stock_id in enumerate(stock_ids):
            scaled_array[:, i, :] = scaled_data_dict[stock_id]

    # This part is common to both logic paths
    X, y = [], []
    in_win, out_win = config.POINT_INPUT_WINDOW_SIZE, config.POINT_OUTPUT_WINDOW_SIZE
    total_len = in_win + out_win
    
    for i in tqdm(range(len(scaled_array) - total_len + 1), desc="Creating graph sequences"):
        X.append(scaled_array[i : i + in_win, :, :])
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
        y.append(scaled_array[i + in_win : i + total_len, :, target_col_idx:target_col_idx+1])

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)
    
    # --- START OF FIX ---
    # The dates must correspond to the target (y) values. For each sequence,
    # we need to capture the 'horizon' number of dates from the original index.
    num_sequences = len(X_np)

    # Create a list of date arrays, one for each sequence's prediction horizon.
    dates_list = []
    for i in range(num_sequences):
        # The target dates start immediately after the input window ends.
        start_idx = i + in_win
        end_idx = start_idx + out_win
        dates_list.append(pivoted_df.index[start_idx:end_idx].to_numpy())

    # Convert the list of arrays into a single 2D numpy array of shape (num_sequences, horizon).
    dates_np = np.array(dates_list)
    # --- END OF FIX ---
    
    logger.info(f"{config.MODEL_TYPE} point prediction preprocessing complete.")
    logger.info(f"Generated X shape: {X_np.shape}, y shape: {y_np.shape}")
    logger.info(f"Generated dates shape: {dates_np.shape}")
    
    # Return the correctly shaped dates array.
    return X_np, y_np, stock_ids, dates_np, scalers, adj_matrix