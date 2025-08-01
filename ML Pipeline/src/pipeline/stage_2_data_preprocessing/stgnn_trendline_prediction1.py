import logging
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils import print_header

# Configure logging
logger = logging.getLogger(__name__)

def _clean_data(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Cleans the raw dataframe by handling missing and erroneous values.
    """
    logger.info("Starting data cleaning for STGNN trend prediction...")
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

def _segment_one_stock_with_dates_bottom_up(args):
    """Worker function to segment a stock's time series and return trends with dates."""
    stock_id, stock_df, target_column, penalty = args
    price_values = stock_df[target_column].values
    dates = stock_df['Date'].values
    if len(price_values) < 2: return stock_id, []

    try:
        algo = rpt.BottomUp(model="l2").fit(price_values)
        breakpoints = algo.predict(pen=penalty)
        all_bps = sorted(list(set([0] + breakpoints + [len(price_values)])))

        trends = []
        for start, end in zip(all_bps[:-1], all_bps[1:]):
            if end - start > 1:
                slope = np.polyfit(np.arange(end - start), price_values[start:end], 1)[0]
                trends.append({
                    'start_date': dates[start],
                    'end_date': dates[end - 1],
                    'slope': slope,
                    'duration': end - start
                })
        return stock_id, trends
    except Exception as e:
        logger.error(f"Error segmenting stock {stock_id} with dates: {e}")
        return stock_id, []

def run(df, config):
    """Prepares trend data in a spatio-temporal graph-compatible format."""
    print_header("Stage 2: Preprocessing (STGNN TREND PREDICTION)")
    cleaned_df = _clean_data(df, config)
    
    task_args = [(sid, sdf, config.TARGET_COLUMN, config.SEGMENTATION_PENALTY) for sid, sdf in cleaned_df.groupby('StockID')]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(_segment_one_stock_with_dates_bottom_up, task_args), total=len(task_args), desc="Segmenting Trends"))
    
    trends_by_stock = {sid: trends for sid, trends in results if trends}
    if not trends_by_stock:
        raise ValueError("Trend segmentation failed for all stocks.")

    all_dates = pd.to_datetime(np.unique(cleaned_df['Date'])).sort_values()
    stock_ids = sorted(trends_by_stock.keys())
    
    daily_slopes = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)
    daily_durations = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)

    for stock_id, trends in trends_by_stock.items():
        for trend in trends:
            daily_slopes.loc[trend['start_date']:trend['end_date'], stock_id] = trend['slope']
            daily_durations.loc[trend['start_date']:trend['end_date'], stock_id] = trend['duration']
    
    daily_slopes.ffill(inplace=True); daily_slopes.bfill(inplace=True)
    daily_durations.ffill(inplace=True); daily_durations.bfill(inplace=True)

    # --- Create Adjacency Matrix (matching the working script's logic) ---
    correlation_matrix = daily_slopes.corr()
    adj_matrix_np = (correlation_matrix.abs() >= 0.5).astype(int).values
    np.fill_diagonal(adj_matrix_np, 0)
    adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32)
    # --- End of Adjacency Matrix Creation ---

    train_end_idx = int(len(all_dates) * config.TRAIN_SPLIT)
    train_dates = all_dates[:train_end_idx]
    
    slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(daily_slopes.loc[train_dates])
    duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(daily_durations.loc[train_dates])
    scalers = {'slope': slope_scaler, 'duration': duration_scaler}

    scaled_slopes = slope_scaler.transform(daily_slopes)
    scaled_durations = duration_scaler.transform(daily_durations)
    graph_data = np.stack([scaled_slopes, scaled_durations], axis=-1)

    X, y = [], []
    in_win = config.TREND_INPUT_WINDOW_SIZE
    total_len = in_win + 1 # Input window + 1 output step

    for i in range(len(graph_data) - total_len + 1):
        X.append(graph_data[i : i + in_win, :, :])
        y.append(graph_data[i + in_win, :, :])
    
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)
    
    logger.info("STGNN trendline prediction preprocessing complete.")
    return X_np, y_np, stock_ids, scalers, adj_matrix