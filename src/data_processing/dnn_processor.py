import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import ruptures as rpt
from .base_processor import BaseProcessor
from ..utils import print_header

from .segmentation import (
    bottomupsegment,
    create_segment_least_squares,
    compute_sum_of_squared_error
)

logger = logging.getLogger(__name__)

def _create_point_sequences_for_stock1(args):
    """Worker function for creating point prediction sequences for a single stock."""
    stock_id, stock_df, scalers, feature_cols, target_col, in_win, out_win = args
    scaler = scalers.get(stock_id)
    if not scaler:
        return stock_id, None, None

    # we are scaling all the data but this is okay because if you remember the scaler is fitted on the training data only and so we are not leaking any information
    scaled_data = scaler.transform(stock_df[feature_cols])
    dates = stock_df['Date'].values
    X, y , sequence_dates= [], [],[]
    
    # Check if we have enough data points to create sequences
    if len(scaled_data) >= in_win + out_win:
        # Create sequences of input and target values
        for i in range(len(scaled_data) - (in_win + out_win) + 1):
            X.append(scaled_data[i : i + in_win, :])
            target_col_idx = feature_cols.index(target_col)
            y.append(scaled_data[i + in_win : i + in_win + out_win, target_col_idx])
            # The date of the prediction corresponds to the first timestamp of the target `y`
            sequence_dates.append(dates[i + in_win])
            
    
    if not X:
        return stock_id, None, None
        
    return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(sequence_dates)

#multistep changes
def _create_point_sequences_for_stock(args):
    """Worker function for creating point prediction sequences for a single stock."""
    stock_id, stock_df, scalers, feature_cols, target_col, in_win, out_win = args
    scaler = scalers.get(stock_id)
    if not scaler:
        return stock_id, None, None, None # Return None for all four expected values

    # we are scaling all the data but this is okay because if you remember the scaler is fitted on the training data only and so we are not leaking any information
    scaled_data = scaler.transform(stock_df[feature_cols])
    dates = stock_df['Date'].values
    X, y , sequence_dates= [], [],[]
    
    # Check if we have enough data points to create sequences
    if len(scaled_data) >= in_win + out_win:
        # Create sequences of input and target values
        for i in range(len(scaled_data) - (in_win + out_win) + 1):
            X.append(scaled_data[i : i + in_win, :])
            target_col_idx = feature_cols.index(target_col)
            y.append(scaled_data[i + in_win : i + in_win + out_win, target_col_idx])
            
            # --- START OF CHANGE ---
            # Capture the entire slice of dates corresponding to the output window (y).
            # This creates an array of dates for each multi-step prediction.
            date_slice = dates[i + in_win : i + in_win + out_win]
            sequence_dates.append(date_slice)
            # --- END OF CHANGE ---
            
    
    if not X:
        return stock_id, None, None, None # Return None for all four expected values
        
    # --- START OF CHANGE ---
    # Return the dates array, which now has a shape of (num_sequences, out_win)
    return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(sequence_dates)
    # --- END OF CHANGE ---


def _segment_one_stock_custom_bottom_up(args):
    """Worker to segment a stock's time series using the custom bottom-up algorithm."""
    stock_id, stock_df, max_error, target_column = args
    price_values = stock_df[target_column].values
    if len(price_values) < 2: 
        return None
        
    try:
        segments = bottomupsegment(
            price_values, 
            create_segment_least_squares, 
            compute_sum_of_squared_error, 
            max_error
        )
        if not segments: return None

        slopes = [seg[3] for seg in segments]
        durations = [seg[2] - seg[0] for seg in segments]
        dates = [stock_df['Date'].iloc[int(seg[0])] for seg in segments]
        
        return (stock_id, np.array(slopes), np.array(durations), dates)
    except Exception as e:
        logger.error(f"Stock {stock_id}: Error during custom segmentation: {e}")
        return None

def _create_trend_sequences_for_stock(args):
    """Worker to create trend sequences for a single stock."""
    stock_id, trends_data, slope_scaler, duration_scaler, window_size = args
    slopes = np.array(trends_data['slopes']).reshape(-1, 1)
    durations = np.array(trends_data['durations']).reshape(-1, 1)
    dates = trends_data['dates']
    scaled_trends = np.column_stack([slope_scaler.transform(slopes), duration_scaler.transform(durations)])
    
    X, y, sequence_dates = [], [], []
    if len(scaled_trends) > window_size:
        for i in range(len(scaled_trends) - window_size):
            X.append(scaled_trends[i:(i + window_size)])
            y.append(scaled_trends[i + window_size])
            sequence_dates.append(dates[i + window_size])
    
    if len(X) > 0:
        return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(sequence_dates)
    return stock_id, None, None, None



class DNNProcessor(BaseProcessor):
    """Processor for non-graph models (TCN, MLP)."""

    def process(self, df: pd.DataFrame):
        print_header(f"Stage 2: DNN Preprocessing ({self.mode} MODE)")
        cleaned_df = self._clean_data(df)

        if self.mode == "POINT":
            return self._process_point(cleaned_df)
        elif self.mode == "TREND":
            return self._process_trend(cleaned_df)
        else:
            raise ValueError(f"Invalid PREDICTION_MODE: {self.mode}")
    
    def _process_point(self, cleaned_df: pd.DataFrame):
        """Handles point prediction for DNN models."""
        scalers = {}
        logger.info("Fitting scalers on training data for each stock...")
        # we split the data by stock and run MinMaxScaler on the training portion
        for stock_id, stock_df in cleaned_df.groupby('StockID'):
            train_size = int(len(stock_df) * self.settings.TRAIN_SPLIT)
            train_data = stock_df.head(train_size)
            if not train_data.empty:
                scalers[stock_id] = MinMaxScaler().fit(train_data[self.settings.FEATURE_COLUMNS])

        #split the dataset into subdataframes by stock ID where sid is the stock identifier and sdf is the dataframe -> and create a tuple of arguments for the worker function for each stock
        task_args = []
        for stock_id, stock_df in cleaned_df.groupby('StockID'):
            args = (
                stock_id,                                # The stock's ID (e.g., 'AAPL')
                stock_df,                                # All historical rows for this stock
                scalers,                                 # Dictionary of MinMaxScalers (already fitted)
                self.settings.FEATURE_COLUMNS,           # Input columns (e.g., ['Open', 'Close', 'Volume'])
                self.settings.TARGET_COLUMN,             # Target column (e.g., 'Close')
                self.settings.POINT_INPUT_WINDOW_SIZE,   # Number of time steps for input (e.g., 5)
                self.settings.POINT_OUTPUT_WINDOW_SIZE   # Steps ahead to predict (e.g., 1 for next-day prediction)
            )
            task_args.append(args)

        all_X, all_y, all_stock_ids, all_dates = [], [], [],[]
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(_create_point_sequences_for_stock, task_args), total=len(task_args), desc="Creating sequences per stock"))
        
        # Collect results from all stocks
        for stock_id, X_stock, y_stock, dates_stock in results:
            # Only append if both X and y are not None
            if X_stock is not None and y_stock is not None:
                # Append the sequences and stock IDs
                all_X.append(X_stock)
                all_y.append(y_stock)
                all_stock_ids.extend([stock_id] * len(X_stock))
                all_dates.append(dates_stock)

        if not all_X: raise ValueError("No sequences could be created from the data.")
        
        return np.concatenate(all_X), np.concatenate(all_y), np.array(all_stock_ids), np.concatenate(all_dates), scalers, None

    def _process_trend(self, cleaned_df: pd.DataFrame):
        """Handles trend prediction for DNN models."""
        logger.info("Segmenting time series using Custom Bottom-Up approach...")
        # Note: The original new code introduced MAX_SEGMENTATION_ERROR. Add it to settings.
        task_args = [
            (sid, sdf.reset_index(drop=True), self.settings.MAX_SEGMENTATION_ERROR, self.settings.TARGET_COLUMN)
            for sid, sdf in cleaned_df.groupby('StockID')
        ]
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(_segment_one_stock_custom_bottom_up, task_args), total=len(task_args), desc="Segmenting All Stocks"))
        
        all_slopes, all_durations, trends_by_stock = [], [], {}
        for res in results:
            if res:
                stock_id, slopes, durations, dates = res
                all_slopes.extend(slopes)
                all_durations.extend(durations)
                trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations, 'dates': dates}

        if not trends_by_stock: raise ValueError("Segmentation failed.")

        slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(all_slopes).reshape(-1, 1))
        duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(all_durations).reshape(-1, 1))
        scalers = {'slope': slope_scaler, 'duration': duration_scaler}

        seq_args = [
            (sid, data, scalers['slope'], scalers['duration'], self.settings.TREND_INPUT_WINDOW_SIZE) 
            for sid, data in trends_by_stock.items()
        ]
        with Pool(processes=cpu_count()) as pool:
            seq_results = list(tqdm(pool.imap(_create_trend_sequences_for_stock, seq_args), total=len(seq_args), desc="Creating Sequences"))

        all_X, all_y, all_stock_ids, all_dates = [], [], [], []
        for stock_id, X_stock, y_stock, dates_stock in seq_results:
            if X_stock is not None and y_stock is not None:
                all_X.append(X_stock)
                all_y.append(y_stock)
                all_stock_ids.extend([stock_id] * len(X_stock))
                all_dates.append(dates_stock)
        
        if not all_X: raise ValueError("Sequencing failed.")
        
        return np.concatenate(all_X), np.concatenate(all_y), np.array(all_stock_ids), None, scalers, None