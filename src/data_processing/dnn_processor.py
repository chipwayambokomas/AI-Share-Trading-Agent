import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import ruptures as rpt
from .base_processor import BaseProcessor
from ..utils import print_header

logger = logging.getLogger(__name__)

def _create_point_sequences_for_stock(args):
    """Worker function for creating point prediction sequences for a single stock."""
    stock_id, stock_df, scalers, feature_cols, target_col, in_win, out_win = args
    scaler = scalers.get(stock_id)
    if not scaler:
        return stock_id, None, None

    # we are scaling all the data but this is okay because if you remember the scaler is fitted on the training data only and so we are not leaking any information
    scaled_data = scaler.transform(stock_df[feature_cols])
    X, y = [], []
    
    # Check if we have enough data points to create sequences
    if len(scaled_data) >= in_win + out_win:
        # Create sequences of input and target values
        for i in range(len(scaled_data) - (in_win + out_win) + 1):
            X.append(scaled_data[i : i + in_win, :])
            target_col_idx = feature_cols.index(target_col)
            y.append(scaled_data[i + in_win : i + in_win + out_win, target_col_idx])
    
    if not X:
        return stock_id, None, None
        
    return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def _segment_one_stock_bottom_up(args):
    """Worker function to segment a single stock's time series into trends."""
    stock_id, price_values, penalty = args
    if len(price_values) < 2: return None
      #initialize a change point detection algorithm using the Bottom-Up method, it starts with the entire time series and iteratively merges segments to find the best fit and by best fit we mean we stop when merging would lose too much accuracy. it uses the l2 model which uses least squares to measure the fit of the segments
    algo = rpt.BottomUp(model="l2").fit(price_values)
    
    #predict the breakpoints in the time series using the penalty value to control how many segments we want -> lower penalty means more segments
    breakpoints = algo.predict(pen=penalty)
    all_bps = [0] + breakpoints
    slopes, durations = [], []
    
    # Iterate through the segments defined by the breakpoints
    for start, end in zip(all_bps[:-1], all_bps[1:]):
        # Ensure we have at least two points to calculate a slope
        if end - start > 1:
            # Calculate the slope of the segment
            y_segment = price_values[start:end]
            slope = np.polyfit(np.arange(end - start), y_segment, 1)[0]
            slopes.append(slope)
            durations.append(end - start)
    if not slopes: return None
    return (stock_id, np.array(slopes), np.array(durations))

def _create_trend_sequences_for_stock(args):
    """Worker function to create trend sequences for a single stock."""
    stock_id, trends_data, slope_scaler, duration_scaler, window_size = args
    # get all the slopes and durations from the trends_data and reshape them to be 2D arrays
    slopes = np.array(trends_data['slopes']).reshape(-1, 1)
    durations = np.array(trends_data['durations']).reshape(-1, 1)
    # scale the slopes and durations using the fitted scalers
    scaled_trends = np.column_stack([slope_scaler.transform(slopes), duration_scaler.transform(durations)])
    
    X, y = [], []
    # Create sequences of length window_size
    if len(scaled_trends) > window_size:
        # same thing as before, we create sequences of input and target values based on the window size
        for i in range(len(scaled_trends) - window_size):
            X.append(scaled_trends[i:(i + window_size)])
            y.append(scaled_trends[i + window_size])
    
    if len(X) > 0:
        return stock_id, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return stock_id, None, None


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

        all_X, all_y, all_stock_ids = [], [], []
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(_create_point_sequences_for_stock, task_args), total=len(task_args), desc="Creating sequences per stock"))
        
        # Collect results from all stocks
        for stock_id, X_stock, y_stock in results:
            # Only append if both X and y are not None
            if X_stock is not None and y_stock is not None:
                # Append the sequences and stock IDs
                all_X.append(X_stock)
                all_y.append(y_stock)
                all_stock_ids.extend([stock_id] * len(X_stock))

        if not all_X: raise ValueError("No sequences could be created from the data.")
        
        return np.concatenate(all_X), np.concatenate(all_y), np.array(all_stock_ids), scalers, None

    def _process_trend(self, cleaned_df: pd.DataFrame):
        """Handles trend prediction for DNN models."""
        logger.info("Segmenting time series using Bottom-Up approach...")
        #we again are grouping records into subdataframes by stock ID and then creating a tuple of arguments for the worker function for each stock
        task_args = [
            (sid, sdf[self.settings.TARGET_COLUMN].values, self.settings.SEGMENTATION_PENALTY)
            for sid, sdf in cleaned_df.groupby('StockID')
        ]
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(_segment_one_stock_bottom_up, task_args), total=len(task_args), desc="Segmenting All Stocks"))
        
        # here we collect all slopes and durations from the results
        # and also create a mapping of stock IDs to their trends -> chnaged to only include training data
        all_slopes, all_durations, trends_by_stock = [], [], {}
        for res in results:
            if res:
                stock_id, slopes, durations = res
                train_size = int(len(slopes) * self.settings.TRAIN_SPLIT)

                train_slopes = slopes[:train_size]
                train_durations = durations[:train_size]

                all_slopes.extend(train_slopes)
                all_durations.extend(train_durations)
                trends_by_stock[stock_id] = {'slopes': slopes, 'durations': durations}

        if not trends_by_stock: raise ValueError("Segmentation failed to produce any trends.")

        logger.info("Fitting global scalers for slope and duration...")
        # Create scalers for slopes and durations
        #get the slope and duration values as numpy arrays of all stocks -> make sure we only have one column for each
        slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(all_slopes).reshape(-1, 1))
        duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(all_durations).reshape(-1, 1))
        scalers = {'slope': slope_scaler, 'duration': duration_scaler}

        logger.info("Creating sequences from trends...")
        # Create sequences for each stock's trends using the fitted scalers
        seq_args = [
            (sid, data, scalers['slope'], scalers['duration'], self.settings.TREND_INPUT_WINDOW_SIZE) 
            for sid, data in trends_by_stock.items()
        ]
        with Pool(processes=cpu_count()) as pool:
            seq_results = list(tqdm(pool.imap(_create_trend_sequences_for_stock, seq_args), total=len(seq_args), desc="Creating Sequences"))

        # using the results from the worker function, we collect all sequences and their corresponding stock IDs
        all_X, all_y, all_stock_ids = [], [], []
        for stock_id, X_stock, y_stock in seq_results:
            if X_stock is not None and y_stock is not None:
                all_X.append(X_stock)
                all_y.append(y_stock)
                all_stock_ids.extend([stock_id] * len(X_stock))
        
        if not all_X: raise ValueError("Sequencing failed: No trend sequences could be created.")
        
        return np.concatenate(all_X), np.concatenate(all_y), np.array(all_stock_ids), scalers, None