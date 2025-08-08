import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .base_processor import BaseProcessor
from ..utils import print_header, create_adjacency_matrix


from .segmentation import (
    bottomupsegment,
    create_segment_least_squares,
    compute_sum_of_squared_error
)

logger = logging.getLogger(__name__)

def _segment_one_stock_custom_bottom_up_with_dates(args):
    """Worker to segment a stock's time series using the custom bottom-up algorithm and return dates."""
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
    
class GraphProcessor(BaseProcessor):
    """Processor for all graph-based models (GNN, STGNN, etc.)."""

    def process(self, df: pd.DataFrame):
        print_header(f"Stage 2: Graph Preprocessing ({self.mode} MODE)")
        cleaned_df = self._clean_data(df)

        if self.mode == "POINT":
            return self._process_point(cleaned_df)
        elif self.mode == "TREND":
            return self._process_trend_custom(cleaned_df)
        else:
            raise ValueError(f"Invalid PREDICTION_MODE: {self.mode}")

    def _process_point1(self, cleaned_df: pd.DataFrame):
        """Handles point prediction for graph models."""
        logger.info("Pivoting data to (dates, stocks, features) format...")
        #this line created a new df, a multi-indexed DataFrame with dates as index and stock ID paird with their respective features as columns, each column contains the feature value for that specific stock -> Date | PriceA | PriceB | VolumeA | VolumeB
        pivoted_df = cleaned_df.pivot(index='Date', columns='StockID', values=self.settings.FEATURE_COLUMNS)
        pivoted_df.ffill(inplace=True); pivoted_df.bfill(inplace=True)
        
        stock_ids = pivoted_df.columns.get_level_values('StockID').unique().tolist()
        
        adj_matrix = create_adjacency_matrix(cleaned_df, self.settings.TARGET_COLUMN, self.settings.CORRELATION_THRESHOLD)
        
        scalers, scaled_array = self._scale_pivoted_data_point(pivoted_df, stock_ids)
        
        logger.info("Creating graph-structured sequences...")
        dates_index = pivoted_df.index
        X, y, all_dates = [], [],[]
        in_win = self.settings.POINT_INPUT_WINDOW_SIZE
        out_win = self.settings.POINT_OUTPUT_WINDOW_SIZE
        target_col_idx = self.settings.FEATURE_COLUMNS.index(self.settings.TARGET_COLUMN)

        #we want to slide a window over the scaled time series data and grab the input and output sequences
        for i in tqdm(range(len(scaled_array) - (in_win + out_win) + 1), desc="Creating graph sequences"):
            #take a window of inputs for all the stocks and for all the features
            X.append(scaled_array[i : i + in_win, :, :])
            #take the output for the target column for all the stocks
            y.append(scaled_array[i + in_win : i + in_win + out_win, :, target_col_idx:target_col_idx+1])
            prediction_date = dates_index[i + in_win]
            all_dates.append(prediction_date)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), stock_ids, np.array(all_dates), scalers, adj_matrix
    #multi step point prediction
    def _process_point(self, cleaned_df: pd.DataFrame):
        """Handles point prediction for graph models."""
        logger.info("Pivoting data to (dates, stocks, features) format...")
        #this line created a new df, a multi-indexed DataFrame with dates as index and stock ID paird with their respective features as columns, each column contains the feature value for that specific stock -> Date | PriceA | PriceB | VolumeA | VolumeB
        pivoted_df = cleaned_df.pivot(index='Date', columns='StockID', values=self.settings.FEATURE_COLUMNS)
        pivoted_df.ffill(inplace=True); pivoted_df.bfill(inplace=True)
        
        stock_ids = pivoted_df.columns.get_level_values('StockID').unique().tolist()
        
        adj_matrix = create_adjacency_matrix(cleaned_df, self.settings.TARGET_COLUMN, self.settings.CORRELATION_THRESHOLD)
        
        scalers, scaled_array = self._scale_pivoted_data_point(pivoted_df, stock_ids)
        
        logger.info("Creating graph-structured sequences...")
        dates_index = pivoted_df.index
        X, y, all_dates = [], [],[]
        in_win = self.settings.POINT_INPUT_WINDOW_SIZE
        out_win = self.settings.POINT_OUTPUT_WINDOW_SIZE
        target_col_idx = self.settings.FEATURE_COLUMNS.index(self.settings.TARGET_COLUMN)

        #we want to slide a window over the scaled time series data and grab the input and output sequences
        for i in tqdm(range(len(scaled_array) - (in_win + out_win) + 1), desc="Creating graph sequences"):
            #take a window of inputs for all the stocks and for all the features
            X.append(scaled_array[i : i + in_win, :, :])
            #take the output for the target column for all the stocks
            y.append(scaled_array[i + in_win : i + in_win + out_win, :, target_col_idx:target_col_idx+1])
            
            # --- START OF CHANGE ---
            # Capture the entire slice of dates corresponding to the output window (y).
            # This creates an array of dates for each multi-step prediction.
            start_idx = i + in_win
            end_idx = start_idx + out_win
            date_slice = dates_index[start_idx:end_idx].to_numpy()
            all_dates.append(date_slice)
            # --- END OF CHANGE ---

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), stock_ids, np.array(all_dates), scalers, adj_matrix
    
    def _process_trend_custom(self, cleaned_df: pd.DataFrame):
        """Handles trend prediction for graph models."""
        # Segment each stock's time series into trend components using multiprocessing
        task_args = [
            (stock_id, stock_df.reset_index(drop=True), 
            self.settings.MAX_SEGMENTATION_ERROR, 
            self.settings.TARGET_COLUMN)
            for stock_id, stock_df in cleaned_df.groupby('StockID')
        ]
        
        # Process all stocks in parallel for efficiency
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(_segment_one_stock_custom_bottom_up_with_dates, task_args), 
                total=len(task_args), 
                desc="Segmenting stock trends"
            ))
        
        # Filter successful results and organize by stock ID
        trends_by_stock = {
            stock_id: {'slopes': slopes, 'durations': durations, 'dates': dates} 
            for stock_id, slopes, durations, dates in results 
            if slopes is not None  # Only include successful segmentations
        }
        
        if not trends_by_stock:
            raise ValueError("Trend segmentation failed for all stocks. Check input data quality.")
        
        print(f"Successfully segmented {len(trends_by_stock)} stocks")
        
        # Create temporal framework - get all unique dates and stock IDs
        all_dates = pd.to_datetime(np.unique(cleaned_df['Date'])).sort_values()
        stock_ids = sorted(trends_by_stock.keys())
        
        # Initialize daily matrices for slopes and durations
        # Each cell [date, stock] will contain the trend value for that stock on that date
        daily_slopes = pd.DataFrame(
            index=all_dates, 
            columns=stock_ids, 
            dtype=float
        )
        daily_durations = pd.DataFrame(
            index=all_dates, 
            columns=stock_ids, 
            dtype=float
        )
        
        for stock_id, trends in tqdm(trends_by_stock.items(), desc="Filling daily matrices"):
            # For each trend segment, fill the corresponding date range with constant values
            for segment_idx in range(len(trends['slopes'])):
                start_date = trends['dates'][segment_idx]
                duration_days = trends['durations'][segment_idx]
                end_date = start_date + pd.to_timedelta(duration_days - 1, unit='D')
                
                # Fill the entire segment period with the same slope and duration values
                daily_slopes.loc[start_date:end_date, stock_id] = trends['slopes'][segment_idx]
                daily_durations.loc[start_date:end_date, stock_id] = trends['durations'][segment_idx]
        
        # Handle any missing values by forward/backward filling
        # This ensures no NaN values remain in the matrices
        daily_slopes.ffill(inplace=True)
        daily_slopes.bfill(inplace=True)
        daily_durations.ffill(inplace=True) 
        daily_durations.bfill(inplace=True)
        
        
        # Convert slope matrix to long format for adjacency matrix computation
        slopes_long = daily_slopes.reset_index().melt(
            id_vars='index', 
            var_name='StockID', 
            value_name='slope'
        )
        slopes_long.rename(columns={'index': 'Date'}, inplace=True)
        
        # Convert duration matrix to long format
        durations_long = daily_durations.reset_index().melt(
            id_vars='index', 
            var_name='StockID', 
            value_name='duration'
        )
        durations_long.rename(columns={'index': 'Date'}, inplace=True)
        
        # Create separate adjacency matrices for slopes and durations
        # These capture similarity between stocks based on their trend characteristics
        slope_adj_matrix = create_adjacency_matrix(
            slopes_long, 
            target_column='slope', 
            threshold=0.5
        )
        
        duration_adj_matrix = create_adjacency_matrix(
            durations_long, 
            target_column='duration', 
            threshold=0.5
        )
        
        # Combine the two adjacency matrices by averaging
        # This creates a unified similarity measure considering both slope and duration patterns
        adj_matrix = (slope_adj_matrix + duration_adj_matrix) / 2.0
        print(f"Created combined adjacency matrix of shape: {adj_matrix.shape}")
        
        # Determine training period for fitting scalers (avoid data leakage)
        train_end_idx = int(len(all_dates) * self.settings.TRAIN_SPLIT)
        train_dates = all_dates[:train_end_idx]
        
        # Fit scalers only on training data to prevent data leakage
        # Slopes: scaled to [-1, 1] to preserve sign information (uptrend/downtrend)
        # Durations: scaled to [0, 1] as they are always positive
        slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(
            daily_slopes.loc[train_dates]
        )
        duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(
            daily_durations.loc[train_dates]
        )
        
        scalers = {
            'slope': slope_scaler, 
            'duration': duration_scaler
        }
        
        # Apply scaling to entire dataset
        scaled_slopes = slope_scaler.transform(daily_slopes)
        scaled_durations = duration_scaler.transform(daily_durations)
        
        # Stack slopes and durations as separate features for each stock
        # Shape: [time_steps, num_stocks, num_features]
        graph_data = np.stack([scaled_slopes, scaled_durations], axis=-1)
        print(f"Created graph data with shape: {graph_data.shape}")
        
        X, y, sequence_dates = [], [], []
        input_window = self.settings.TREND_INPUT_WINDOW_SIZE
        total_sequence_length = input_window + 1  # input + 1 prediction step
        
        # Create overlapping sequences for time series prediction
        for i in range(len(graph_data) - total_sequence_length + 1):
            # Input: sequence of length 'input_window'
            X.append(graph_data[i:i + input_window, :, :])
            
            # Target: next time step after the input sequence
            y.append(graph_data[i + input_window, :, :])
            
            # The date corresponds to the prediction date (start of the target period)
            sequence_dates.append(all_dates[i + input_window])
        
        # Convert to numpy arrays with appropriate dtype for neural network training
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        sequence_dates = np.array(sequence_dates)
        
        print(f"Created {len(X)} training sequences")
        print(f"Input shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y, stock_ids, None, scalers, adj_matrix

    def _scale_pivoted_data_point(self, pivoted_df: pd.DataFrame, stock_ids: list):
        """Scales pivoted data for each stock individually (POINT mode)."""
        #we will scale each stock's features independently, using MinMaxScaler to ensure all features are in the range [0, 1]
        scalers = {}
        # Create a dictionary to hold scaled data for each stock
        scaled_data_dict = {}
        
        # Get the end date for training split because we are only supposed to scale the training data to avoid data leakage
        train_end_date = pivoted_df.index[int(len(pivoted_df) * self.settings.TRAIN_SPLIT)]

        for stock_id in tqdm(stock_ids, desc="Scaling features per stock"):
            # Extract the stock's features and scale them
            stock_features_df = pivoted_df.xs(stock_id, level='StockID', axis=1)
            #fit the scaler on the training data only, we are saving the minimum and maximum values of the training data to scale the test data later
            scaler = MinMaxScaler().fit(stock_features_df.loc[:train_end_date])
            scalers[stock_id] = scaler
            # Transform the stock's features and store in the dictionary
            scaled_data_dict[stock_id] = scaler.transform(stock_features_df)
        
        num_dates, num_nodes = len(pivoted_df), len(stock_ids)
        num_features = len(self.settings.FEATURE_COLUMNS)
        scaled_array = np.zeros((num_dates, num_nodes, num_features))
        # Fill the scaled array with the scaled data for each stock -> for all dates, put this stock's scaled data in the correct position
        for i, stock_id in enumerate(stock_ids):
            scaled_array[:, i, :] = scaled_data_dict[stock_id]
            
        return scalers, scaled_array