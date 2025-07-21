import logging
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import ruptures as rpt
from multiprocessing import Pool, cpu_count
from .base_processor import BaseProcessor
from ..utils import print_header, create_adjacency_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def _segment_one_stock_with_dates_bottom_up(args):
    """Worker function to segment a stock's time series and return trends with dates."""
    stock_id, stock_df, target_column, penalty = args
    price_values = stock_df[target_column].values
    dates = stock_df['Date'].values
    if len(price_values) < 2: return stock_id, []

    #initialize a change point detection algorithm using the Bottom-Up method, it starts with the entire time series and iteratively merges segments to find the best fit and by best fit we mean we stop when merging would lose too much accuracy. it uses the l2 model which uses least squares to measure the fit of the segments
    algo = rpt.BottomUp(model="l2").fit(price_values)
    
    #predict the breakpoints in the time series using the penalty value to control how many segments we want -> lower penalty means more segments
    breakpoints = algo.predict(pen=penalty)
    
    #ensure start and end points are included
    all_bps = sorted(list(set([0] + breakpoints + [len(price_values)])))

    trends = []
    #loop through pairs of breakpoints to calculate slopes and durations
    for start, end in zip(all_bps[:-1], all_bps[1:]):
        #if the segment has more than one point, calculate the slope and duration
        if end - start > 1:
            slope = np.polyfit(np.arange(end - start), price_values[start:end], 1)[0]
            trends.append({
                'start_date': dates[start],
                'end_date': dates[end - 1],
                'slope': slope,
                'duration': end - start
            })
    return stock_id, trends

class GraphProcessor(BaseProcessor):
    """Processor for all graph-based models (GNN, STGNN, etc.)."""

    def process(self, df: pd.DataFrame):
        print_header(f"Stage 2: Graph Preprocessing ({self.mode} MODE)")
        cleaned_df = self._clean_data(df)

        if self.mode == "POINT":
            return self._process_point(cleaned_df)
        elif self.mode == "TREND":
            return self._process_trend(cleaned_df)
        else:
            raise ValueError(f"Invalid PREDICTION_MODE: {self.mode}")

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
        X, y = [], []
        in_win = self.settings.POINT_INPUT_WINDOW_SIZE
        out_win = self.settings.POINT_OUTPUT_WINDOW_SIZE
        target_col_idx = self.settings.FEATURE_COLUMNS.index(self.settings.TARGET_COLUMN)

        #we want to slide a window over the scaled time series data and grab the input and output sequences
        for i in tqdm(range(len(scaled_array) - (in_win + out_win) + 1), desc="Creating graph sequences"):
            #take a window of inputs for all the stocks and for all the features
            X.append(scaled_array[i : i + in_win, :, :])
            #take the output for the target column for all the stocks
            y.append(scaled_array[i + in_win : i + in_win + out_win, :, target_col_idx:target_col_idx+1])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), stock_ids, scalers, adj_matrix

    def _process_trend(self, cleaned_df: pd.DataFrame):
        """Handles trend prediction for graph models."""
        #for each stock we get a tuple of stock id, sdf, target column, and segmentation penalty
        task_args = [(sid, sdf, self.settings.TARGET_COLUMN, self.settings.SEGMENTATION_PENALTY) for sid, sdf in cleaned_df.groupby('StockID')]#group the cleaned df by stock id and start to work on that dataframe that was created
        # Use multiprocessing to segment trends in parallel so results contains tuples of (stock_id, trends)
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(_segment_one_stock_with_dates_bottom_up, task_args), total=len(task_args), desc="Segmenting Trends"))
        
        # create a dictionary of trends by stock ID, filtering out stocks with no trends
        trends_by_stock = {sid: trends for sid, trends in results if trends}
        if not trends_by_stock: raise ValueError("Trend segmentation failed for all stocks.")

        # Create a DataFrame to hold daily slopes and durations for each stock
        all_dates = pd.to_datetime(np.unique(cleaned_df['Date'])).sort_values()
        stock_ids = sorted(trends_by_stock.keys())
        daily_slopes = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)
        daily_durations = pd.DataFrame(index=all_dates, columns=stock_ids, dtype=float)


        for stock_id, trends in trends_by_stock.items():
            for trend in trends:
                # for a given trend, fill the daily slopes and durations DataFrames -> basically go by and fill the slope and duration for each date in the trend
                daily_slopes.loc[trend['start_date']:trend['end_date'], stock_id] = trend['slope']
                daily_durations.loc[trend['start_date']:trend['end_date'], stock_id] = trend['duration']
        
        # Fill NaN values with forward and backward fill to ensure continuity
        daily_slopes.ffill(inplace=True); daily_slopes.bfill(inplace=True)
        daily_durations.ffill(inplace=True); daily_durations.bfill(inplace=True)
        
        # Create adjacency matrix based on correlation of daily slopes and durations
        slopes_correlation_matrix = daily_slopes.corr()
        duration_correlation_matrix = daily_durations.corr()
        combined_correlation_matrix = (slopes_correlation_matrix + duration_correlation_matrix) / 2
        adj_matrix_np = (combined_correlation_matrix.abs() >= 0.5).astype(int).values
        np.fill_diagonal(adj_matrix_np, 0)
        adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32)

        # Scale slopes and durations globally
        train_end_idx = int(len(all_dates) * self.settings.TRAIN_SPLIT)
        train_dates = all_dates[:train_end_idx]
        
        slope_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(daily_slopes.loc[train_dates])
        duration_scaler = MinMaxScaler(feature_range=(0, 1)).fit(daily_durations.loc[train_dates])
        scalers = {'slope': slope_scaler, 'duration': duration_scaler}

        scaled_slopes = slope_scaler.transform(daily_slopes)
        scaled_durations = duration_scaler.transform(daily_durations)
        #take multiple arrays of the same shape and combines them along a new dimension to create a 3D array
        graph_data = np.stack([scaled_slopes, scaled_durations], axis=-1)

        X, y = [], []
        in_win = self.settings.TREND_INPUT_WINDOW_SIZE
        for i in range(len(graph_data) - in_win):
            X.append(graph_data[i : i + in_win, :, :])
            y.append(graph_data[i + in_win, :, :])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), stock_ids, scalers, adj_matrix

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