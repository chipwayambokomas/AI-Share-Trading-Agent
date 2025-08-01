import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# --- Segmentation Code (from previous example) ---

class Segment:
    """A class to represent a line segment in the time series."""
    def __init__(self, start_index, end_index, data):
        self.start_index = start_index
        self.end_index = end_index
        # Ensure data is a numpy array for calculations
        self.points = np.array(data[start_index : end_index + 1])
        self.line = self._calculate_best_fit_line()

    def _calculate_best_fit_line(self):
        """Calculates the best-fit line (linear regression) for the segment's points."""
        x = np.arange(self.start_index, self.end_index + 1)
        y = self.points
        if len(x) < 2:
            # Handle segments with less than 2 points
            return np.array([0, y[0] if len(y) > 0 else 0])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.array([m, c])

    def get_error(self):
        """Calculates the sum of squared errors for the segment."""
        x = np.arange(self.start_index, self.end_index + 1)
        y = self.points
        m, c = self.line
        y_fit = m * x + c
        return np.sum((y - y_fit) ** 2)
        
    @property
    def slope(self):
        return self.line[0]
        
    @property
    def duration(self):
        return (self.end_index - self.start_index) + 1

    def __repr__(self):
        return f"Segment({self.start_index}, {self.end_index}, slope={self.slope:.2f}, duration={self.duration})"

def calculate_merge_cost(segment1, segment2, original_data):
    """Calculates the error of a potential merged segment."""
    start_index = segment1.start_index
    end_index = segment2.end_index
    merged_segment = Segment(start_index, end_index, original_data)
    return merged_segment.get_error()

def bottom_up_segmentation(data_series, max_error):
    """Performs bottom-up time series segmentation on a pandas Series."""
    data = data_series.values
    segments = [Segment(i, i + 1, data) for i in range(len(data) - 1)]

    if not segments:
        return []

    merge_costs = [calculate_merge_cost(segments[i], segments[i+1], data) for i in range(len(segments) - 1)]

    while merge_costs and min(merge_costs) < max_error:
        min_cost_index = np.argmin(merge_costs)
        segment1 = segments[min_cost_index]
        segment2 = segments[min_cost_index + 1]
        
        new_segment = Segment(segment1.start_index, segment2.end_index, data)
        segments[min_cost_index] = new_segment
        del segments[min_cost_index + 1]
        del merge_costs[min_cost_index]

        if min_cost_index > 0:
            merge_costs[min_cost_index - 1] = calculate_merge_cost(segments[min_cost_index - 1], new_segment, data)
        if min_cost_index < len(segments) - 1:
            merge_costs[min_cost_index] = calculate_merge_cost(new_segment, segments[min_cost_index + 1], data)
            
    return segments

def plot_segmentation(data_series, segments):
    """Plots the original data series and the segmented approximation."""
    plt.figure(figsize=(15, 6))
    plt.plot(data_series.index, data_series.values, label='Original Time Series', color='blue', alpha=0.7)
    
    for seg in segments:
        # Get the corresponding dates from the original series index
        start_date = data_series.index[seg.start_index]
        end_date = data_series.index[seg.end_index]
        
        # Create a date range for plotting
        x_dates = pd.date_range(start=start_date, end=end_date)
        
        # We need numerical x-values for the line equation
        x_numeric = np.arange(len(x_dates))
        
        m, c = seg.line
        
        # We need to adjust the intercept 'c' because our x_numeric starts from 0
        # The original equation was y = m * original_index + c
        # The new equation is y = m * (x_numeric) + new_c
        # new_c = m * original_start_index + c
        adjusted_c = m * seg.start_index + c
        
        y_fit = m * np.arange(seg.start_index, seg.end_index + 1) + c
        
        plt.plot(data_series.index[seg.start_index:seg.end_index+1], y_fit, 'r-', linewidth=3)

    plt.title('Bottom-Up Time Series Segmentation')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Your Original Data Processing Script (Modified) ---

def create_features(df, segments=None):
    """
    Adds new features to the DataFrame, including segmentation features.
    """
    df_featured = df.copy()
    df_featured['month'] = df_featured.index.month
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['close_lag1'] = df_featured['close'].shift(1)
    df_featured['close_ma7'] = df_featured['close'].rolling(window=7).mean()
    df_featured['close_ma30'] = df_featured['close'].rolling(window=30).mean()
    
    if segments:
        # Create columns for segment features and initialize with NaN
        df_featured['segment_slope'] = np.nan
        df_featured['segment_duration'] = np.nan
        
        # Map each data point to its corresponding segment's features
        for seg in segments:
            df_featured.loc[df_featured.index[seg.start_index:seg.end_index+1], 'segment_slope'] = seg.slope
            df_featured.loc[df_featured.index[seg.start_index:seg.end_index+1], 'segment_duration'] = seg.duration

    df_featured.dropna(inplace=True)
    return df_featured

def process_stock_for_aggregation(excel_path, sheet_name, output_folder, is_first_stock=False):
    """
    Processes a single stock and returns its scaled train, validation, and test sets.
    """
    print(f"\n▶️ Processing: {sheet_name}")

    # 1. Load Data
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # Ensure we only have the 'close' column for segmentation
        df = df[['close']].copy()
    except Exception as e:
        print(f"⚠️ Could not read sheet '{sheet_name}'. Skipping. Error: {e}")
        return None, None, None

    # 2. Split and Impute
    train_size = int(len(df) * 0.60)
    val_size = int(len(df) * 0.20)
    train_set = df.iloc[:train_size].copy()
    validation_set = df.iloc[train_size:(train_size + val_size)].copy()
    test_set = df.iloc[(train_size + val_size):].copy()
    for dataset in [train_set, validation_set, test_set]:
        dataset.fillna(method='ffill', inplace=True)
        dataset.fillna(method='bfill', inplace=True)

    # 3. Perform Segmentation on the 'close' price
    # Tune this value based on your data's scale and desired granularity
    max_approximation_error = 100.0  
    train_segments = bottom_up_segmentation(train_set['close'], max_approximation_error)
    validation_segments = bottom_up_segmentation(validation_set['close'], max_approximation_error)
    test_segments = bottom_up_segmentation(test_set['close'], max_approximation_error)
    
    print(f"   - Segmented training data into {len(train_segments)} segments.")
    
    # Plot segmentation for the first stock for verification
    if is_first_stock:
        print("   - Plotting segmentation for the first stock's training data...")
        plot_segmentation(train_set['close'], train_segments)

    # 4. Feature Engineering (with segmentation features)
    train_featured = create_features(train_set, train_segments)
    validation_featured = create_features(validation_set, validation_segments)
    test_featured = create_features(test_set, test_segments)

    # 5. Min-Max Scaling
    # Identify feature columns (excluding the target 'close')
    feature_cols = [col for col in train_featured.columns if col != 'close']
    scaler = MinMaxScaler()
    
    # Fit scaler only on the training features
    scaler.fit(train_featured[feature_cols])
    
    # Transform all sets
    train_featured[feature_cols] = scaler.transform(train_featured[feature_cols])
    validation_featured[feature_cols] = scaler.transform(validation_featured[feature_cols])
    test_featured[feature_cols] = scaler.transform(test_featured[feature_cols])

    # 6. Add a ticker column
    train_featured['stock_ticker'] = sheet_name
    validation_featured['stock_ticker'] = sheet_name
    test_featured['stock_ticker'] = sheet_name
    
    print(f"✅ Finished processing {sheet_name}")
    return train_featured, validation_featured, test_featured

if __name__ == "__main__":
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    output_directory = "master_datasets_segmented"
    os.makedirs(output_directory, exist_ok=True)

    # Use a placeholder for the file path if it's not in the same directory
    # In a real scenario, ensure the script can find this file.
    script_directory = "." 
    full_path = os.path.join(script_directory, cleaned_excel_file)

    if not os.path.exists(full_path):
        print(f"--- ERROR ---: File not found at {full_path}")
        print("Please ensure 'JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx' is in the same directory as the script.")
    else:
        xls = pd.ExcelFile(full_path)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} stocks to process.")
        
        all_train_sets, all_validation_sets, all_test_sets = [], [], []

        for i, sheet in enumerate(sheet_names):
            # Pass is_first_stock=True only for the first iteration to generate a plot
            train_df, val_df, test_df = process_stock_for_aggregation(full_path, sheet, output_directory, is_first_stock=(i==0))
            if train_df is not None:
                all_train_sets.append(train_df)
                all_validation_sets.append(val_df)
                all_test_sets.append(test_df)
        
        if all_train_sets:
            master_train_set = pd.concat(all_train_sets)
            master_validation_set = pd.concat(all_validation_sets)
            master_test_set = pd.concat(all_test_sets)

            master_train_set.to_csv(os.path.join(output_directory, "master_train_set.csv"))
            master_validation_set.to_csv(os.path.join(output_directory, "master_validation_set.csv"))
            master_test_set.to_csv(os.path.join(output_directory, "master_test_set.csv"))

            print(f"\n{'='*60}")
            print(f"🎉 Master datasets with segmentation features created in '{output_directory}'.")
            print(f"Total training rows: {len(master_train_set)}")
            print(f"Total validation rows: {len(master_validation_set)}")
            print(f"Total test rows: {len(master_test_set)}")
            print(f"{'='*60}")
        else:
            print("No data was processed.")
