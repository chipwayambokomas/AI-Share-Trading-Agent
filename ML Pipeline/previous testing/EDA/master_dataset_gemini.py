import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def create_features(df):
    """Adds new features to the DataFrame."""
    df_featured = df.copy()
    df_featured['month'] = df_featured.index.month
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['close_lag1'] = df_featured['close'].shift(1)
    df_featured['close_ma7'] = df_featured['close'].rolling(window=7).mean()
    df_featured['close_ma30'] = df_featured['close'].rolling(window=30).mean()
    df_featured.dropna(inplace=True)
    return df_featured

def process_stock_for_aggregation(excel_path, sheet_name, output_folder):
    """
    Processes a single stock and returns its scaled train, validation, and test sets.
    """
    print(f"\n▶️ Processing: {sheet_name}")

    # 1. Load Data
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    except Exception as e:
        print(f"⚠️ Could not read sheet '{sheet_name}'. Skipping. Error: {e}")
        return None, None, None

    # 2. Split and Impute
    train_size = int(len(df) * 0.60); val_size = int(len(df) * 0.20)
    train_set = df.iloc[:train_size].copy()
    validation_set = df.iloc[train_size:(train_size + val_size)].copy()
    test_set = df.iloc[(train_size + val_size):].copy()
    for dataset in [train_set, validation_set, test_set]:
        dataset.fillna(method='ffill', inplace=True); dataset.fillna(method='bfill', inplace=True)

    # 3. Feature Engineering
    train_featured = create_features(train_set)
    validation_featured = create_features(validation_set)
    test_featured = create_features(test_set)

    # 4. Min-Max Scaling
    scaler = MinMaxScaler()
    train_scaled_np = scaler.fit_transform(train_featured)
    validation_scaled_np = scaler.transform(validation_featured)
    test_scaled_np = scaler.transform(test_featured)

    train_scaled = pd.DataFrame(train_scaled_np, index=train_featured.index, columns=train_featured.columns)
    validation_scaled = pd.DataFrame(validation_scaled_np, index=validation_featured.index, columns=validation_featured.columns)
    test_scaled = pd.DataFrame(test_scaled_np, index=test_featured.index, columns=test_featured.columns)
    
    # 5. Add a ticker column to identify the stock after combining
    train_scaled['stock_ticker'] = sheet_name
    validation_scaled['stock_ticker'] = sheet_name
    test_scaled['stock_ticker'] = sheet_name
    
    print(f"✅ Finished processing {sheet_name}")
    return train_scaled, validation_scaled, test_scaled


if __name__ == "__main__":
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    output_directory = "master_datasets"
    os.makedirs(output_directory, exist_ok=True)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, cleaned_excel_file)

    if not os.path.exists(full_path):
        print(f"--- ERROR ---: File not found at {full_path}")
    else:
        xls = pd.ExcelFile(full_path)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} stocks to process.")
        
        # --- Aggregation Step ---
        # Initialize empty lists to hold the data from all stocks
        all_train_sets = []
        all_validation_sets = []
        all_test_sets = []

        # Loop through each sheet, process it, and collect the results
        for sheet in sheet_names:
            train_df, val_df, test_df = process_stock_for_aggregation(full_path, sheet, output_directory)
            if train_df is not None:
                all_train_sets.append(train_df)
                all_validation_sets.append(val_df)
                all_test_sets.append(test_df)
        
        # --- Final Combination ---
        # Concatenate all the individual DataFrames into master sets
        if all_train_sets:
            master_train_set = pd.concat(all_train_sets)
            master_validation_set = pd.concat(all_validation_sets)
            master_test_set = pd.concat(all_test_sets)

            # --- Save the Master Datasets ---
            master_train_set.to_csv(os.path.join(output_directory, "master_train_set.csv"))
            master_validation_set.to_csv(os.path.join(output_directory, "master_validation_set.csv"))
            master_test_set.to_csv(os.path.join(output_directory, "master_test_set.csv"))

            print(f"\n{'='*60}")
            print(f"🎉 Master datasets created successfully in the '{output_directory}' folder.")
            print(f"Total training rows: {len(master_train_set)}")
            print(f"Total validation rows: {len(master_validation_set)}")
            print(f"Total test rows: {len(master_test_set)}")
            print(f"{'='*60}")
        else:
            print("No data was processed.")