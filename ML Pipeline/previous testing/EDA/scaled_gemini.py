import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def create_features(df):
    """Adds new features to the DataFrame."""
    df_featured = df.copy()
    # Time-based features
    df_featured['month'] = df_featured.index.month
    df_featured['day_of_week'] = df_featured.index.dayofweek
    # Lag features
    df_featured['close_lag1'] = df_featured['close'].shift(1)
    # Rolling window features (moving averages)
    df_featured['close_ma7'] = df_featured['close'].rolling(window=7).mean()
    df_featured['close_ma30'] = df_featured['close'].rolling(window=30).mean()
    # Drop rows with NaNs created by new features
    df_featured.dropna(inplace=True)
    return df_featured

def plot_scaling_effect(original_data, scaled_data, feature_name, stock_name, set_name, save_path):
    """Visualizes the data distribution before and after Min-Max scaling."""
    plt.figure(figsize=(14, 6))
    
    # Plot Before Scaling
    plt.subplot(1, 2, 1)
    original_data[feature_name].hist(bins=50, color='blue', alpha=0.7)
    plt.title(f'Before Scaling: {feature_name} ({set_name})', fontsize=14)
    plt.xlabel('Original Value')
    plt.ylabel('Frequency')

    # Plot After Scaling
    plt.subplot(1, 2, 2)
    scaled_data[feature_name].hist(bins=50, color='green', alpha=0.7)
    plt.title(f'After Scaling: {feature_name} ({set_name})', fontsize=14)
    plt.xlabel('Scaled Value (0 to 1)')
    
    plt.suptitle(f'Min-Max Scaling Effect for {stock_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved scaling visualization to: {save_path}")

def process_stock_pipeline(excel_path, sheet_name, output_folder):
    """
    Runs the full pipeline: load, split, impute, feature engineer, scale, save, and plot.
    """
    print(f"\n{'='*60}")
    print(f"▶️ Starting Full Pipeline for: {sheet_name}")
    print(f"{'='*60}")

    # 1. Load Data
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    except Exception as e:
        print(f"⚠️ Could not read sheet '{sheet_name}'. Error: {e}"); return

    # 2. Split Data Chronologically (60:20:20)
    train_size = int(len(df) * 0.60)
    val_size = int(len(df) * 0.20)
    train_set = df.iloc[:train_size].copy()
    validation_set = df.iloc[train_size:(train_size + val_size)].copy()
    test_set = df.iloc[(train_size + val_size):].copy()

    # 3. Impute Missing Values (in isolation)
    for dataset in [train_set, validation_set, test_set]:
        dataset.fillna(method='ffill', inplace=True)
        dataset.fillna(method='bfill', inplace=True)
        
    # 4. Feature Engineering
    train_featured = create_features(train_set)
    validation_featured = create_features(validation_set)
    test_featured = create_features(test_set)
    print("✅ Feature engineering complete.")

    # 5. Min-Max Scaling (Best Practice)
    # The scaler is FIT ONLY on the training data.
    scaler = MinMaxScaler()
    # We fit the scaler and transform the training set
    train_scaled_np = scaler.fit_transform(train_featured)
    # We ONLY TRANSFORM the validation and test sets using the scaler from the training data.
    validation_scaled_np = scaler.transform(validation_featured)
    test_scaled_np = scaler.transform(test_featured)

    # Convert scaled numpy arrays back to DataFrames
    train_scaled = pd.DataFrame(train_scaled_np, index=train_featured.index, columns=train_featured.columns)
    validation_scaled = pd.DataFrame(validation_scaled_np, index=validation_featured.index, columns=validation_featured.columns)
    test_scaled = pd.DataFrame(test_scaled_np, index=test_featured.index, columns=test_featured.columns)
    print("✅ Min-Max scaling complete (fit on train, applied to all).")

    # 6. Save and Plot
    stock_folder = os.path.join(output_folder, sheet_name)
    os.makedirs(stock_folder, exist_ok=True)
    
    # Save final scaled data
    train_scaled.to_csv(os.path.join(stock_folder, f"{sheet_name}_train_scaled.csv"))
    validation_scaled.to_csv(os.path.join(stock_folder, f"{sheet_name}_validation_scaled.csv"))
    test_scaled.to_csv(os.path.join(stock_folder, f"{sheet_name}_test_scaled.csv"))
    print(f"✅ Saved final scaled data to folder: {stock_folder}")

    # Plot scaling effect on the 'close' price for each set
    plot_scaling_effect(train_featured, train_scaled, 'close', sheet_name, 'Train', os.path.join(stock_folder, f"{sheet_name}_train_scaling_plot.png"))
    plot_scaling_effect(validation_featured, validation_scaled, 'close', sheet_name, 'Validation', os.path.join(stock_folder, f"{sheet_name}_validation_scaling_plot.png"))
    plot_scaling_effect(test_featured, test_scaled, 'close', sheet_name, 'Test', os.path.join(stock_folder, f"{sheet_name}_test_scaling_plot.png"))


if __name__ == "__main__":
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    output_directory = "full_pipeline_output"
    os.makedirs(output_directory, exist_ok=True)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, cleaned_excel_file)

    if not os.path.exists(full_path):
        print(f"--- ERROR ---: File not found at {full_path}")
    else:
        xls = pd.ExcelFile(full_path)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} stocks to process. Output will be in '{output_directory}'.")
        
        for sheet in sheet_names:
            process_stock_pipeline(full_path, sheet, output_directory)
            
        print(f"\n🎉 All worksheets have been processed.")