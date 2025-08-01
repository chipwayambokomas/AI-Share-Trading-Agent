import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

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

def plot_decomposition(series, title, save_path):
    """Performs seasonal decomposition and plots the components."""
    # Using a period of 21 for monthly seasonality in business days
    result = seasonal_decompose(series, model='additive', period=21)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(411)
    plt.plot(result.observed, label='Observed')
    plt.legend(loc='upper left')
    
    plt.subplot(412)
    plt.plot(result.trend, label='Trend', color='blue')
    plt.legend(loc='upper left')
    
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal', color='green')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(result.resid, label='Residual', color='red')
    plt.legend(loc='upper left')
    
    plt.suptitle(f'Trend Decomposition for {title}', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved decomposition plot to: {save_path}")

def process_stock_pipeline(excel_path, sheet_name, output_folder):
    """
    Runs the full pipeline: load, split, impute, feature engineer, scale, decompose, save, and plot.
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

    # 2. Split Data
    train_size = int(len(df) * 0.60); val_size = int(len(df) * 0.20)
    train_set = df.iloc[:train_size].copy()
    validation_set = df.iloc[train_size:(train_size + val_size)].copy()
    test_set = df.iloc[(train_size + val_size):].copy()

    # 3. Impute Missing Values
    for dataset in [train_set, validation_set, test_set]:
        dataset.fillna(method='ffill', inplace=True); dataset.fillna(method='bfill', inplace=True)
        
    # 4. Feature Engineering
    train_featured = create_features(train_set)
    validation_featured = create_features(validation_set)
    test_featured = create_features(test_set)
    print("✅ Feature engineering complete.")

    # --- Create subfolder for this stock's output ---
    stock_folder = os.path.join(output_folder, sheet_name)
    os.makedirs(stock_folder, exist_ok=True)

    # 5. Visualize Decomposition BEFORE Scaling
    print("\nVisualizing decomposition before scaling...")
    plot_decomposition(train_featured['close'], f'{sheet_name} - Train Set (Before Scaling)', os.path.join(stock_folder, f"{sheet_name}_train_decomp_before.png"))
    plot_decomposition(validation_featured['close'], f'{sheet_name} - Validation Set (Before Scaling)', os.path.join(stock_folder, f"{sheet_name}_validation_decomp_before.png"))
    plot_decomposition(test_featured['close'], f'{sheet_name} - Test Set (Before Scaling)', os.path.join(stock_folder, f"{sheet_name}_test_decomp_before.png"))

    # 6. Min-Max Scaling
    scaler = MinMaxScaler()
    train_scaled_np = scaler.fit_transform(train_featured)
    validation_scaled_np = scaler.transform(validation_featured)
    test_scaled_np = scaler.transform(test_featured)

    train_scaled = pd.DataFrame(train_scaled_np, index=train_featured.index, columns=train_featured.columns)
    validation_scaled = pd.DataFrame(validation_scaled_np, index=validation_featured.index, columns=validation_featured.columns)
    test_scaled = pd.DataFrame(test_scaled_np, index=test_featured.index, columns=test_featured.columns)
    print("\n✅ Min-Max scaling complete.")

    # 7. Visualize Decomposition AFTER Scaling
    print("\nVisualizing decomposition after scaling...")
    plot_decomposition(train_scaled['close'], f'{sheet_name} - Train Set (After Scaling)', os.path.join(stock_folder, f"{sheet_name}_train_decomp_after.png"))
    plot_decomposition(validation_scaled['close'], f'{sheet_name} - Validation Set (After Scaling)', os.path.join(stock_folder, f"{sheet_name}_validation_decomp_after.png"))
    plot_decomposition(test_scaled['close'], f'{sheet_name} - Test Set (After Scaling)', os.path.join(stock_folder, f"{sheet_name}_test_decomp_after.png"))

    # 8. Save Final Scaled Data
    train_scaled.to_csv(os.path.join(stock_folder, f"{sheet_name}_train_scaled.csv"))
    validation_scaled.to_csv(os.path.join(stock_folder, f"{sheet_name}_validation_scaled.csv"))
    test_scaled.to_csv(os.path.join(stock_folder, f"{sheet_name}_test_scaled.csv"))
    print(f"\n✅ Saved final scaled data to folder: {stock_folder}")


if __name__ == "__main__":
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    output_directory = "full_pipeline_output_with_decomp"
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