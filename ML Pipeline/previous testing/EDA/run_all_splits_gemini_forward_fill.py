import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

def plot_forward_fill_effect(original_series, filled_series, title, save_path):
    """
    Visualizes the effect of forward filling on a time series.
    It only plots the changes where NaNs were filled.
    """
    # Find the indices where the original data was missing
    missing_indices = original_series[original_series.isnull()].index
    
    # If there were no missing values, there's nothing to plot.
    if missing_indices.empty:
        print(f"No missing values found in '{title}'. Plot not needed.")
        return

    print(f"Visualizing {len(missing_indices)} filled values for '{title}'...")

    plt.figure(figsize=(16, 8))
    # Plot the whole series as a line for context
    plt.plot(filled_series, label='Full Series', color='gray', alpha=0.5)
    # Highlight the specific points that were filled
    plt.scatter(
        missing_indices,
        filled_series.loc[missing_indices],
        color='red',
        marker='o',
        s=50, # size of the marker
        label='Forward Filled Values'
    )
    plt.title(f'Forward Fill Effect on {title}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved forward fill plot to: {save_path}")


def process_and_split_stock(excel_path, sheet_name, output_folder):
    """
    Loads, splits, imputes, and saves the data for a single stock.
    """
    print(f"\n--- Processing: {sheet_name} ---")

    # 1. Load Data
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    except Exception as e:
        print(f"⚠️ Could not read or process sheet '{sheet_name}'. Error: {e}")
        return

    # 2. Split Data Chronologically
    train_size = int(len(df) * 0.60)
    validation_size = int(len(df) * 0.20)
    train_set = df.iloc[:train_size].copy()
    validation_set = df.iloc[train_size:(train_size + validation_size)].copy()
    test_set = df.iloc[(train_size + validation_size):].copy()

    # --- 3. Forward Fill in Isolation and Plot Changes ---
    stock_folder = os.path.join(output_folder, sheet_name)
    os.makedirs(stock_folder, exist_ok=True)

    # Process each set (train, validation, test)
    for dataset, name in [(train_set, "Training"), (validation_set, "Validation"), (test_set, "Test")]:
        original_close = dataset['close'].copy() # Keep a copy before filling
        
        # Apply forward fill, then backfill for any leading NaNs
        dataset['close'].fillna(method='ffill', inplace=True)
        dataset.fillna(method='bfill', inplace=True)
        
        # Plot the effect of the fill
        plot_title = f"{sheet_name} - {name} Set"
        plot_save_path = os.path.join(stock_folder, f"{sheet_name}_{name.lower()}_ffill_plot.png")
        plot_forward_fill_effect(original_close, dataset['close'], plot_title, plot_save_path)

    # --- 4. Save the Split and Imputed Data ---
    train_set.to_csv(os.path.join(stock_folder, f"{sheet_name}_train_set.csv"))
    validation_set.to_csv(os.path.join(stock_folder, f"{sheet_name}_validation_set.csv"))
    test_set.to_csv(os.path.join(stock_folder, f"{sheet_name}_test_set.csv"))
    print(f"✅ Saved split and imputed data to folder: {stock_folder}")


if __name__ == "__main__":
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    output_directory = "split_stock_data_with_ffill"
    os.makedirs(output_directory, exist_ok=True)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, cleaned_excel_file)

    if not os.path.exists(full_path):
        print(f"--- ERROR ---: File not found at {full_path}")
    else:
        xls = pd.ExcelFile(full_path)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} stocks to process. Output will be saved in '{output_directory}'.")
        
        for sheet in sheet_names:
            process_and_split_stock(full_path, sheet, output_directory)
            
        print(f"\n🎉 All worksheets have been processed.")