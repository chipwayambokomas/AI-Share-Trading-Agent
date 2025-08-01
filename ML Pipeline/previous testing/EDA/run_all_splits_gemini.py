import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

def process_and_split_stock(excel_path, sheet_name, output_folder):
    """
    Loads, splits, and saves the data for a single stock.
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

    # 2. Split Data Chronologically (60:20:20)
    total_rows = len(df)
    train_size = int(total_rows * 0.60)
    validation_size = int(total_rows * 0.20)
    
    train_end_idx = train_size
    validation_end_idx = train_size + validation_size

    train_set = df.iloc[:train_end_idx]
    validation_set = df.iloc[train_end_idx:validation_end_idx]
    test_set = df.iloc[validation_end_idx:]

    print(f"Splitting into {len(train_set)} (train), {len(validation_set)} (val), {len(test_set)} (test) rows.")

    # 3. Save the Split Data to CSV Files
    # Create a subfolder for this stock's data
    stock_folder = os.path.join(output_folder, sheet_name)
    os.makedirs(stock_folder, exist_ok=True)
    
    train_set.to_csv(os.path.join(stock_folder, f"{sheet_name}_train_set.csv"))
    validation_set.to_csv(os.path.join(stock_folder, f"{sheet_name}_validation_set.csv"))
    test_set.to_csv(os.path.join(stock_folder, f"{sheet_name}_test_set.csv"))
    print(f"✅ Saved split data to folder: {stock_folder}")

    # 4. Save a Plot of the Split
    plt.figure(figsize=(16, 8))
    plt.title(f'Chronological Data Split for {sheet_name}', fontsize=16)
    plt.plot(train_set['close'], label='Training Set (60%)', color='blue')
    plt.plot(validation_set['close'], label='Validation Set (20%)', color='orange')
    plt.plot(test_set['close'], label='Test Set (20%)', color='green')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(stock_folder, f"{sheet_name}_split_plot.png")
    plt.savefig(plot_filename)
    plt.close() # Close the plot to prevent it from popping up
    print(f"✅ Saved split visualization to: {plot_filename}")


if __name__ == "__main__":
    # Define file and folder names
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    output_directory = "split_stock_data"
    
    # Create the main output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get the full path for the input file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, cleaned_excel_file)

    if not os.path.exists(full_path):
        print(f"--- ERROR ---")
        print(f"The file was not found at: {full_path}")
    else:
        xls = pd.ExcelFile(full_path)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} stocks to process. Output will be saved in '{output_directory}'.")
        
        for sheet in sheet_names:
            process_and_split_stock(full_path, sheet, output_directory)
            
        print(f"\n🎉 All worksheets have been processed.")