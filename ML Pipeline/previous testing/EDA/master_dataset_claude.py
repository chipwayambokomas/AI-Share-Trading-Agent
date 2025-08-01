import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Data Loading and Splitting (Unchanged) ---

def load_and_split_jse_data(excel_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Load cleaned JSE data from Excel file and split chronologically into train/val/test sets.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        print(f"Loading data from {len(sheet_names)} sheets...")
        
        split_data = {}
        for sheet_name in sheet_names:
            print(f"\n--- Processing Sheet: {sheet_name} ---")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            else:
                print(f"⚠️ Warning: No 'Date' column found in sheet '{sheet_name}'. Using row order.")
            
            total_rows = len(df)
            if total_rows == 0:
                print(f"⚠️ Warning: No data in sheet '{sheet_name}'. Skipping.")
                continue
            
            train_end = int(total_rows * train_ratio)
            val_end = int(total_rows * (train_ratio + val_ratio))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            split_data[sheet_name] = {
                'train': train_df, 'validation': val_df, 'test': test_df, 'full': df.copy()
            }
            
            print(f"Total rows: {total_rows}")
            print(f"Training set: {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)")
            print(f"Validation set: {len(val_df)} rows ({len(val_df)/total_rows*100:.1f}%)")
            print(f"Test set: {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)")
            
            if 'Date' in df.columns and not train_df.empty and not val_df.empty and not test_df.empty:
                print(f"Date ranges:")
                print(f"  Training: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
                print(f"  Validation: {val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
                print(f"  Test: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
        
        return split_data
        
    except FileNotFoundError:
        print(f"--- ERROR --- The file was not found at: {excel_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def apply_forward_fill(df):
    """Apply forward fill to handle missing values in OHLCV data."""
    df_filled = df.copy()
    numerical_cols = ['open', 'high', 'low', 'close', 'vwap']
    existing_numerical_cols = [col for col in numerical_cols if col in df_filled.columns]
    
    if existing_numerical_cols:
        df_filled[existing_numerical_cols] = df_filled[existing_numerical_cols].ffill()
        df_filled[existing_numerical_cols] = df_filled[existing_numerical_cols].bfill()
    
    return df_filled

def save_splits_to_csv(split_data, output_dir):
    """
    Save the train/val/test splits to separate CSV files with forward fill applied.
    """
    data_dir = os.path.join(output_dir, "1_forward_filled_data")
    os.makedirs(data_dir, exist_ok=True)
    filled_data = {}
    
    for sheet_name, data in split_data.items():
        filled_data[sheet_name] = {}
        clean_name = sheet_name.replace(' ', '_').replace('/', '_').replace('.', '_')
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in data and not data[split_name].empty:
                df_original = data[split_name]
                df_filled = apply_forward_fill(df_original)
                
                df_filled.dropna(how='all', subset=['open', 'high', 'low', 'close', 'vwap'], inplace=True)
                
                if df_filled.empty:
                    continue

                filled_data[sheet_name][split_name] = df_filled
                
                csv_path = os.path.join(data_dir, f"{clean_name}_{split_name}.csv")
                df_filled.to_csv(csv_path, index=False)
    
    print(f"✅ Forward-filled data saved in: {data_dir}")
    return filled_data

# --- Scaling and Plotting Functions (Unchanged) ---

def scale_data(filled_split_data, columns_to_scale=['open', 'high', 'low', 'close', 'vwap']):
    """
    Applies Min-Max scaling based on the specified logic.
    """
    print("\n=== Scaling Data ===")
    scaled_data = {}
    scalers = {}

    for symbol, splits in filled_split_data.items():
        train_df = splits.get('train')
        val_df = splits.get('validation')
        test_df = splits.get('test')

        if train_df is None or val_df is None or test_df is None:
            continue

        scaled_data[symbol] = {}
        scalers[symbol] = {}
        
        scaler_train = MinMaxScaler()
        valid_cols = [col for col in columns_to_scale if col in train_df.columns]
        if not valid_cols:
            continue

        scaler_train.fit(train_df[valid_cols])
        
        train_scaled = train_df.copy()
        train_scaled[valid_cols] = scaler_train.transform(train_df[valid_cols])
        scaled_data[symbol]['train'] = train_scaled
        
        val_scaled = val_df.copy()
        val_scaled[valid_cols] = scaler_train.transform(val_df[valid_cols])
        scaled_data[symbol]['validation'] = val_scaled
        
        scalers[symbol]['scaler_train_to_val'] = scaler_train

        scaler_val = MinMaxScaler()
        scaler_val.fit(val_df[valid_cols])
        
        test_scaled = test_df.copy()
        test_scaled[valid_cols] = scaler_val.transform(test_df[valid_cols])
        scaled_data[symbol]['test'] = test_scaled
        
        scalers[symbol]['scaler_val_to_test'] = scaler_val

    return scaled_data, scalers

def save_scaled_data(scaled_data, original_data, output_dir):
    """Saves scaled data to CSVs and generates comparison plots."""
    data_dir = os.path.join(output_dir, "2_scaled_data")
    plots_dir = os.path.join(output_dir, "3_scaling_comparison_plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    for symbol, splits in scaled_data.items():
        for split_name, df_scaled in splits.items():
            clean_symbol = symbol.replace(' ', '_').replace('/', '_').replace('.', '_')
            
            csv_path = os.path.join(data_dir, f"{clean_symbol}_{split_name}_scaled.csv")
            df_scaled.to_csv(csv_path, index=False)
            
            df_original = original_data[symbol][split_name]
            plot_scaling_comparison(df_original, df_scaled, symbol, split_name, plots_dir)
    
    print(f"✅ Scaled data saved in: {data_dir}")
    print(f"📊 Scaling comparison plots saved in: {plots_dir}")

def plot_scaling_comparison(df_before, df_after, symbol, split_name, plots_dir):
    """Creates a plot comparing data before and after scaling."""
    if 'close' not in df_before.columns or 'Date' not in df_before.columns:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Scaling Comparison for {symbol} - {split_name.capitalize()} Set', fontsize=16, fontweight='bold')
    
    ax1.plot(df_before['Date'], df_before['close'], label='Original Close Price', color='blue')
    ax1.set_title('Before Scaling')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.5)
    ax1.legend()
    
    ax2.plot(df_after['Date'], df_after['close'], label='Scaled Close Price', color='green')
    ax2.set_title('After Scaling (Min-Max)')
    ax2.set_ylabel('Scaled Value (0 to 1)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.5)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f"{symbol.replace(' ', '_')}_{split_name}_scaling_comparison.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()

# --- Trend Decomposition Visualization (Unchanged) ---

def plot_trend_decomposition(df, column, symbol, split_name, output_dir, is_scaled=False):
    """
    Performs and plots seasonal decomposition with a dynamic period.
    """
    n_obs = len(df)
    if n_obs > 252:
        period = 21
    elif n_obs > 60:
        period = 5
    else:
        period = 2
        
    if n_obs <= 2 * period:
        print(f"  ⚠️ Skipping decomposition for {symbol} {split_name}: not enough data points ({n_obs}) for period ({period}).")
        return

    df_indexed = df.set_index('Date')
    model_type = 'additive' if is_scaled else 'multiplicative'
    
    try:
        result = seasonal_decompose(df_indexed[column], model=model_type, period=period)
        
        fig = result.plot()
        fig.set_size_inches(12, 8)
        
        title_suffix = 'Scaled' if is_scaled else 'Original'
        model_info = f"(Model: {model_type.capitalize()}, Period: {period})"
        fig.suptitle(f"Trend Decomposition for {symbol} - {split_name.capitalize()} Set ({title_suffix})\n{model_info}", 
                     y=1.02, fontsize=16)
        
        subfolder = "5_decomposition_after_scaling" if is_scaled else "4_decomposition_before_scaling"
        plots_dir = os.path.join(output_dir, subfolder)
        os.makedirs(plots_dir, exist_ok=True)
        
        clean_symbol = symbol.replace(' ', '_').replace('/', '_').replace('.', '_')
        plot_filename = f"{clean_symbol}_{split_name}_decomposition.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"  ❌ Error during decomposition for {symbol} {split_name}: {e}")

def run_analysis_visualizations(filled_split_data, scaled_data, output_dir):
    """
    Orchestrates the creation of all analysis plots, including decomposition.
    """
    print("\n=== Generating Trend Decomposition Plots ===")
    
    for symbol, splits in filled_split_data.items():
        print(f"--- Generating plots for {symbol} ---")
        for split_name, df_before in splits.items():
            if df_before.empty or 'close' not in df_before.columns:
                continue
            
            df_scaled = scaled_data.get(symbol, {}).get(split_name)
            if df_scaled is None:
                continue

            print(f"  Plotting {split_name} (Before Scaling)...")
            plot_trend_decomposition(df_before, 'close', symbol, split_name, output_dir, is_scaled=False)
            
            print(f"  Plotting {split_name} (After Scaling)...")
            plot_trend_decomposition(df_scaled, 'close', symbol, split_name, output_dir, is_scaled=True)
    
    print(f"\n📊 Trend decomposition plots saved in subfolders within: {output_dir}")

# --- NEW: Function to Combine Datasets ---

def create_and_save_combined_datasets(split_data_dict, output_dir, data_label):
    """
    Combines all train, validation, and test sets across all symbols into
    single DataFrames and saves them to CSV.

    Args:
        split_data_dict (dict): Dictionary containing the split data (e.g., filled or scaled).
        output_dir (str): The main output directory.
        data_label (str): A label for the filename, e.g., "filled" or "scaled".
    """
    print(f"\n=== Combining All '{data_label.capitalize()}' Datasets ===")
    combined_dir = os.path.join(output_dir, "6_combined_datasets")
    os.makedirs(combined_dir, exist_ok=True)

    for split_name in ['train', 'validation', 'test']:
        list_of_dfs = []
        
        # Gather all dataframes for the current split type
        for symbol, splits in split_data_dict.items():
            if split_name in splits and not splits[split_name].empty:
                df = splits[split_name].copy()
                # Add the symbol column to identify the stock
                df['symbol'] = symbol 
                list_of_dfs.append(df)
        
        if not list_of_dfs:
            print(f"  No data found for combined '{split_name}' set. Skipping.")
            continue

        # Concatenate them into one big dataframe
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        
        # Sort by date and then symbol for chronological order
        if 'Date' in combined_df.columns:
            combined_df = combined_df.sort_values(by=['Date', 'symbol']).reset_index(drop=True)

        # Save the combined dataframe to a CSV file
        filename = f"combined_{split_name}_{data_label}.csv"
        filepath = os.path.join(combined_dir, filename)
        combined_df.to_csv(filepath, index=False)
        
        print(f"  ✅ Saved combined '{split_name}' set to {filepath} ({len(combined_df)} rows)")

# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Define file paths and the main output directory
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_excel_path = os.path.join(script_directory, cleaned_excel_file)
    
    main_output_dir = "jse_analysis_output"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 2. Load and split the data chronologically
    print("=== 1. Loading and Splitting JSE Data ===")
    raw_split_data = load_and_split_jse_data(full_excel_path)
    
    if raw_split_data:
        # 3. Apply forward fill and save the cleaned, unscaled splits
        print("\n=== 2. Applying Forward Fill and Saving Base Splits ===")
        filled_split_data = save_splits_to_csv(raw_split_data, output_dir=main_output_dir)
        
        # 4. Scale the data using the specified logic
        scaled_data, scalers = scale_data(filled_split_data)
        
        # 5. Save the scaled data and create before/after comparison plots
        print("\n=== 3. Saving Scaled Data and Generating Comparison Plots ===")
        save_scaled_data(scaled_data, filled_split_data, output_dir=main_output_dir)
        
        # 6. Run decomposition visualizations for both original and scaled data
        run_analysis_visualizations(filled_split_data, scaled_data, output_dir=main_output_dir)
        
        # 7. NEW: Create and save combined datasets for both filled and scaled data
        create_and_save_combined_datasets(filled_split_data, main_output_dir, data_label="filled")
        create_and_save_combined_datasets(scaled_data, main_output_dir, data_label="scaled")

        # 8. Show what was created
        print("\n=== All Tasks Complete! Files Created in: jse_analysis_output/ ===")
        print("   ├── 1_forward_filled_data/         (Unscaled CSVs per stock)")
        print("   ├── 2_scaled_data/                 (Scaled CSVs per stock)")
        print("   ├── 3_scaling_comparison_plots/    (Before/After Scaling Plots)")
        print("   ├── 4_decomposition_before_scaling/(Decomposition of Original Data)")
        print("   ├── 5_decomposition_after_scaling/ (Decomposition of Scaled Data)")
        print("   └── 6_combined_datasets/           (CSVs combining all stocks)")
        print("       ├── combined_train_filled.csv")
        print("       ├── combined_validation_filled.csv")
        print("       ├── ... (and scaled versions)")
        
        print("\nData is ready for machine learning!")
    else:
        print("Failed to load and split data. Exiting.")