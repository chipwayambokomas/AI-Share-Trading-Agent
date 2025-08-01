import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Data Loading and Splitting (Existing Functions, with minor fixes) ---

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
            
            # Forward fill is now applied later, after splitting
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
        # FIX: Use modern .ffill() and .bfill() to avoid FutureWarning
        df_filled[existing_numerical_cols] = df_filled[existing_numerical_cols].ffill()
        df_filled[existing_numerical_cols] = df_filled[existing_numerical_cols].bfill()
    
    return df_filled

def save_splits_to_csv(split_data, output_dir="split_data"):
    """
    Save the train/val/test splits to separate CSV files with forward fill applied.
    """
    os.makedirs(output_dir, exist_ok=True)
    filled_data = {}
    
    for sheet_name, data in split_data.items():
        print(f"\n--- Applying forward fill and saving for {sheet_name} ---")
        filled_data[sheet_name] = {}
        clean_name = sheet_name.replace(' ', '_').replace('/', '_').replace('.', '_')
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in data and not data[split_name].empty:
                df_original = data[split_name]
                df_filled = apply_forward_fill(df_original)
                
                # Drop any rows that are still all NaN (can happen if a whole sheet is empty)
                df_filled.dropna(how='all', subset=['open', 'high', 'low', 'close', 'vwap'], inplace=True)
                
                if df_filled.empty:
                    print(f"⚠️ Skipping empty split {split_name} for {sheet_name}")
                    continue

                filled_data[sheet_name][split_name] = df_filled
                
                csv_path = os.path.join(output_dir, f"{clean_name}_{split_name}.csv")
                df_filled.to_csv(csv_path, index=False)
                print(f"✅ Saved {split_name} data: {csv_path}")
    
    return filled_data

# --- NEW: Scaling and Plotting Functions ---

def scale_data(filled_split_data, columns_to_scale=['open', 'high', 'low', 'close', 'vwap']):
    """
    Applies Min-Max scaling based on the specified logic.

    - Fits a scaler on the training data.
    - Transforms the training and validation data with this scaler.
    - Fits a NEW scaler on the validation data.
    - Transforms the test data with the second scaler.

    Returns:
    - A dictionary with the scaled dataframes.
    - A dictionary containing the scalers used for each symbol.
    """
    print("\n=== Scaling Data ===")
    scaled_data = {}
    scalers = {}

    for symbol, splits in filled_split_data.items():
        print(f"--- Scaling {symbol} ---")
        
        train_df = splits.get('train')
        val_df = splits.get('validation')
        test_df = splits.get('test')

        if train_df is None or val_df is None or test_df is None:
            print(f"⚠️ Skipping {symbol} due to missing data splits.")
            continue

        scaled_data[symbol] = {}
        scalers[symbol] = {}
        
        # --- Scaler 1: Fit on Train, Transform Train & Validation ---
        scaler_train = MinMaxScaler()
        
        # Ensure columns exist before trying to scale
        valid_cols = [col for col in columns_to_scale if col in train_df.columns]
        if not valid_cols:
            print(f"⚠️ No columns to scale for {symbol}. Skipping scaling.")
            continue

        scaler_train.fit(train_df[valid_cols])
        
        # Transform Train
        train_scaled = train_df.copy()
        train_scaled[valid_cols] = scaler_train.transform(train_df[valid_cols])
        scaled_data[symbol]['train'] = train_scaled
        
        # Transform Validation
        val_scaled = val_df.copy()
        val_scaled[valid_cols] = scaler_train.transform(val_df[valid_cols])
        scaled_data[symbol]['validation'] = val_scaled
        
        scalers[symbol]['scaler_train_to_val'] = scaler_train
        print(f"  Applied 'train' scaler to 'train' and 'validation' sets.")

        # --- Scaler 2: Fit on Validation, Transform Test ---
        # NOTE: This is a non-standard ML practice. Typically, the scaler fit on the training
        # data would be used for the test set as well to prevent data leakage and ensure
        # consistent transformation. This implementation follows the user's specific request.
        scaler_val = MinMaxScaler()
        scaler_val.fit(val_df[valid_cols]) # Fit on original validation data
        
        # Transform Test
        test_scaled = test_df.copy()
        test_scaled[valid_cols] = scaler_val.transform(test_df[valid_cols])
        scaled_data[symbol]['test'] = test_scaled
        
        scalers[symbol]['scaler_val_to_test'] = scaler_val
        print(f"  Applied 'validation' scaler to 'test' set.")

    return scaled_data, scalers

def plot_scaling_comparison(df_before, df_after, symbol, split_name, output_dir):
    """Creates a plot comparing data before and after scaling."""
    if 'close' not in df_before.columns or 'Date' not in df_before.columns:
        return

    plots_dir = os.path.join(output_dir, "scaling_comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Scaling Comparison for {symbol} - {split_name.capitalize()} Set', fontsize=16, fontweight='bold')
    
    # Before scaling
    ax1.plot(df_before['Date'], df_before['close'], label='Original Close Price', color='blue')
    ax1.set_title('Before Scaling')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.5)
    ax1.legend()
    
    # After scaling
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

def save_scaled_data(scaled_data, original_data, output_dir="scaled_data"):
    """Saves scaled data to CSVs and generates comparison plots."""
    print("\n=== Saving Scaled Data and Generating Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, splits in scaled_data.items():
        for split_name, df_scaled in splits.items():
            clean_symbol = symbol.replace(' ', '_').replace('/', '_').replace('.', '_')
            
            # Save scaled data to CSV
            csv_path = os.path.join(output_dir, f"{clean_symbol}_{split_name}_scaled.csv")
            df_scaled.to_csv(csv_path, index=False)
            print(f"✅ Saved scaled {split_name} data: {csv_path}")
            
            # Generate comparison plot
            df_original = original_data[symbol][split_name]
            plot_scaling_comparison(df_original, df_scaled, symbol, split_name, output_dir)
    
    print(f"📊 Scaling comparison plots saved in: {os.path.join(output_dir, 'scaling_comparison_plots')}")

# --- NEW: Convenience Functions to Get Combined DataFrames ---

def get_combined_dataframe(split_data, split_type='train'):
    """
    Combine all sheets into a single DataFrame for a specific split.
    """
    combined_dfs = []
    for sheet_name, data in split_data.items():
        if split_type in data and not data[split_type].empty:
            df = data[split_type].copy()
            df['symbol'] = sheet_name
            combined_dfs.append(df)
    
    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        if 'Date' in combined_df.columns:
            combined_df = combined_df.sort_values(['Date', 'symbol']).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()

def get_train_df(split_data):
    """Returns a combined DataFrame of all training sets."""
    return get_combined_dataframe(split_data, 'train')

def get_val_df(split_data):
    """Returns a combined DataFrame of all validation sets."""
    return get_combined_dataframe(split_data, 'validation')

def get_test_df(split_data):
    """Returns a combined DataFrame of all test sets."""
    return get_combined_dataframe(split_data, 'test')


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Define file paths
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_excel_path = os.path.join(script_directory, cleaned_excel_file)
    
    # 2. Load and split the data chronologically
    print("=== 1. Loading and Splitting JSE Data ===")
    raw_split_data = load_and_split_jse_data(full_excel_path)
    
    if raw_split_data:
        # 3. Apply forward fill and save the cleaned, unscaled splits
        print("\n=== 2. Applying Forward Fill and Saving Base Splits ===")
        filled_split_data = save_splits_to_csv(raw_split_data, output_dir="split_data")
        
        # 4. Scale the data using the specified logic
        scaled_data, scalers = scale_data(filled_split_data)
        
        # 5. Save the scaled data and create before/after comparison plots
        save_scaled_data(scaled_data, filled_split_data, output_dir="scaled_data")
        
        # 6. Example usage of new convenience functions
        print("\n=== 3. Example Usage of Combined DataFrames ===")
        
        # Get combined dataframes (after forward-fill but before scaling)
        combined_train_df = get_train_df(filled_split_data)
        combined_val_df = get_val_df(filled_split_data)
        combined_test_df = get_test_df(filled_split_data)
        
        print(f"Combined Training DataFrame shape: {combined_train_df.shape}")
        print(f"Combined Validation DataFrame shape: {combined_val_df.shape}")
        print(f"Combined Test DataFrame shape: {combined_test_df.shape}")
        if not combined_train_df.empty:
            print(f"Unique symbols in combined train data: {combined_train_df['symbol'].nunique()}")

        # 7. Show what was created
        print("\n=== All Tasks Complete! Files Created: ===")
        print("📁 split_data/ (Forward-filled, unscaled data)")
        print("   └── 📄 [symbol]_[split].csv")
        print("📁 scaled_data/ (Forward-filled AND scaled data)")
        print("   ├── 📊 scaling_comparison_plots/")
        print("   │   └── 🖼️ [symbol]_[split]_scaling_comparison.png")
        print("   └── 📄 [symbol]_[split]_scaled.csv")
        
        print("\nData is ready for machine learning!")
    else:
        print("Failed to load and split data. Exiting.")