import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_split_jse_data(excel_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Load cleaned JSE data from Excel file and split chronologically into train/val/test sets.
    
    Parameters:
    - excel_path: Path to the cleaned Excel file
    - train_ratio: Proportion for training set (default 0.6)
    - val_ratio: Proportion for validation set (default 0.2)
    - test_ratio: Proportion for test set (default 0.2)
    
    Returns:
    - Dictionary containing train, validation, and test DataFrames for each sheet
    """
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    try:
        # Load the Excel file
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        print(f"Loading data from {len(sheet_names)} sheets...")
        
        # Dictionary to store split data for each sheet
        split_data = {}
        
        for sheet_name in sheet_names:
            print(f"\n--- Processing Sheet: {sheet_name} ---")
            
            # Read the sheet into a DataFrame
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Convert Date column to datetime if it isn't already
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                # Sort by date to ensure chronological order
                df = df.sort_values('Date').reset_index(drop=True)
            else:
                print(f"⚠️ Warning: No 'Date' column found in sheet '{sheet_name}'. Using row order.")
            
            # Remove any rows with missing data
            df_clean = df.dropna()
            total_rows = len(df_clean)
            
            if total_rows == 0:
                print(f"⚠️ Warning: No valid data in sheet '{sheet_name}'. Skipping.")
                continue
            
            # Calculate split indices
            train_end = int(total_rows * train_ratio)
            val_end = int(total_rows * (train_ratio + val_ratio))
            
            # Split the data chronologically
            train_df = df_clean.iloc[:train_end].copy()
            val_df = df_clean.iloc[train_end:val_end].copy()
            test_df = df_clean.iloc[val_end:].copy()
            
            # Store the splits
            split_data[sheet_name] = {
                'train': train_df,
                'validation': val_df,
                'test': test_df,
                'full': df_clean
            }
            
            # Print split information
            print(f"Total rows: {total_rows}")
            print(f"Training set: {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)")
            print(f"Validation set: {len(val_df)} rows ({len(val_df)/total_rows*100:.1f}%)")
            print(f"Test set: {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)")
            
            if 'Date' in df.columns:
                print(f"Date ranges:")
                print(f"  Training: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
                print(f"  Validation: {val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
                print(f"  Test: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
        
        return split_data
        
    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"The file was not found at: {excel_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def apply_forward_fill(df):
    """
    Apply forward fill to handle missing values in OHLCV data.
    
    Parameters:
    - df: DataFrame with OHLCV data
    
    Returns:
    - DataFrame with forward fill applied
    """
    df_filled = df.copy()
    
    # Apply forward fill to numerical columns (preserve Date column as is)
    numerical_cols = ['open', 'high', 'low', 'close', 'vwap']
    existing_numerical_cols = [col for col in numerical_cols if col in df_filled.columns]
    
    if existing_numerical_cols:
        df_filled[existing_numerical_cols] = df_filled[existing_numerical_cols].fillna(method='ffill')
        
        # If there are still NaN values at the beginning, use backward fill
        df_filled[existing_numerical_cols] = df_filled[existing_numerical_cols].fillna(method='bfill')
    
    return df_filled

def plot_ohlcv_data(df, title, output_dir, filename_prefix):
    """
    Create comprehensive plots for OHLCV data.
    
    Parameters:
    - df: DataFrame with OHLCV data
    - title: Title for the plots
    - output_dir: Directory to save plots
    - filename_prefix: Prefix for plot filenames
    """
    
    if df.empty or 'Date' not in df.columns:
        print(f"⚠️ Cannot plot {title}: insufficient data")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - OHLCV Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Price movements (OHLC)
    ax1 = axes[0, 0]
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        ax1.plot(df['Date'], df['open'], label='Open', alpha=0.7, linewidth=1)
        ax1.plot(df['Date'], df['high'], label='High', alpha=0.7, linewidth=1)
        ax1.plot(df['Date'], df['low'], label='Low', alpha=0.7, linewidth=1)
        ax1.plot(df['Date'], df['close'], label='Close', alpha=0.9, linewidth=2)
        ax1.set_title('OHLC Prices Over Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Volume
    ax2 = axes[0, 1]
    if 'vwap' in df.columns:
        ax2.plot(df['Date'], df['vwap'], label='VWAP', color='purple', linewidth=2)
        ax2.set_title('Volume Weighted Average Price (VWAP)')
        ax2.set_ylabel('VWAP')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Price volatility (High-Low spread)
    ax3 = axes[1, 0]
    if all(col in df.columns for col in ['high', 'low']):
        volatility = df['high'] - df['low']
        ax3.plot(df['Date'], volatility, color='red', alpha=0.7, linewidth=1)
        ax3.fill_between(df['Date'], volatility, alpha=0.3, color='red')
        ax3.set_title('Daily Price Volatility (High - Low)')
        ax3.set_ylabel('Price Spread')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Close price with moving averages
    ax4 = axes[1, 1]
    if 'close' in df.columns:
        ax4.plot(df['Date'], df['close'], label='Close Price', linewidth=2)
        
        # Add moving averages if we have enough data
        if len(df) >= 20:
            ma_20 = df['close'].rolling(window=20).mean()
            ax4.plot(df['Date'], ma_20, label='20-day MA', alpha=0.8, linestyle='--')
        
        if len(df) >= 50:
            ma_50 = df['close'].rolling(window=50).mean()
            ax4.plot(df['Date'], ma_50, label='50-day MA', alpha=0.8, linestyle='--')
        
        ax4.set_title('Close Price with Moving Averages')
        ax4.set_ylabel('Price')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{filename_prefix}_analysis.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {plot_path}")

def save_splits_to_csv(split_data, output_dir="split_data"):
    """
    Save the train/val/test splits to separate CSV files with forward fill applied.
    
    Parameters:
    - split_data: Dictionary returned from load_and_split_jse_data()
    - output_dir: Directory to save CSV files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for sheet_name, data in split_data.items():
        print(f"\n--- Processing {sheet_name} for CSV export and plotting ---")
        
        # Clean sheet name for filename
        clean_name = sheet_name.replace(' ', '_').replace('/', '_').replace('.', '_')
        
        # Process each split
        for split_name in ['train', 'validation', 'test']:
            if split_name in data:
                # Apply forward fill
                df_filled = apply_forward_fill(data[split_name])
                
                # Save to CSV
                csv_path = os.path.join(output_dir, f"{clean_name}_{split_name}.csv")
                df_filled.to_csv(csv_path, index=False)
                
                # Create plot
                plot_title = f"{sheet_name} - {split_name.capitalize()} Set"
                plot_prefix = f"{clean_name}_{split_name}"
                plot_ohlcv_data(df_filled, plot_title, output_dir, plot_prefix)
                
                print(f"✅ Saved {split_name} data: {csv_path}")
        
        print(f"✅ Completed processing for '{sheet_name}'")

def create_summary_plots(split_data, output_dir="split_data"):
    """
    Create summary plots showing data distribution across splits.
    
    Parameters:
    - split_data: Dictionary returned from load_and_split_jse_data()
    - output_dir: Directory to save plots
    """
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create summary statistics
    summary_stats = []
    
    for sheet_name, data in split_data.items():
        for split_name in ['train', 'validation', 'test']:
            if split_name in data and not data[split_name].empty:
                df = data[split_name]
                if 'Date' in df.columns:
                    summary_stats.append({
                        'Symbol': sheet_name,
                        'Split': split_name,
                        'Start_Date': df['Date'].min(),
                        'End_Date': df['Date'].max(),
                        'Rows': len(df),
                        'Days': (df['Date'].max() - df['Date'].min()).days
                    })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        # Plot 1: Data distribution by split
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Split Summary Analysis', fontsize=16, fontweight='bold')
        
        # Rows per split
        ax1 = axes[0, 0]
        split_counts = summary_df.groupby('Split')['Rows'].sum()
        ax1.bar(split_counts.index, split_counts.values, color=['blue', 'orange', 'green'])
        ax1.set_title('Total Rows per Split')
        ax1.set_ylabel('Number of Rows')
        
        # Timeline visualization
        ax2 = axes[0, 1]
        colors = {'train': 'blue', 'validation': 'orange', 'test': 'green'}
        for _, row in summary_df.iterrows():
            ax2.barh(row['Symbol'], (row['End_Date'] - row['Start_Date']).days, 
                    left=row['Start_Date'], color=colors[row['Split']], alpha=0.7)
        ax2.set_title('Timeline Coverage by Symbol and Split')
        ax2.set_xlabel('Date')
        
        # Days per split
        ax3 = axes[1, 0]
        split_days = summary_df.groupby('Split')['Days'].sum()
        ax3.bar(split_days.index, split_days.values, color=['blue', 'orange', 'green'])
        ax3.set_title('Total Days per Split')
        ax3.set_ylabel('Number of Days')
        
        # Symbols per split
        ax4 = axes[1, 1]
        split_symbols = summary_df.groupby('Split')['Symbol'].nunique()
        ax4.bar(split_symbols.index, split_symbols.values, color=['blue', 'orange', 'green'])
        ax4.set_title('Number of Symbols per Split')
        ax4.set_ylabel('Number of Symbols')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_plot_path = os.path.join(plots_dir, "data_split_summary.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        summary_csv_path = os.path.join(output_dir, "split_summary_statistics.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"📊 Summary plot saved: {summary_plot_path}")
        print(f"📊 Summary statistics saved: {summary_csv_path}")
    else:
        print("⚠️ No data available for summary plots")

def get_combined_dataframe(split_data, split_type='train'):
    """
    Combine all sheets into a single DataFrame for a specific split.
    
    Parameters:
    - split_data: Dictionary returned from load_and_split_jse_data()
    - split_type: 'train', 'validation', 'test', or 'full'
    
    Returns:
    - Combined DataFrame with an additional 'symbol' column
    """
    
    combined_dfs = []
    
    for sheet_name, data in split_data.items():
        if split_type in data:
            df = data[split_type].copy()
            df['symbol'] = sheet_name  # Add symbol column
            combined_dfs.append(df)
    
    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        if 'Date' in combined_df.columns:
            combined_df = combined_df.sort_values(['Date', 'symbol']).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Define the cleaned Excel file path
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    
    # Get the full path
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_excel_path = os.path.join(script_directory, cleaned_excel_file)
    
    # Load and split the data
    print("=== Loading and Splitting JSE Data ===")
    split_data = load_and_split_jse_data(full_excel_path)
    
    if split_data:
        print(f"\n=== Data Successfully Split for {len(split_data)} Symbols ===")
        
        # Automatically save splits to CSV files with forward fill and plots
        print("\n=== Applying Forward Fill and Saving Splits ===")
        save_splits_to_csv(split_data)
        
        # Create summary plots
        print("\n=== Creating Summary Analysis ===")
        create_summary_plots(split_data)
        
        # Example: Access specific data
        print("\n=== Example Usage ===")
        first_symbol = list(split_data.keys())[0]
        print(f"First symbol: {first_symbol}")
        print(f"Training data shape: {split_data[first_symbol]['train'].shape}")
        print(f"Training data columns: {split_data[first_symbol]['train'].columns.tolist()}")
        
        # Example: Get combined training data across all symbols
        combined_train = get_combined_dataframe(split_data, 'train')
        print(f"\nCombined training data shape: {combined_train.shape}")
        print(f"Unique symbols in combined data: {combined_train['symbol'].nunique()}")
        
        # Show what was created
        print(f"\n=== Files Created ===")
        print("📁 split_data/")
        print("   ├── 📊 plots/ (individual symbol plots for each split)")
        print("   ├── 📈 data_split_summary.png (overall analysis)")
        print("   ├── 📋 split_summary_statistics.csv (summary stats)")
        print("   └── 📄 [symbol]_[split].csv files (forward-filled data)")
        
        print("\n=== Data is ready for machine learning! ===")
        print("Access your data using:")
        print("- split_data['SYMBOL_NAME']['train'] for training data")
        print("- split_data['SYMBOL_NAME']['validation'] for validation data") 
        print("- split_data['SYMBOL_NAME']['test'] for test data")
        print("\nAll splits have been saved with forward fill applied!")
    else:
        print("Failed to load and split data.")