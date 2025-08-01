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

def plot_forward_fill_comparison(df_before, df_after, title, output_dir, filename_prefix):
    """
    Create comparison plots showing before and after forward fill application.
    
    Parameters:
    - df_before: DataFrame before forward fill
    - df_after: DataFrame after forward fill
    - title: Title for the plots
    - output_dir: Directory to save plots
    - filename_prefix: Prefix for plot filenames
    """
    
    if df_before.empty or df_after.empty or 'Date' not in df_before.columns:
        print(f"⚠️ Cannot create forward fill comparison for {title}: insufficient data")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots", "forward_fill_comparison")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Identify missing data in original
    numerical_cols = ['open', 'high', 'low', 'close', 'vwap']
    existing_cols = [col for col in numerical_cols if col in df_before.columns]
    
    # Count missing values
    missing_counts_before = df_before[existing_cols].isnull().sum()
    missing_counts_after = df_after[existing_cols].isnull().sum()
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{title} - Forward Fill Impact Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Missing values comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = range(len(existing_cols))
    width = 0.35
    
    ax1.bar([x - width/2 for x in x_pos], missing_counts_before[existing_cols], 
            width, label='Before Forward Fill', color='red', alpha=0.7)
    ax1.bar([x + width/2 for x in x_pos], missing_counts_after[existing_cols], 
            width, label='After Forward Fill', color='green', alpha=0.7)
    
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Number of Missing Values')
    ax1.set_title('Missing Values: Before vs After')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(existing_cols, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data availability heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create heatmap showing missing data patterns
    missing_data_before = df_before[existing_cols].isnull().astype(int)
    if len(missing_data_before) > 0:
        # Sample data for visualization if too many rows
        sample_size = min(100, len(missing_data_before))
        if len(missing_data_before) > sample_size:
            step = len(missing_data_before) // sample_size
            missing_sample = missing_data_before.iloc[::step]
        else:
            missing_sample = missing_data_before
        
        im = ax2.imshow(missing_sample.T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        ax2.set_title('Missing Data Pattern (Red = Missing)')
        ax2.set_xlabel('Time Index (sampled)')
        ax2.set_ylabel('Columns')
        ax2.set_yticks(range(len(existing_cols)))
        ax2.set_yticklabels(existing_cols)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Missing (1) vs Present (0)')
    
    # Plot 3: Forward fill statistics
    ax3 = fig.add_subplot(gs[0, 2])
    
    fill_stats = []
    for col in existing_cols:
        if col in df_before.columns and col in df_after.columns:
            original_missing = df_before[col].isnull().sum()
            filled_missing = df_after[col].isnull().sum()
            filled_count = original_missing - filled_missing
            fill_stats.append({
                'Column': col,
                'Original_Missing': original_missing,
                'Still_Missing': filled_missing,
                'Values_Filled': filled_count
            })
    
    if fill_stats:
        stats_df = pd.DataFrame(fill_stats)
        ax3.bar(stats_df['Column'], stats_df['Values_Filled'], color='blue', alpha=0.7)
        ax3.set_title('Values Filled by Forward Fill')
        ax3.set_xlabel('Columns')
        ax3.set_ylabel('Number of Values Filled')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Plots 4-6: Time series comparisons for each main price column
    price_cols = ['open', 'high', 'low', 'close']
    plot_positions = [(1, 0), (1, 1), (1, 2), (2, 0)]
    
    for i, col in enumerate(price_cols):
        if col in existing_cols and i < len(plot_positions):
            row, col_pos = plot_positions[i]
            ax = fig.add_subplot(gs[row, col_pos])
            
            # Plot original data
            ax.plot(df_before['Date'], df_before[col], 'o-', markersize=2, 
                   linewidth=1, alpha=0.7, label=f'Before Fill', color='red')
            
            # Plot filled data
            ax.plot(df_after['Date'], df_after[col], 'o-', markersize=2, 
                   linewidth=1, alpha=0.8, label=f'After Fill', color='blue')
            
            # Highlight filled values
            filled_mask = df_before[col].isnull() & df_after[col].notnull()
            if filled_mask.any():
                filled_dates = df_after.loc[filled_mask, 'Date']
                filled_values = df_after.loc[filled_mask, col]
                ax.scatter(filled_dates, filled_values, color='green', s=30, 
                          label='Filled Values', alpha=0.8, edgecolors='black')
            
            ax.set_title(f'{col.capitalize()} Price - Forward Fill Impact')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 7: VWAP comparison
    if 'vwap' in existing_cols:
        ax7 = fig.add_subplot(gs[2, 1])
        
        ax7.plot(df_before['Date'], df_before['vwap'], 'o-', markersize=2, 
               linewidth=1, alpha=0.7, label='Before Fill', color='red')
        ax7.plot(df_after['Date'], df_after['vwap'], 'o-', markersize=2, 
               linewidth=1, alpha=0.8, label='After Fill', color='blue')
        
        # Highlight filled values
        filled_mask = df_before['vwap'].isnull() & df_after['vwap'].notnull()
        if filled_mask.any():
            filled_dates = df_after.loc[filled_mask, 'Date']
            filled_values = df_after.loc[filled_mask, 'vwap']
            ax7.scatter(filled_dates, filled_values, color='green', s=30, 
                      label='Filled Values', alpha=0.8, edgecolors='black')
        
        ax7.set_title('VWAP - Forward Fill Impact')
        ax7.set_ylabel('VWAP')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.tick_params(axis='x', rotation=45)
    
    # Plot 8: Summary statistics comparison
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Calculate completion rates
    completion_before = []
    completion_after = []
    
    for col in existing_cols:
        total_rows = len(df_before)
        complete_before = total_rows - df_before[col].isnull().sum()
        complete_after = total_rows - df_after[col].isnull().sum()
        
        completion_before.append(complete_before / total_rows * 100)
        completion_after.append(complete_after / total_rows * 100)
    
    x_pos = range(len(existing_cols))
    width = 0.35
    
    ax8.bar([x - width/2 for x in x_pos], completion_before, width, 
           label='Before Fill', color='orange', alpha=0.7)
    ax8.bar([x + width/2 for x in x_pos], completion_after, width, 
           label='After Fill', color='green', alpha=0.7)
    
    ax8.set_xlabel('Columns')
    ax8.set_ylabel('Data Completion Rate (%)')
    ax8.set_title('Data Completion: Before vs After')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(existing_cols, rotation=45)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 105)
    
    # Plot 9: Data quality metrics
    ax9 = fig.add_subplot(gs[3, :])
    
    # Create detailed statistics table as text
    stats_text = "Forward Fill Impact Summary:\n\n"
    
    for col in existing_cols:
        original_missing = df_before[col].isnull().sum()
        after_missing = df_after[col].isnull().sum()
        filled_count = original_missing - after_missing
        total_rows = len(df_before)
        
        stats_text += f"{col.upper()}: "
        stats_text += f"Missing Before: {original_missing} ({original_missing/total_rows*100:.1f}%) | "
        stats_text += f"Missing After: {after_missing} ({after_missing/total_rows*100:.1f}%) | "
        stats_text += f"Values Filled: {filled_count}\n"
    
    total_missing_before = df_before[existing_cols].isnull().sum().sum()
    total_missing_after = df_after[existing_cols].isnull().sum().sum()
    total_filled = total_missing_before - total_missing_after
    
    stats_text += f"\nOVERALL: {total_filled} values filled out of {total_missing_before} missing values "
    stats_text += f"({total_filled/total_missing_before*100:.1f}% filled)" if total_missing_before > 0 else "(No missing values)"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    # Save the plot
    plot_filename = f"{filename_prefix}_forward_fill_comparison.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔍 Forward fill comparison plot saved: {plot_path}")
    
    # Return summary statistics
    return {
        'total_missing_before': total_missing_before,
        'total_missing_after': total_missing_after,
        'total_filled': total_filled,
        'completion_rate_improvement': (total_filled/total_missing_before*100) if total_missing_before > 0 else 0
    }
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
    Also create detailed forward fill comparison plots.
    
    Parameters:
    - split_data: Dictionary returned from load_and_split_jse_data()
    - output_dir: Directory to save CSV files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track forward fill statistics across all symbols and splits
    all_fill_stats = []
    
    for sheet_name, data in split_data.items():
        print(f"\n--- Processing {sheet_name} for CSV export and forward fill analysis ---")
        
        # Clean sheet name for filename
        clean_name = sheet_name.replace(' ', '_').replace('/', '_').replace('.', '_')
        
        # Process each split
        for split_name in ['train', 'validation', 'test']:
            if split_name in data:
                df_original = data[split_name].copy()
                
                # Apply forward fill
                df_filled = apply_forward_fill(df_original)
                
                # Create forward fill comparison plot
                plot_title = f"{sheet_name} - {split_name.capitalize()} Set"
                plot_prefix = f"{clean_name}_{split_name}"
                
                fill_stats = plot_forward_fill_comparison(
                    df_original, df_filled, plot_title, output_dir, plot_prefix
                )
                
                if fill_stats:
                    fill_stats.update({
                        'symbol': sheet_name,
                        'split': split_name,
                        'total_rows': len(df_original)
                    })
                    all_fill_stats.append(fill_stats)
                
                # Save to CSV
                csv_path = os.path.join(output_dir, f"{clean_name}_{split_name}.csv")
                df_filled.to_csv(csv_path, index=False)
                
                # Create regular OHLCV plot with filled data
                plot_ohlcv_data(df_filled, plot_title, output_dir, plot_prefix)
                
                print(f"✅ Saved {split_name} data: {csv_path}")
        
        print(f"✅ Completed processing for '{sheet_name}'")
    
    # Create summary of all forward fill operations
    if all_fill_stats:
        create_forward_fill_summary(all_fill_stats, output_dir)
    
    return all_fill_stats

def create_forward_fill_summary(fill_stats, output_dir):
    """
    Create a comprehensive summary of forward fill operations across all splits.
    
    Parameters:
    - fill_stats: List of dictionaries with forward fill statistics
    - output_dir: Directory to save summary
    """
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    stats_df = pd.DataFrame(fill_stats)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Forward Fill Operations Summary - All Symbols & Splits', fontsize=16, fontweight='bold')
    
    # Plot 1: Total values filled by split
    ax1 = axes[0, 0]
    split_totals = stats_df.groupby('split')['total_filled'].sum()
    colors = ['blue', 'orange', 'green']
    bars = ax1.bar(split_totals.index, split_totals.values, color=colors, alpha=0.7)
    ax1.set_title('Total Values Filled by Split')
    ax1.set_ylabel('Number of Values Filled')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Average completion rate improvement by split
    ax2 = axes[0, 1]
    completion_avg = stats_df.groupby('split')['completion_rate_improvement'].mean()
    bars = ax2.bar(completion_avg.index, completion_avg.values, color=colors, alpha=0.7)
    ax2.set_title('Average Completion Rate Improvement')
    ax2.set_ylabel('Improvement (%)')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Distribution of missing values before fill
    ax3 = axes[0, 2]
    ax3.hist(stats_df['total_missing_before'], bins=20, alpha=0.7, color='red', edgecolor='black')
    ax3.set_title('Distribution of Missing Values\n(Before Forward Fill)')
    ax3.set_xlabel('Missing Values per Split')
    ax3.set_ylabel('Frequency')
    
    # Plot 4: Forward fill effectiveness by symbol
    ax4 = axes[1, 0]
    symbol_effectiveness = stats_df.groupby('symbol')['completion_rate_improvement'].mean().sort_values(ascending=True)
    
    # Show top 10 symbols with most improvement
    top_symbols = symbol_effectiveness.tail(10)
    y_pos = np.arange(len(top_symbols))
    ax4.barh(y_pos, top_symbols.values, alpha=0.7, color='green')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_symbols.index, fontsize=8)
    ax4.set_title('Top 10 Symbols: Forward Fill Effectiveness')
    ax4.set_xlabel('Average Completion Rate Improvement (%)')
    
    # Plot 5: Missing values before vs after
    ax5 = axes[1, 1]
    total_before = stats_df['total_missing_before'].sum()
    total_after = stats_df['total_missing_after'].sum()
    
    categories = ['Before Forward Fill', 'After Forward Fill']
    values = [total_before, total_after]
    colors_ba = ['red', 'green']
    
    bars = ax5.bar(categories, values, color=colors_ba, alpha=0.7)
    ax5.set_title('Total Missing Values: Before vs After')
    ax5.set_ylabel('Number of Missing Values')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Summary statistics table
    ax6 = axes[1, 2]
    
    # Calculate overall statistics
    total_splits = len(stats_df)
    total_symbols = stats_df['symbol'].nunique()
    total_rows_processed = stats_df['total_rows'].sum()
    total_values_filled = stats_df['total_filled'].sum()
    avg_completion_improvement = stats_df['completion_rate_improvement'].mean()
    
    # Create summary text
    summary_text = f"""FORWARD FILL SUMMARY STATISTICS
    
Total Symbols Processed: {total_symbols}
Total Splits Processed: {total_splits}
Total Rows Processed: {total_rows_processed:,}

Missing Values Before: {total_before:,}
Missing Values After: {total_after:,}
Total Values Filled: {total_values_filled:,}

Overall Completion Rate Improvement: {avg_completion_improvement:.1f}%
Fill Success Rate: {(total_values_filled/total_before*100):.1f}%

Split Breakdown:
• Training Sets: {len(stats_df[stats_df['split']=='train'])} processed
• Validation Sets: {len(stats_df[stats_df['split']=='validation'])} processed  
• Test Sets: {len(stats_df[stats_df['split']=='test'])} processed
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = os.path.join(plots_dir, "forward_fill_summary.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics to CSV
    summary_csv_path = os.path.join(output_dir, "forward_fill_statistics.csv")
    stats_df.to_csv(summary_csv_path, index=False)
    
    print(f"📊 Forward fill summary plot saved: {summary_plot_path}")
    print(f"📊 Forward fill statistics saved: {summary_csv_path}")
    print(f"🔍 Individual comparison plots saved in: {os.path.join(output_dir, 'plots', 'forward_fill_comparison')}")


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
        print("   ├── 📊 plots/")
        print("   │   ├── forward_fill_comparison/ (before/after comparison plots)")
        print("   │   │   └── [symbol]_[split]_forward_fill_comparison.png")
        print("   │   ├── [symbol]_[split]_analysis.png (OHLCV analysis)")
        print("   │   ├── forward_fill_summary.png (overall forward fill analysis)")
        print("   │   └── data_split_summary.png (split distribution)")
        print("   ├── 📈 forward_fill_statistics.csv (detailed fill stats)")
        print("   ├── 📋 split_summary_statistics.csv (split summary)")
        print("   └── 📄 [symbol]_[split].csv files (forward-filled data)")
        
        print("\n=== Forward Fill Analysis Complete! ===")
        print("🔍 Check the forward_fill_comparison folder for detailed before/after plots")
        print("📊 View forward_fill_summary.png for overall statistics")
        print("📈 Review forward_fill_statistics.csv for detailed metrics")
        
        print("\n=== Data is ready for machine learning! ===")
        print("Access your data using:")
        print("- split_data['SYMBOL_NAME']['train'] for training data")
        print("- split_data['SYMBOL_NAME']['validation'] for validation data") 
        print("- split_data['SYMBOL_NAME']['test'] for test data")
        print("\nAll splits have been saved with forward fill applied and analyzed!")
    else:
        print("Failed to load and split data.")