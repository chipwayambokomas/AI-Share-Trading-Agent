#!/usr/bin/env python3
"""
Preprocess ABG stock data by sorting chronologically
"""
import pandas as pd
import os

def preprocess_abg_data():
    """Sort ABG data chronologically (oldest to newest)"""
    
    # File paths
    input_file = 'data/ABG.csv'
    output_file = 'data/ABG_sorted.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print(f"Loading data from: {input_file}")
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Show original date range
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # Handle DD/MM/YYYY format
    print(f"Original date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Original order: {'REVERSE' if df['Date'].iloc[0] > df['Date'].iloc[-1] else 'CHRONOLOGICAL'}")
    
    # Sort chronologically (oldest first)
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    # Show new date range
    print(f"Sorted date range: {df_sorted['Date'].min()} to {df_sorted['Date'].max()}")
    print(f"New order: CHRONOLOGICAL (oldest → newest)")
    
    # Save the sorted data
    df_sorted.to_csv(output_file, index=False)
    print(f"Sorted data saved to: {output_file}")
    
    # Show first and last few rows
    print("\n📅 First 5 rows (oldest):")
    print(df_sorted[['Date', 'Value']].head())
    
    print("\n📅 Last 5 rows (newest):")
    print(df_sorted[['Date', 'Value']].tail())
    
    return output_file

if __name__ == "__main__":
    print("=" * 50)
    print("ABG Data Preprocessing")
    print("=" * 50)
    
    output_file = preprocess_abg_data()
    
    if output_file:
        print("\n✅ Preprocessing completed successfully!")
        print(f"✅ Use this file for training: {output_file}")
        print("\n📋 Next steps:")
        print("1. Update configs/tcnconfig.yaml:")
        print('   file_path: "data/ABG_sorted.csv"')
        print("2. Run training:")
        print("   python3 TCNmain.py --mode train")
    else:
        print("\n❌ Preprocessing failed!")