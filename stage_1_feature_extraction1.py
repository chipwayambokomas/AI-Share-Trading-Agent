# src/pipeline/stage_1_feature_extraction.py

import pandas as pd
from src.utils import print_header

def run(config):
    print_header("Stage 1: Feature Extraction")
    print("""
Purpose: Read data from all sheets in the Excel file, stack them into a single
DataFrame, and add a 'StockID' to identify each security.
Executing Code: Loading and concatenating data from all sheets...
    """)

    try:
        all_sheets = pd.read_excel(config.FILE_PATH, sheet_name=None)
        print(f"Found {len(all_sheets)} sheets (stocks) in the file: {list(all_sheets.keys())}")
    except FileNotFoundError:
        print(f"Error: The file '{config.FILE_PATH}' was not found.")
        raise

    processed_dfs = []
    for sheet_name, df in all_sheets.items():
        if df.empty:
            print(f"    - WARNING: Sheet '{sheet_name}' is empty. Skipping.")
            continue

        df = df.iloc[:, :2]
        df.columns = ['Date', config.TARGET_COLUMN]
        
        # --- FIX: Be explicit about the date format to remove the warning ---
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        # --- END OF FIX ---
        
        df.dropna(subset=['Date', config.TARGET_COLUMN], inplace=True)

        if df.empty:
            print(f"    - WARNING: Sheet '{sheet_name}' has no valid data after cleaning. Skipping.")
            continue

        df['StockID'] = sheet_name
        processed_dfs.append(df)

    if not processed_dfs:
        raise ValueError("No valid data could be loaded from any sheet in the Excel file.")

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.sort_values(by=['StockID', 'Date'], inplace=True)

    print(f"\nSuccessfully loaded and combined data from all stocks.")
    print(f"  - Total rows in combined DataFrame: {len(combined_df)}")
    print(f"  - Unique stocks found: {combined_df['StockID'].nunique()}")

    return combined_df
