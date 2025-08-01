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
        xls = pd.ExcelFile(config.FILE_PATH)
        sheet_names = xls.sheet_names
        print(f"Found {len(sheet_names)} sheets (stocks) in the file: {sheet_names}")
    except FileNotFoundError:
        print(f"Error: The file '{config.FILE_PATH}' was not found.")
        raise

    processed_dfs = []
    for sheet_name in sheet_names:
        try:
            # Read the sheet using the known structure: header is the first row,
            # skip the next two metadata rows.
            df = pd.read_excel(
                config.FILE_PATH,
                sheet_name=sheet_name,
                header=0,
                skiprows=[1, 2]
            )

            if df.empty:
                print(f"    - WARNING: Sheet '{sheet_name}' is empty after loading. Skipping.")
                continue

            # --- START OF FINAL FIX ---
            # 1. Standardize all column names to lowercase for robust access.
            df.columns = [str(c).lower().strip() for c in df.columns]

            # 2. The first column is the date column, but its header was 'Price'.
            #    After lowercasing, it's 'price'. We rename it to 'date' for clarity.
            df.rename(columns={'price': 'date'}, inplace=True)

            # 3. Define the required columns using lowercase.
            required_cols = ['date'] + config.FEATURE_COLUMNS

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"    - WARNING: Sheet '{sheet_name}' is missing columns: {missing_cols}. Skipping.")
                continue

            # 4. Add StockID.
            df['StockID'] = sheet_name

            # 5. Convert date column and feature columns to their correct types.
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            for col in config.FEATURE_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 6. Drop rows with any invalid data.
            df.dropna(subset=['date'] + config.FEATURE_COLUMNS, how='any', inplace=True)

            if df.empty:
                print(f"    - WARNING: Sheet '{sheet_name}' has no valid data after cleaning. Skipping.")
                continue

            # 7. Rename the 'date' column to 'Date' (capital D) to match downstream expectations.
            df.rename(columns={'date': 'Date'}, inplace=True)

            # 8. Select and reorder the final columns.
            final_df = df[['Date', 'StockID'] + config.FEATURE_COLUMNS]
            # --- END OF FINAL FIX ---

            processed_dfs.append(final_df)

        except Exception as e:
            print(f"    - ERROR: Could not process sheet '{sheet_name}'. Error: {e}. Skipping.")
            continue

    if not processed_dfs:
        raise ValueError("No valid data could be loaded from any sheet in the Excel file. Check file format and column names.")

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.sort_values(by=['StockID', 'Date'], inplace=True)

    print(f"\nSuccessfully loaded and combined data from {len(processed_dfs)} sheets.")
    print(f"  - Total rows in combined DataFrame: {len(combined_df)}")
    print(f"  - Unique stocks found: {combined_df['StockID'].nunique()}")

    return combined_df