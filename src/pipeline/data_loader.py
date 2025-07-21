import pandas as pd
from ..utils import print_header

def run(settings):
    """
    Stage 1: Reads data from all sheets, combines them, adds a 'StockID',
    and performs initial cleaning and filtering.
    """
    print_header("Stage 1: Data Loading & Feature Extraction")
    print(f"Loading data from '{settings.FILE_PATH}'...")

    try:
        # Load the Excel file and get all sheet names
        xls = pd.ExcelFile(settings.FILE_PATH)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        print(f"Error: The file '{settings.FILE_PATH}' was not found.")
        raise

    processed_dfs = []
    for sheet_name in sheet_names:
        try:
            # Read each sheet into a DataFrame
            df = pd.read_excel(
                settings.FILE_PATH,
                sheet_name=sheet_name,
                header=0,
                skiprows=[1, 2]
            )

            if df.empty:
                continue
            
            # convert all column names to lowercase and strip whitespace and rename 'price' to 'date'
            df.columns = [str(c).lower().strip() for c in df.columns]
            df.rename(columns={'price': 'date'}, inplace=True)

            # Check if required columns are present
            required_cols = ['date'] + settings.FEATURE_COLUMNS
            if not all(col in df.columns for col in required_cols):
                continue

            # Add 'StockID' column and convert 'date' to datetime
            df['StockID'] = sheet_name
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Convert feature columns to numeric, coercing errors
            for col in settings.FEATURE_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaN in 'date' or any feature column
            df.dropna(subset=['date'] + settings.FEATURE_COLUMNS, how='any', inplace=True)
            if df.empty:
                continue
            
            # Ensure 'date' is the first column and sort by 'StockID' and 'date'
            start_date = df['date'].min()
            if start_date.year != 2014:
                continue
            # Ensure 'date' is the first column
            df.rename(columns={'date': 'Date'}, inplace=True)
            final_df = df[['Date', 'StockID'] + settings.FEATURE_COLUMNS]
            processed_dfs.append(final_df)

        except Exception as e:
            print(f"    - ERROR: Could not process sheet '{sheet_name}'. Error: {e}. Skipping.")
            continue

    if not processed_dfs:
        raise ValueError("No valid data could be loaded. Check file format, columns, and dates.")

    # Combine all processed DataFrames into one
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.sort_values(by=['StockID', 'Date'], inplace=True)

    print(f"Successfully loaded and combined data from {len(processed_dfs)} sheets.")
    print(f"Total rows: {len(combined_df)}, Unique stocks: {combined_df['StockID'].nunique()}")
    
    return combined_df