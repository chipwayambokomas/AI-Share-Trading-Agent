import pandas as pd
import os

def clean_jse_worksheet(excel_path, output_path):
    """
    This script reads an Excel file, cleans the headers of each sheet
    as per the specified requirements, and saves the cleaned data to a new Excel file.
    """
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        print(f"Found {len(sheet_names)} sheets to process.")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name in sheet_names:
                print(f"\n--- Processing Sheet: {sheet_name} ---")

                # --- The Cleaning Process ---
                # 1. Read the data skipping the first two irrelevant header rows.
                #    The 3rd row (index 2) in the Excel file becomes the header.
                df_cleaned = pd.read_excel(excel_path, sheet_name=sheet_name, header=2)

                # --- "Before" State ---
                print("--- Columns Before Cleaning ---")
                print(df_cleaned.columns.tolist())

                # 2. Define the new, correct column names.
                new_column_names = ['Date', 'open', 'high', 'low', 'close', 'vwap']
                
                # 3. Check if the number of columns matches before renaming.
                if len(df_cleaned.columns) >= len(new_column_names):
                    # Keep only the columns we need and rename them.
                    df_cleaned = df_cleaned.iloc[:, :len(new_column_names)]
                    df_cleaned.columns = new_column_names
                else:
                    print(f"⚠️ Warning: Sheet '{sheet_name}' has fewer columns than expected. Skipping.")
                    continue

                # --- "After" State ---
                print("\n--- Columns After Cleaning ---")
                print(df_cleaned.columns.tolist())
                print("\n--- First 3 Rows of Cleaned Data ---")
                print(df_cleaned.head(3))
                
                # 4. Save the cleaned DataFrame to the new Excel file.
                df_cleaned.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"✅ Sheet '{sheet_name}' has been cleaned and saved.")

        print(f"\n🎉 All sheets have been processed and saved to '{output_path}'")

    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"The file was not found at: {excel_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define the input and output file paths
    input_excel_file = "JSE_Top40_OHLCV_2014_2024.xlsx"
    output_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"

    # Get the full path for the input file in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(script_directory, input_excel_file)
    full_output_path = os.path.join(script_directory, output_excel_file)

    # Run the cleaning function
    clean_jse_worksheet(full_input_path, full_output_path)