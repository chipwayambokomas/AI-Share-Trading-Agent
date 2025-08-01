import pandas as pd

# Define the names for your input and output files
# ---- IMPORTANT: Change this to the actual name of your CSV file ----
input_csv_path = 'evaluation_results_POINT_GraphWaveNet_20_5.csv'
output_csv_path = 'evaluation_results_POINT_GraphWaveNet_20_5_sorted_data.csv'

try:
    # 1. Read the data from the specified CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # 2. Convert the 'Date' column to datetime objects
    # This is essential for sorting chronologically rather than alphabetically.
    # Pandas is generally smart about recognizing common date formats.
    df['Date'] = pd.to_datetime(df['Date'])

    # 3. Sort the DataFrame by 'Date', then by 'StockID' and 'Horizon_S'
    # Sorting by multiple columns ensures a consistent order for records on the same date.
    sorted_df = df.sort_values(by=['Date', 'StockID', 'Horizon_Step'], ascending=True)

    # 4. Print the sorted DataFrame to the console for review
    # to_string() ensures the entire DataFrame is displayed.
    print("--- Data Organized by Date ---")
    print(sorted_df.to_string(index=False))

    # 5. Save the sorted DataFrame to a new CSV file
    # The index=False argument is important to prevent pandas from writing
    # the row numbers into a new column in your CSV file.
    sorted_df.to_csv(output_csv_path, index=False)

    print(f"\nSuccessfully sorted the data and saved it to '{output_csv_path}'")

except FileNotFoundError:
    print(f"Error: The file '{input_csv_path}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script, or provide the full file path.")
except KeyError as e:
    print(f"Error: A required column was not found: {e}.")
    print("Please ensure your CSV file contains the columns 'Date', 'StockID', and 'Horizon_S'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")