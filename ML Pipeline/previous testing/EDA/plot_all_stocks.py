import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Suppress any harmless warnings
warnings.filterwarnings("ignore")

def plot_all_stock_prices(excel_path):
    """
    This function reads a cleaned Excel file containing multiple stock data sheets
    and plots the closing price of all stocks on a single chart.
    """
    print(f"--- Loading Data from: {excel_path} ---")

    # Check if the Excel file exists
    if not os.path.exists(excel_path):
        print(f"--- ERROR ---")
        print(f"The file was not found at: {excel_path}")
        print("Please make sure the cleaned Excel file is in the same folder as the script.")
        return

    try:
        # Load the entire Excel file to get all sheet names
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        print(f"Found {len(sheet_names)} stocks to plot.")

        # --- Create the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
        plt.figure(figsize=(16, 9)) # Create a large figure to hold the plot

        # --- Loop Through Each Sheet and Plot the Data ---
        for sheet_name in sheet_names:
            # Read the data from the current sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Ensure 'Date' and 'close' columns exist
            if 'Date' in df.columns and 'close' in df.columns:
                # Convert the 'Date' column to datetime objects for proper plotting
                df['Date'] = pd.to_datetime(df['Date'])
                # Plot the 'Date' vs. the 'close' price
                plt.plot(df['Date'], df['close'], label=sheet_name)
            else:
                print(f"⚠️ Warning: Sheet '{sheet_name}' is missing 'Date' or 'close' column. Skipping.")

        # --- Finalize the Plot ---
        plt.title('JSE Top 40 Closing Prices (2014-2024)', fontsize=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Closing Price (ZAR)', fontsize=14)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Stocks") # Place legend outside the plot
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
        plt.grid(True)
        
        # Save the plot to a file
        output_filename = "JSE_Top40_Price_Comparison.png"
        plt.savefig(output_filename, dpi=300)
        print(f"\n✅ Plot saved as '{output_filename}'")
        
        # Display the plot
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # The name of your cleaned Excel file
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    
    # Get the full path for the input file located in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, cleaned_excel_file)
    
    # Run the plotting function
    plot_all_stock_prices(full_path)