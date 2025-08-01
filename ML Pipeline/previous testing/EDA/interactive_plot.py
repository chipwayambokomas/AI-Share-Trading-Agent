import pandas as pd
import plotly.graph_objects as go
import os
import warnings

# Suppress any harmless warnings
warnings.filterwarnings("ignore")

def create_interactive_html_plot(excel_path, output_filename="JSE_Top40_Interactive_Plot.html"):
    """
    This function reads a cleaned Excel file and creates an interactive HTML plot
    of all stock prices using Plotly.
    """
    print(f"--- Loading Data from: {excel_path} ---")

    # Check if the Excel file exists
    if not os.path.exists(excel_path):
        print(f"--- ERROR ---")
        print(f"The file was not found at: {excel_path}")
        return

    try:
        # Load the entire Excel file to get all sheet names
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        print(f"Found {len(sheet_names)} stocks to plot.")

        # --- Create a Plotly Figure ---
        fig = go.Figure()

        # --- Loop Through Each Sheet and Add a Trace to the Figure ---
        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            if 'Date' in df.columns and 'close' in df.columns:
                # Add a line for the current stock
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['close'],
                    name=sheet_name,  # This name appears in the legend and on hover
                    mode='lines'
                ))
            else:
                print(f"⚠️ Warning: Sheet '{sheet_name}' is missing 'Date' or 'close' column. Skipping.")

        # --- Customize the Layout of the Plot ---
        fig.update_layout(
            title_text='JSE Top 40 Closing Prices (2014-2024)',
            xaxis_title='Date',
            yaxis_title='Closing Price (ZAR)',
            legend_title_text='Stocks',
            hovermode='x unified' # Shows all stock prices for a given date on hover
        )

        # --- Save the Plot to an HTML File ---
        fig.write_html(output_filename)
        print(f"\n✅ Interactive plot saved as '{output_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # The name of your cleaned Excel file
    cleaned_excel_file = "JSE_Top40_OHLCV_2014_2024_CLEANED.xlsx"
    
    # Get the full path for the input file located in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, cleaned_excel_file)
    
    # Run the plotting function
    create_interactive_html_plot(full_path)