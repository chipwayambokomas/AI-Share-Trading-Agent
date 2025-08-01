import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

# Suppress harmless warnings for a cleaner output
warnings.filterwarnings("ignore")

def run_forecasting_pipeline(excel_path, sheet_name, output_folder):
    """
    This function runs the entire ARIMA forecasting pipeline following a tidy workflow
    and saves the output.
    """
    print(f"\n{'='*60}")
    print(f"▶️ Starting Analysis for: {sheet_name}")
    print(f"{'='*60}")

    # --- 1. Data Preparation (Tidy) ---
    try:
        # Load the first column (index 0) as the date index.
        df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0)
        # Clean column names to remove hidden spaces.
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Could not read sheet '{sheet_name}'. Error: {e}")
        return

    if 'Price' not in df.columns:
        print(f"Worksheet '{sheet_name}' is missing the 'Price' column. Skipping.")
        return

    # Prepare the time series data
    df.index.name = 'Date'
    df.index = pd.to_datetime(df.index)
    data = df['Price'].asfreq('B').fillna(method='ffill')

    # --- 2. Plot the Data (Visualize) ---
    print(f"Visualizing historical data for {sheet_name}...")
    plt.figure(figsize=(14, 7))
    plt.plot(data)
    plt.title(f'{sheet_name} Closing Price History')
    plt.grid(True)
    plt.show()

    # --- 3. Define and Train the Model (Specify & Estimate) ---
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.9)
    train, test = data.iloc[0:train_size], data.iloc[train_size:]
    
    print(f"\nFinding best ARIMA model for {sheet_name}...")
    # The auto_arima function specifies and estimates the model in one step
    arima_model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, trace=False)

    print("\n--- Model Summary ---")
    print(arima_model.summary())

    # --- 4. Check Model Performance (Evaluate) ---
    # This step is crucial and was implied in the previous script. Here we make it explicit.
    print(f"\n--- Model Evaluation for {sheet_name} ---")
    predictions = arima_model.predict(n_periods=len(test))
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Visualize the evaluation
    plt.figure(figsize=(14, 7))
    plt.plot(train, label='Training Data')
    plt.plot(test, label='Actual Prices (Test)', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'ARIMA Model Evaluation for {sheet_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 5. Produce and Save Forecasts (Forecast) ---
    print(f"\nForecasting future values for {sheet_name}...")
    # Refit the model on the full dataset for the final forecast
    final_model = auto_arima(data, seasonal=False, stepwise=True, suppress_warnings=True, trace=False)
    
    future_forecast, conf_int = final_model.predict(n_periods=30, return_conf_int=True)
    future_index = pd.date_range(start=data.index[-1], periods=31, freq='B')[1:]
    forecast_df = pd.DataFrame(future_forecast, index=future_index, columns=['Forecast'])
    conf_int_df = pd.DataFrame(conf_int, index=future_index, columns=['Lower Confidence', 'Upper Confidence'])
    
    # Combine the forecast and confidence intervals into one DataFrame
    final_forecast_df = pd.concat([forecast_df, conf_int_df], axis=1)

    # Save the forecast data to a CSV file
    csv_filename = os.path.join(output_folder, f"{sheet_name}_forecast.csv")
    final_forecast_df.to_csv(csv_filename)
    print(f"✅ Forecast data saved to: {csv_filename}")

    # Save the forecast plot to a PNG image file
    plt.figure(figsize=(14, 7))
    plt.plot(data['2023':], label='Historical Data (from 2023)')
    plt.plot(forecast_df, label='Future Forecast', color='green')
    plt.fill_between(conf_int_df.index, conf_int_df['Lower Confidence'], conf_int_df['Upper Confidence'], color='k', alpha=.15, label='95% Confidence Interval')
    plt.title(f'Future Forecast for {sheet_name} (Next 30 Business Days)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(output_folder, f"{sheet_name}_forecast_plot.png")
    plt.savefig(plot_filename)
    print(f"✅ Forecast plot saved to: {plot_filename}")
    plt.close() # Close the plot to prevent it from popping up

    print(f"✅ Analysis for {sheet_name} complete.")


if __name__ == "__main__":
    excel_file_path = "C:\\Users\\PIC\\Desktop\\simple_pipeline\\JSE_Top40_OHLCV_2014_2024.xlsx"
    
    # Create an output folder if it doesn't exist
    output_directory = "forecast_outputs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not os.path.exists(excel_file_path):
        print(f"--- FATAL ERROR ---")
        print(f"The Excel file was not found at: {excel_file_path}")
    else:
        xls = pd.ExcelFile(excel_file_path)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} sheets to analyze: {sheet_names}")
        
        for sheet in sheet_names:
            run_forecasting_pipeline(excel_file_path, sheet, output_directory)
            
        print(f"\n{'='*60}")
        print("🎉 All worksheets have been analyzed.")
        print(f"{'='*60}")