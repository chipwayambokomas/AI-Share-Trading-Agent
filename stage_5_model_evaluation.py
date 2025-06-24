# src/pipeline/stage_5_model_evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import print_header

def _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    print("\nEvaluating TREND prediction performance...")
    predicted_scaled = model(X_test_t).detach().numpy()
    actual_scaled = y_test_t.numpy()

    actual_slopes = scalers['slope'].inverse_transform(actual_scaled[:, 0].reshape(-1, 1))
    actual_durations = scalers['duration'].inverse_transform(actual_scaled[:, 1].reshape(-1, 1))
    predicted_slopes = scalers['slope'].inverse_transform(predicted_scaled[:, 0].reshape(-1, 1))
    predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[:, 1].reshape(-1, 1))

    rmse_slope = np.sqrt(mean_squared_error(actual_slopes, predicted_slopes))
    mae_slope = mean_absolute_error(actual_slopes, predicted_slopes)
    print(f"\n  --- Slope Prediction ---")
    print(f"  - RMSE: {rmse_slope:.4f}, MAE: {mae_slope:.4f}")

    rmse_duration = np.sqrt(mean_squared_error(actual_durations, predicted_durations))
    mae_duration = mean_absolute_error(actual_durations, predicted_durations)
    print(f"\n  --- Duration Prediction ---")
    print(f"  - RMSE: {rmse_duration:.2f} (days), MAE: {mae_duration:.2f} (days)")

    results_df = pd.DataFrame({
        'StockID': stock_ids_test,
        'Actual_Slope': actual_slopes.flatten(), 'Predicted_Slope': predicted_slopes.flatten(),
        'Actual_Duration': actual_durations.flatten(), 'Predicted_Duration': predicted_durations.flatten()
    })
    save_path = f"evaluation_results_TREND_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nTrend evaluation results saved to '{save_path}'.")

def _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    print("\nEvaluating POINT prediction performance...")
    predicted_scaled = model(X_test_t).detach().numpy()
    actual_scaled = y_test_t.numpy()

    # --- FIX: Handle inverse transform for multivariate data ---
    try:
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    except ValueError:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in FEATURE_COLUMNS.")
    
    num_features = len(config.FEATURE_COLUMNS)
    actual_prices, predicted_prices = [], []

    for i in range(len(X_test_t)):
        stock_id = stock_ids_test[i]
        scaler = scalers[stock_id]
        
        # Create dummy arrays with the same shape the scaler was trained on
        dummy_actual = np.zeros((1, num_features))
        dummy_predicted = np.zeros((1, num_features))
        
        # Place the single predicted/actual value into the correct column (the target column)
        # Note: y_test_t and predicted_scaled have shape (samples, output_window_size)
        dummy_actual[0, target_col_idx] = actual_scaled[i, 0]
        dummy_predicted[0, target_col_idx] = predicted_scaled[i, 0]
        
        # Inverse transform the dummy arrays
        actual_inv = scaler.inverse_transform(dummy_actual)
        predicted_inv = scaler.inverse_transform(dummy_predicted)
        
        # Extract the unscaled value from the target column
        actual_prices.append(actual_inv[0, target_col_idx])
        predicted_prices.append(predicted_inv[0, target_col_idx])

    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)

    print("\nQuantitative Metrics on Combined Test Set (All Stocks):")
    print(f"  - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    results_df = pd.DataFrame({
        'StockID': stock_ids_test,
        'Actual_Price': actual_prices, 'Predicted_Price': predicted_prices
    })
    save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint prediction evaluation results saved to '{save_path}'.")

def run(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    print_header("Stage 5: Model Evaluation")
    print(f"Purpose: Assess final model performance for {config.PREDICTION_MODE} mode.")
    model.eval()
    with torch.no_grad():
        if config.PREDICTION_MODE == "TREND":
            _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
        elif config.PREDICTION_MODE == "POINT":
            _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
        else:
            raise ValueError(f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'.")