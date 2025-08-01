# src/pipeline/stage_5_model_evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import print_header

def _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    """
    Evaluates the model for trend prediction, converting raw slopes to angles.
    """
    print("\nEvaluating TREND prediction performance...")
    predicted_scaled = model(X_test_t).detach().numpy()
    actual_scaled = y_test_t.numpy()

    # --- Step 1: Inverse transform to get raw slope and duration values ---
    actual_raw_slopes = scalers['slope'].inverse_transform(actual_scaled[:, 0].reshape(-1, 1))
    actual_durations = scalers['duration'].inverse_transform(actual_scaled[:, 1].reshape(-1, 1))
    predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_scaled[:, 0].reshape(-1, 1))
    predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[:, 1].reshape(-1, 1))

    # --- Step 2: Convert raw slopes to angles in degrees ---
    # This implements: slope_angle = degrees(arctan(raw_slope))
    actual_slope_angles = np.degrees(np.arctan(actual_raw_slopes))
    predicted_slope_angles = np.degrees(np.arctan(predicted_raw_slopes))

    # --- Step 3: Calculate metrics on the slope angles ---
    rmse_slope_angle = np.sqrt(mean_squared_error(actual_slope_angles, predicted_slope_angles))
    mae_slope_angle = mean_absolute_error(actual_slope_angles, predicted_slope_angles)
    print(f"\n  --- Slope Angle Prediction (in degrees) ---")
    print(f"  - RMSE: {rmse_slope_angle:.4f}, MAE: {mae_slope_angle:.4f}")

    # --- Step 4: Calculate metrics on the duration ---
    rmse_duration = np.sqrt(mean_squared_error(actual_durations, predicted_durations))
    mae_duration = mean_absolute_error(actual_durations, predicted_durations)
    print(f"\n  --- Duration Prediction ---")
    print(f"  - RMSE: {rmse_duration:.2f} (days), MAE: {mae_duration:.2f} (days)")

    # --- Step 5: Save results with slope angles to CSV ---
    results_df = pd.DataFrame({
        'StockID': stock_ids_test,
        'Actual_Slope_Angle': actual_slope_angles.flatten(),
        'Predicted_Slope_Angle': predicted_slope_angles.flatten(),
        'Actual_Duration': actual_durations.flatten(),
        'Predicted_Duration': predicted_durations.flatten()
    })
    save_path = f"evaluation_results_TREND_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nTrend evaluation results (with slope angles) saved to '{save_path}'.")


def _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    print("\nEvaluating POINT prediction performance...")
    predicted_scaled = model(X_test_t).detach().numpy()
    actual_scaled = y_test_t.numpy()

    try:
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    except ValueError:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in FEATURE_COLUMNS.")
    
    num_features = len(config.FEATURE_COLUMNS)
    actual_prices, predicted_prices = [], []

    for i in range(len(X_test_t)):
        stock_id = stock_ids_test[i]
        scaler = scalers[stock_id]
        
        dummy_actual = np.zeros((1, num_features))
        dummy_predicted = np.zeros((1, num_features))
        
        dummy_actual[0, target_col_idx] = actual_scaled[i, 0]
        dummy_predicted[0, target_col_idx] = predicted_scaled[i, 0]
        
        actual_inv = scaler.inverse_transform(dummy_actual)
        predicted_inv = scaler.inverse_transform(dummy_predicted)
        
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
