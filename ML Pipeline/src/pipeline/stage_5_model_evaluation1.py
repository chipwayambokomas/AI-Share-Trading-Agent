# src/pipeline/stage_5_model_evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import print_header
import networkx as nx
from torch.utils.data import TensorDataset, DataLoader

def _calculate_graph_metrics(adj_matrix, stock_ids, config):
    """
    Calculates and prints graph-based network metrics from the adjacency matrix.
    """
    print_header("Stage 5: Network Analysis Metrics")
    print("Purpose: Evaluate the learned dependency structure using network theory.")

    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    G = nx.from_numpy_array(adj_matrix)
    label_mapping = {i: stock_id for i, stock_id in enumerate(stock_ids)}
    G = nx.relabel_nodes(G, label_mapping)

    print("\nCalculating network centrality metrics...")

    degree_centrality = {n: f"{d:.4f}" for n, d in G.degree(weight='weight')}
    betweenness_centrality = {n: f"{b:.4f}" for n, b in nx.betweenness_centrality(G, weight='weight').items()}
    try:
        eigenvector_centrality = {n: f"{e:.4f}" for n, e in nx.eigenvector_centrality(G, weight='weight', max_iter=1000).items()}
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {"Error": "Algorithm failed to converge."}

    clustering_coefficient = {n: f"{c:.4f}" for n, c in nx.clustering(G, weight='weight').items()}

    try:
        communities = nx.community.louvain_communities(G, weight='weight')
        modularity = nx.community.modularity(G, communities, weight='weight')
        print(f"\nDetected {len(communities)} communities with a modularity score of: {modularity:.4f}")
    except:
        modularity = "Could not be calculated."
        communities = []
        print("\nCommunity detection failed.")

    metrics_df = pd.DataFrame({
        'Degree Centrality': pd.Series(degree_centrality),
        'Betweenness Centrality': pd.Series(betweenness_centrality),
        'Eigenvector Centrality': pd.Series(eigenvector_centrality),
        'Clustering Coefficient': pd.Series(clustering_coefficient)
    })

    print("\n--- Network Metrics per Stock ---")
    print(metrics_df)


def _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    """
    Evaluates the model for trend prediction, converting raw slopes to angles.
    """
    print("\nEvaluating TREND prediction performance...")
    model.eval()

    # Create a DataLoader for the test set to handle batching
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for X_batch_test, y_batch_test in test_loader:
            pred_batch = model(X_batch_test)
            
            # This logic must match the permutation logic in stage_4
            if config.MODEL_TYPE == 'GraphWaveNet':
                 pred_batch = pred_batch[..., -1:]
                 if config.PREDICTION_MODE == 'TREND':
                     pred_batch = pred_batch.squeeze(-1).permute(0, 2, 1)

            all_predictions.append(pred_batch.cpu().numpy())
            all_actuals.append(y_batch_test.cpu().numpy())
    
    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = np.concatenate(all_actuals, axis=0)

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        actual_slopes_scaled = actual_scaled[..., 0]
        actual_durations_scaled = actual_scaled[..., 1]
        predicted_slopes_scaled = predicted_scaled[..., 0]
        predicted_durations_scaled = predicted_scaled[..., 1]

        # --- START OF FIX ---
        # Remove the incorrect `.reshape(-1, 1)` from all four lines.
        # The data already has the correct shape (n_samples, n_nodes) for the scaler.
        actual_raw_slopes = scalers['slope'].inverse_transform(actual_slopes_scaled).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_durations_scaled).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_slopes_scaled).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_durations_scaled).flatten()
        # --- END OF FIX ---
        

        
        stock_ids_test_flat = np.tile(stock_ids_test, actual_scaled.shape[0])

    else: # Original logic for TCN/MLP
        actual_raw_slopes = scalers['slope'].inverse_transform(actual_scaled[:, 0].reshape(-1, 1)).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_scaled[:, 1].reshape(-1, 1)).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_scaled[:, 0].reshape(-1, 1)).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[:, 1].reshape(-1, 1)).flatten()
        stock_ids_test_flat = stock_ids_test

    actual_slope_angles = np.degrees(np.arctan(actual_raw_slopes))
    predicted_slope_angles = np.degrees(np.arctan(predicted_raw_slopes))

    rmse_slope_angle = np.sqrt(mean_squared_error(actual_slope_angles, predicted_slope_angles))
    mae_slope_angle = mean_absolute_error(actual_slope_angles, predicted_slope_angles)
    mape_slope_angle = np.mean(np.abs((actual_slope_angles - predicted_slope_angles) / (actual_slope_angles + 1e-8))) * 100

    print(f"\n  --- Slope Angle Prediction (in degrees) ---")
    print(f"  - RMSE: {rmse_slope_angle:.4f}, MAE: {mae_slope_angle:.4f}, MAPE: {mape_slope_angle:.2f}%")

    rmse_duration = np.sqrt(mean_squared_error(actual_durations, predicted_durations))
    mae_duration = mean_absolute_error(actual_durations, predicted_durations)
    mape_duration = np.mean(np.abs((actual_durations - predicted_durations) / (actual_durations + 1e-8))) * 100

    print(f"\n  --- Duration Prediction ---")
    print(f"  - RMSE: {rmse_duration:.2f} (days), MAE: {mae_duration:.2f} (days), MAPE: {mape_duration:.2f}%")

    results_df = pd.DataFrame({
        'StockID': stock_ids_test_flat,
        'Actual_Slope_Angle': actual_slope_angles,
        'Predicted_Slope_Angle': predicted_slope_angles,
        'Actual_Duration': actual_durations,
        'Predicted_Duration': predicted_durations
    })
    save_path = f"evaluation_results_TREND_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nTrend evaluation results saved to '{save_path}'.")


def _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config):
    print("\nEvaluating POINT prediction performance...")
    model.eval()

    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for X_batch_test in test_loader:
            pred_batch = model(X_batch_test[0])
            
            if config.MODEL_TYPE == 'GraphWaveNet':
                # The model outputs a sequence; we take only the last time step.
                pred_batch = pred_batch[..., -1:]
            
            all_predictions.append(pred_batch.cpu().numpy())

    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = y_test_t.cpu().numpy()

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        if actual_scaled.ndim == 4:
            actual_scaled = actual_scaled.squeeze()
        if predicted_scaled.ndim > 2:
            predicted_scaled = predicted_scaled.squeeze()

        all_actual_prices, all_predicted_prices = [], []
        stock_ids_ordered = stock_ids_test
        
        num_samples = actual_scaled.shape[0]
        num_nodes = actual_scaled.shape[1]
        
        actual_flat = actual_scaled.reshape(num_samples * num_nodes)
        predicted_flat = predicted_scaled.reshape(num_samples * num_nodes)
        
        stock_ids_tiled = np.tile(stock_ids_ordered, num_samples)

        for i in range(len(actual_flat)):
            stock_id = stock_ids_tiled[i]
            scaler = scalers[stock_id]
            num_features = len(config.FEATURE_COLUMNS)
            target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)

            dummy_actual = np.zeros((1, num_features))
            dummy_predicted = np.zeros((1, num_features))

            dummy_actual[0, target_col_idx] = actual_flat[i]
            dummy_predicted[0, target_col_idx] = predicted_flat[i]

            all_actual_prices.append(scaler.inverse_transform(dummy_actual)[0, target_col_idx])
            all_predicted_prices.append(scaler.inverse_transform(dummy_predicted)[0, target_col_idx])

        actual_prices = np.array(all_actual_prices)
        predicted_prices = np.array(all_predicted_prices)

    else: # Original logic for TCN/MLP
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
        num_features = len(config.FEATURE_COLUMNS)
        actual_prices, predicted_prices = [], []

        actual_scaled_flat = actual_scaled.flatten()
        predicted_scaled_flat = predicted_scaled.flatten()

        for i in range(len(stock_ids_test)):
            stock_id = stock_ids_test[i]
            scaler = scalers[stock_id]
            dummy_actual = np.zeros((1, num_features))
            dummy_predicted = np.zeros((1, num_features))
            dummy_actual[0, target_col_idx] = actual_scaled_flat[i]
            dummy_predicted[0, target_col_idx] = predicted_scaled_flat[i]

            actual_inv = scaler.inverse_transform(dummy_actual)
            predicted_inv = scaler.inverse_transform(dummy_predicted)

            actual_prices.append(actual_inv[0, target_col_idx])
            predicted_prices.append(predicted_inv[0, target_col_idx])

        actual_prices = np.array(actual_prices)
        predicted_prices = np.array(predicted_prices)

    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-8))) * 100

    print("\nQuantitative Metrics on Combined Test Set (All Stocks):")
    print(f"  - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    results_df = pd.DataFrame({
        'StockID': stock_ids_tiled if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN'] else stock_ids_test,
        'Actual_Price': actual_prices, 
        'Predicted_Price': predicted_prices
    })
    save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint prediction evaluation results saved to '{save_path}'.")

def run(model, X_test_t, y_test_t, stock_ids_test, scalers, config, adj_matrix=None):
    print_header("Stage 5: Model Evaluation")
    print(f"Purpose: Assess final model performance for {config.PREDICTION_MODE} mode.")

    if config.PREDICTION_MODE == "TREND":
        _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
    elif config.PREDICTION_MODE == "POINT":
        _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
    else:
        raise ValueError(f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'.")

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN'] and adj_matrix is not None:
        stock_ids_ordered = sorted(scalers.keys())
        _calculate_graph_metrics(adj_matrix, stock_ids_ordered, config)