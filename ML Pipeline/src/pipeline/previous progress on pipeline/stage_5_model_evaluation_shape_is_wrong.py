# src/pipeline/stage_5_model_evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import print_header
import networkx as nx

def _calculate_graph_metrics(adj_matrix, stock_ids, config):
    """
    Calculates and prints graph-based network metrics from the adjacency matrix.
    """
    print_header("Stage 5: Network Analysis Metrics")
    print("Purpose: Evaluate the learned dependency structure using network theory.")

    # Ensure matrix is on CPU and in numpy format
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Create a mapping from integer index to stock ID for labeling
    label_mapping = {i: stock_id for i, stock_id in enumerate(stock_ids)}
    G = nx.relabel_nodes(G, label_mapping)

    print("\nCalculating network centrality metrics...")

    # 1. Weighted Degree Centrality
    degree_centrality = {n: f"{d:.4f}" for n, d in G.degree(weight='weight')}

    # 2. Weighted Betweenness Centrality
    betweenness_centrality = {n: f"{b:.4f}" for n, b in nx.betweenness_centrality(G, weight='weight').items()}

    # 3. Weighted Eigenvector Centrality
    try:
        eigenvector_centrality = {n: f"{e:.4f}" for n, e in nx.eigenvector_centrality(G, weight='weight', max_iter=1000).items()}
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {"Error": "Algorithm failed to converge."}

    # 4. Weighted Local Clustering Coefficient
    clustering_coefficient = {n: f"{c:.4f}" for n, c in nx.clustering(G, weight='weight').items()}

    # 5. Community Detection and Modularity
    try:
        communities = nx.community.louvain_communities(G, weight='weight')
        modularity = nx.community.modularity(G, communities, weight='weight')
        print(f"\nDetected {len(communities)} communities with a modularity score of: {modularity:.4f}")
    except:
        modularity = "Could not be calculated."
        communities = []
        print("\nCommunity detection failed.")

    # Combine metrics into a DataFrame for clear output
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
    Handles both standard and graph-based model outputs.
    """
    print("\nEvaluating TREND prediction performance...")
    model.eval()
    with torch.no_grad():
        predicted_scaled_tensor = model(X_test_t)

    predicted_scaled = predicted_scaled_tensor.detach().cpu().numpy()
    actual_scaled = y_test_t.cpu().numpy()

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        if config.MODEL_TYPE == 'GraphWaveNet':
            # For GraphWaveNet: (batch, out_seq, nodes, features) -> (batch, nodes, out_seq, features)
            predicted_scaled = predicted_scaled.transpose(0, 2, 1, 3)

        # For both graph models, squeeze the feature dimension if it's 1
        if predicted_scaled.shape[-1] == 1:
            predicted_scaled = predicted_scaled.squeeze(-1)

        # Assuming the output order is [slope, duration]
        actual_slopes_scaled = actual_scaled[..., 0]
        actual_durations_scaled = actual_scaled[..., 1]
        predicted_slopes_scaled = predicted_scaled[..., 0]
        predicted_durations_scaled = predicted_scaled[..., 1]

        # Inverse transform the scaled values
        actual_raw_slopes = scalers['slope'].inverse_transform(actual_slopes_scaled.reshape(-1, 1)).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_durations_scaled.reshape(-1, 1)).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_slopes_scaled.reshape(-1, 1)).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_durations_scaled.reshape(-1, 1)).flatten()

        # Flatten stock IDs for results DataFrame
        if config.MODEL_TYPE == 'GraphWaveNet':
             stock_ids_test_flat = np.tile(stock_ids_test, actual_scaled.shape[0] * actual_scaled.shape[2])
        else: # DSTAGNN
            stock_ids_test_flat = np.tile(stock_ids_test, actual_scaled.shape[0])


    else: # Original logic for TCN/MLP
        actual_raw_slopes = scalers['slope'].inverse_transform(actual_scaled[:, 0].reshape(-1, 1)).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_scaled[:, 1].reshape(-1, 1)).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_scaled[:, 0].reshape(-1, 1)).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[:, 1].reshape(-1, 1)).flatten()
        stock_ids_test_flat = stock_ids_test

    actual_slope_angles = np.degrees(np.arctan(actual_raw_slopes))
    predicted_slope_angles = np.degrees(np.arctan(predicted_raw_slopes))

    # --- Calculate and Print Metrics ---
    rmse_slope_angle = np.sqrt(mean_squared_error(actual_slope_angles, predicted_slope_angles))
    mae_slope_angle = mean_absolute_error(actual_slope_angles, predicted_slope_angles)
    mape_slope_angle = np.mean(np.abs((actual_slope_angles - predicted_slope_angles) / (actual_slope_angles + 1e-8))) * 100

    print(f"\n  --- Slope Angle Prediction (in degrees) ---")
    print(f"  - RMSE: {rmse_slope_angle:.4f}, MAE: {mae_slope_angle:.4f}, MAPE: {mape_slope_angle:.2f}%")

    # --- START OF FIX ---
    # Calculate and print duration metrics
    rmse_duration = np.sqrt(mean_squared_error(actual_durations, predicted_durations))
    mae_duration = mean_absolute_error(actual_durations, predicted_durations)
    mape_duration = np.mean(np.abs((actual_durations - predicted_durations) / (actual_durations + 1e-8))) * 100

    print(f"\n  --- Duration Prediction ---")
    print(f"  - RMSE: {rmse_duration:.2f} (days), MAE: {mae_duration:.2f} (days), MAPE: {mape_duration:.2f}%")
    # --- END OF FIX ---

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
    with torch.no_grad():
        predicted_scaled_tensor = model(X_test_t)

    predicted_scaled = predicted_scaled_tensor.detach().cpu().numpy()
    actual_scaled = y_test_t.cpu().numpy()

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        if config.MODEL_TYPE == 'GraphWaveNet':
            if predicted_scaled.ndim == 4:
                 predicted_scaled = predicted_scaled[..., -1]
            if actual_scaled.ndim == 4:
                actual_scaled = actual_scaled.squeeze(-1)
        # DSTAGNN output is (batch, nodes, out_seq), so no transpose is needed
        # actual_scaled for DSTAGNN is (batch, out_seq, nodes) -> need to align

        if config.MODEL_TYPE == 'DSTAGNN':
            actual_scaled = actual_scaled.transpose(0, 2, 1) # Now (batch, nodes, out_seq)


        all_actual_prices, all_predicted_prices = [], []
        stock_ids_ordered = stock_ids_test

        for i in range(actual_scaled.shape[0]): # Iterate through batches
            for j in range(actual_scaled.shape[1]): # Iterate through nodes (stocks)
                stock_id = stock_ids_ordered[j]
                scaler = scalers[stock_id]
                num_features = len(config.FEATURE_COLUMNS)
                target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)

                for t in range(actual_scaled.shape[2]): # Iterate through the output sequence
                    dummy_actual = np.zeros((1, num_features))
                    dummy_predicted = np.zeros((1, num_features))

                    dummy_actual[0, target_col_idx] = actual_scaled[i, j, t]
                    dummy_predicted[0, target_col_idx] = predicted_scaled[i, j, t]

                    all_actual_prices.append(scaler.inverse_transform(dummy_actual)[0, target_col_idx])
                    all_predicted_prices.append(scaler.inverse_transform(dummy_predicted)[0, target_col_idx])

        actual_prices = np.array(all_actual_prices)
        predicted_prices = np.array(all_predicted_prices)

    else: # Original logic for TCN/MLP
        target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
        num_features = len(config.FEATURE_COLUMNS)
        actual_prices, predicted_prices = [], []

        # This assumes output_window is 1 for now
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

    if config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN']:
        results_df = pd.DataFrame({
            'StockID': stock_ids_test,
            'Actual_Price': actual_prices, 'Predicted_Price': predicted_prices
        })
        save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}.csv"
        results_df.to_csv(save_path, index=False)
        print(f"\nPoint prediction evaluation results saved to '{save_path}'.")

def run(model, X_test_t, y_test_t, stock_ids_test, scalers, config, adj_matrix=None):
    print_header("Stage 5: Model Evaluation")
    print(f"Purpose: Assess final model performance for {config.PREDICTION_MODE} mode.")

    with torch.no_grad():
        if config.PREDICTION_MODE == "TREND":
            _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
        elif config.PREDICTION_MODE == "POINT":
            _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
        else:
            raise ValueError(f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'.")

    # Calculate and display graph metrics if a graph model was used
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN'] and adj_matrix is not None:
        stock_ids_ordered = sorted(scalers.keys())
        _calculate_graph_metrics(adj_matrix, stock_ids_ordered, config)