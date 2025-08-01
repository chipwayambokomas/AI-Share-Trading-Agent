# src/pipeline/stage_5_model_evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import print_header
import networkx as nx
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def _calculate_graph_metrics(adj_matrix, stock_ids, config):
    """
    Calculates and prints graph-based network metrics from the adjacency matrix.
    (This function remains unchanged).
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


def _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config):
    """
    Evaluates the model for trend prediction.
    (This function remains unchanged).
    """
    print("\nEvaluating TREND prediction performance...")
    model.eval()
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    all_predictions, all_actuals = [], []
    with torch.no_grad():
        for X_batch_test, y_batch_test in test_loader:
            pred_batch = model(X_batch_test)
            
            if config.MODEL_TYPE == 'HSDGNN':
                if config.PREDICTION_MODE == 'TREND':
                    pred_batch = pred_batch.squeeze(-1).permute(0, 2, 1)
            
            elif config.MODEL_TYPE == 'GraphWaveNet':
                 if config.PREDICTION_MODE == 'TREND':
                     pred_batch = pred_batch.permute(0, 2, 1)
                 else:
                     pred_batch = pred_batch[..., -1:]
            
            all_predictions.append(pred_batch.cpu().numpy())
            all_actuals.append(y_batch_test.cpu().numpy())
            
    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = np.concatenate(all_actuals, axis=0)
    
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        actual_slopes_scaled = actual_scaled[..., 0]
        actual_durations_scaled = actual_scaled[..., 1]
        predicted_slopes_scaled = predicted_scaled[..., 0]
        predicted_durations_scaled = predicted_scaled[..., 1]

        actual_raw_slopes = scalers['slope'].inverse_transform(actual_slopes_scaled).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_durations_scaled).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_slopes_scaled).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_durations_scaled).flatten()
    else:
        actual_raw_slopes = scalers['slope'].inverse_transform(actual_scaled[:, 0].reshape(-1, 1)).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_scaled[:, 1].reshape(-1, 1)).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_scaled[:, 0].reshape(-1, 1)).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[:, 1].reshape(-1, 1)).flatten()

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        stock_ids_test_flat = np.tile(stock_ids_test, actual_scaled.shape[0])
        dates_test_flat = np.repeat(dates_test, len(stock_ids_test))
    else: 
        stock_ids_test_flat = stock_ids_test
        dates_test_flat = dates_test

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
        'Date': dates_test_flat, 
        'StockID': stock_ids_test_flat, 
        'Actual_Slope_Angle': actual_slope_angles, 
        'Predicted_Slope_Angle': predicted_slope_angles, 
        'Actual_Duration': actual_durations, 
        'Predicted_Duration': predicted_durations
    })
    save_path = f"evaluation_results_TREND_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nTrend evaluation results saved to '{save_path}'.")


# --- START OF IMPLEMENTATION ---
def _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config):
    """
    Evaluates the model for point prediction (e.g., predicting the 'close' price).
    """
    print("\nEvaluating POINT prediction performance...")
    model.eval()
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    all_predictions, all_actuals = [], []
    
    with torch.no_grad():
        for X_batch_test, y_batch_test in tqdm(test_loader, desc="Evaluating Test Set"):
            pred_batch = model(X_batch_test)
            
            # Reshape predictions to match actuals, same logic as in training loop
            if config.MODEL_TYPE == 'HSDGNN':
                B, _, N, _ = pred_batch.shape
                H = config.POINT_OUTPUT_WINDOW_SIZE
                D_out = y_batch_test.shape[-1]
                pred_batch = pred_batch.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
            elif config.MODEL_TYPE == 'GraphWaveNet':
                pred_batch = pred_batch[..., -1:]
            elif config.MODEL_TYPE == 'DSTAGNN':
                y_batch_test = y_batch_test.permute(0, 2, 1, 3).squeeze(-1)

            all_predictions.append(pred_batch.cpu().numpy())
            all_actuals.append(y_batch_test.cpu().numpy())

    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = np.concatenate(all_actuals, axis=0)

    # The scaler for point prediction is typically for a single column (e.g., 'close')
    # Ensure the key matches what is returned from the point prediction preprocessing module.
    scaler = scalers.get(config.TARGET_COLUMN)
    if scaler is None:
        raise ValueError(f"Scaler for target column '{config.TARGET_COLUMN}' not found in scalers dict.")

    # Inverse transform the data to its original scale
    # We need to handle both graph (3D) and non-graph (2D) shapes
    if predicted_scaled.ndim == 3: # Graph models (Batch, Nodes, Horizon)
        num_samples, num_nodes, horizon = predicted_scaled.shape
        # Flatten for inverse transform
        predicted_flat = predicted_scaled.reshape(-1, horizon)
        actual_flat = actual_scaled.reshape(-1, horizon)
        
        predicted_raw = scaler.inverse_transform(predicted_flat)
        actual_raw = scaler.inverse_transform(actual_flat)
        
        # Reshape back to original for easier analysis if needed
        predicted_raw = predicted_raw.reshape(num_samples, num_nodes, horizon)
        actual_raw = actual_raw.reshape(num_samples, num_nodes, horizon)
        
    else: # Non-graph models (Batch, Horizon)
        predicted_raw = scaler.inverse_transform(predicted_scaled)
        actual_raw = scaler.inverse_transform(actual_scaled)

    # Calculate metrics on the raw, unscaled data
    rmse = np.sqrt(mean_squared_error(actual_raw.flatten(), predicted_raw.flatten()))
    mae = mean_absolute_error(actual_raw.flatten(), predicted_raw.flatten())
    
    print(f"\n  --- Point Prediction Performance ---")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE:  {mae:.4f}")

    # Create a detailed results DataFrame
    if predicted_scaled.ndim == 3: # Graph models
        # Flatten everything for the DataFrame
        dates_flat = np.repeat(dates_test, num_nodes)
        stock_ids_flat = np.tile(stock_ids_test, len(dates_test))
        actual_df_flat = actual_raw.flatten()
        predicted_df_flat = predicted_raw.flatten()
    else: # Non-graph models
        dates_flat = dates_test
        stock_ids_flat = stock_ids_test
        actual_df_flat = actual_raw.flatten()
        predicted_df_flat = predicted_raw.flatten()

    results_df = pd.DataFrame({
        'Date': dates_flat,
        'StockID': stock_ids_flat,
        'Actual_Price': actual_df_flat,
        'Predicted_Price': predicted_df_flat
    })
    
    save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint evaluation results saved to '{save_path}'.")
# --- END OF IMPLEMENTATION ---


def run(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config, adj_matrix=None):
    print_header("Stage 5: Model Evaluation")
    print(f"Purpose: Assess final model performance for {config.PREDICTION_MODE} mode.")

    if config.PREDICTION_MODE == "TREND":
        _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config)
    elif config.PREDICTION_MODE == "POINT":
        _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config)
    else:
        raise ValueError(f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'.")

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN'] and adj_matrix is not None:
        # For graph models, stock_ids_test is the list of all nodes
        _calculate_graph_metrics(adj_matrix, stock_ids_test, config)