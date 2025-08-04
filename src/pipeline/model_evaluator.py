import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx
from ..utils import print_header
from ..models.base_handler import BaseModelHandler
import community as community_louvain

def _evaluate_point_prediction1(model, X_test_t, y_test_t, test_stock_ids, test_dates,scalers, handler, settings):
    print("\nEvaluating POINT prediction performance...")
    model.eval()

    # Create a DataLoader for the test set for batching
    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    all_predictions = []
    #disable gradient calculation for inference
    with torch.no_grad():
        # Iterate through the test data in batches ->get the predictions per batch -> adjust the output shape if needed -> move predictions to CPU for further processing
        for X_batch_test in test_loader:
            pred_batch_raw = model(X_batch_test[0])
            pred_batch, _ = handler.adapt_output_for_loss(pred_batch_raw, None)
            all_predictions.append(pred_batch.cpu().numpy())

    # Concatenate all predictions into a single array
    predicted_scaled = np.concatenate(all_predictions, axis=0)
    # Move test targets to CPU for further processing
    actual_scaled = y_test_t.cpu().numpy()

    # Inverse transform predictions to get actual prices
    if handler.is_graph_based():
        # Shape: (samples, nodes) or (samples, nodes, out_window) -> squeeze
        if actual_scaled.ndim > 2: actual_scaled = actual_scaled.squeeze()
        if predicted_scaled.ndim > 2: predicted_scaled = predicted_scaled.squeeze()
        
        # Reshape to (samples, num_nodes) for inverse transformation
        num_samples, num_nodes = actual_scaled.shape
        actual_flat = actual_scaled.reshape(-1)
        predicted_flat = predicted_scaled.reshape(-1)
        # Tile stock IDs to match the flattened predictions -> this basically repeats the stock IDs for each sample in the flattened array
        stock_ids_tiled = np.tile(test_stock_ids, num_samples) # test_stock_ids is the ordered list
        # The test_dates array has one date per sample. We need to repeat each date
        # for every node (stock) in that sample to match the other arrays.
        dates_tiled = np.repeat(test_dates, num_nodes)
        
        actual_prices, predicted_prices = [], []
        target_col_idx = settings.FEATURE_COLUMNS.index(settings.TARGET_COLUMN)
        num_features = len(settings.FEATURE_COLUMNS)

        # Iterate through each stock ID and apply the scaler
        for i in range(len(actual_flat)):
            # Get the stock ID and corresponding scaler
            stock_id = stock_ids_tiled[i]
            scaler = scalers[stock_id]
            # Create dummy arrays to hold the actual and predicted values
            dummy_actual = np.zeros((1, num_features))
            dummy_predicted = np.zeros((1, num_features))
            # Set the target column to the actual and predicted values
            dummy_actual[0, target_col_idx] = actual_flat[i]
            dummy_predicted[0, target_col_idx] = predicted_flat[i]
            
            # Inverse transform the dummy arrays to get actual prices and appedd to the lists -> goal is to get a list of actual and predicted prices for each stock ID
            actual_prices.append(scaler.inverse_transform(dummy_actual)[0, target_col_idx])
            predicted_prices.append(scaler.inverse_transform(dummy_predicted)[0, target_col_idx])

    else: # For TCN/MLP
        actual_prices, predicted_prices = [], []
        target_col_idx = settings.FEATURE_COLUMNS.index(settings.TARGET_COLUMN)
        num_features = len(settings.FEATURE_COLUMNS)

        for i in range(len(test_stock_ids)):
            stock_id = test_stock_ids[i]
            scaler = scalers[stock_id]
            
            dummy_actual = np.zeros((1, num_features))
            dummy_predicted = np.zeros((1, num_features))
            dummy_actual[0, target_col_idx] = actual_scaled[i]
            dummy_predicted[0, target_col_idx] = predicted_scaled[i]
            
            actual_prices.append(scaler.inverse_transform(dummy_actual)[0, target_col_idx])
            predicted_prices.append(scaler.inverse_transform(dummy_predicted)[0, target_col_idx])
        stock_ids_tiled = test_stock_ids # For DataFrame consistency
        dates_tiled = test_dates # For DataFrame consistency

    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)

    # Calculate metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-8))) * 100

    print("\nQuantitative Metrics (All Stocks):")
    print(f"  - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Save results
    results_df = pd.DataFrame({
        'Date': dates_tiled,
        'StockID': stock_ids_tiled,
        'Actual_Price': actual_prices, 
        'Predicted_Price': predicted_prices
    })
    
    results_df.sort_values(by=['Date', 'StockID'], inplace=True)
    
    save_path = f"{settings.RESULTS_DIR}/{settings.MODEL_TYPE}/evaluation_POINT_{settings.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint prediction evaluation results saved to '{save_path}'.")

def _evaluate_point_prediction(model, X_test_t, y_test_t, test_stock_ids, test_dates,scalers, handler, settings):
    print("\nEvaluating POINT prediction performance...")
    model.eval()

    # Create a DataLoader for the test set for batching
    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    all_predictions = []
    #disable gradient calculation for inference
    with torch.no_grad():
        # Iterate through the test data in batches ->get the predictions per batch -> adjust the output shape if needed -> move predictions to CPU for further processing
        for X_batch_test in test_loader:
            pred_batch_raw = model(X_batch_test[0])
            pred_batch, _ = handler.adapt_output_for_loss(pred_batch_raw, None)
            all_predictions.append(pred_batch.cpu().numpy())

    # Concatenate all predictions into a single array
    predicted_scaled = np.concatenate(all_predictions, axis=0)
    # Move test targets to CPU for further processing
    actual_scaled = y_test_t.cpu().numpy()

    # --- START OF CHANGE ---
    
    # Get dimensions for creating the final report
    num_samples = actual_scaled.shape[0]
    horizon = settings.POINT_OUTPUT_WINDOW_SIZE

    # Flatten the multi-dimensional actual and predicted arrays into long 1D lists of values.
    # This creates a single list of all individual predictions across all samples and horizons.
    actual_flat = actual_scaled.flatten()
    predicted_flat = predicted_scaled.flatten()

    # Create corresponding Date, StockID, and Horizon Step arrays that align with the flattened prices.
    if handler.is_graph_based():
        num_nodes = actual_scaled.shape[2]
        stock_ids_ordered = test_stock_ids  # This is the full list of nodes

        # Each date corresponds to a sample and must be repeated for each node.
        dates_tiled = np.repeat(test_dates.flatten(), num_nodes)
        
        # The list of stocks must be tiled to match the flattened structure.
        stock_ids_tiled = np.tile(stock_ids_ordered, num_samples * horizon)
        
        # The horizon steps pattern must be created and tiled for each sample.
        horizon_steps_pattern = np.repeat(np.arange(1, horizon + 1), num_nodes)
        horizon_steps = np.tile(horizon_steps_pattern, num_samples)

    else:  # Logic for TCN, MLP
        # The dates array is already (num_samples, horizon), so just flatten it.
        dates_tiled = test_dates.flatten()
        
        # Repeat each stock ID for every step in its prediction horizon.
        stock_ids_tiled = np.repeat(test_stock_ids, horizon)
        
        # Tile the horizon steps for each sample.
        horizon_steps = np.tile(np.arange(1, horizon + 1), num_samples)

    # Inverse transform the prices one by one, using the correct scaler for each stock.
    actual_prices, predicted_prices = [], []
    target_col_idx = settings.FEATURE_COLUMNS.index(settings.TARGET_COLUMN)
    num_features = len(settings.FEATURE_COLUMNS)

    # The loop now correctly iterates over every single prediction point.
    for i in range(len(actual_flat)):
        stock_id = stock_ids_tiled[i]
        scaler = scalers.get(stock_id)
        if scaler is None: continue

        # Create dummy arrays with the shape the scaler expects (num_features,)
        dummy_actual = np.zeros(num_features)
        dummy_predicted = np.zeros(num_features)

        # Place the scaled scalar value in the correct target column position
        dummy_actual[target_col_idx] = actual_flat[i]
        dummy_predicted[target_col_idx] = predicted_flat[i]

        # Inverse transform and append the single price value
        actual_prices.append(scaler.inverse_transform(dummy_actual.reshape(1, -1))[0, target_col_idx])
        predicted_prices.append(scaler.inverse_transform(dummy_predicted.reshape(1, -1))[0, target_col_idx])

    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)

    # Calculate metrics over all prediction points
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-8))) * 100

    print("\nQuantitative Metrics (All Stocks, All Horizon Steps):")
    print(f"  - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Save results to a detailed DataFrame
    results_df = pd.DataFrame({
        'Date': dates_tiled,
        'StockID': stock_ids_tiled,
        'Horizon_Step': horizon_steps,
        'Actual_Price': actual_prices, 
        'Predicted_Price': predicted_prices
    })
    
    results_df.sort_values(by=['Date', 'StockID', 'Horizon_Step'], inplace=True)
    
    save_path = f"{settings.RESULTS_DIR}/{settings.MODEL_TYPE}/evaluation_POINT_{settings.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint prediction evaluation results saved to '{save_path}'.")
    # --- END OF CHANGE ---

def _evaluate_trend_prediction(model, X_test_t, y_test_t, test_stock_ids, scalers, handler, settings):
    print("\nEvaluating TREND prediction performance...")
    model.eval()

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    all_predictions, all_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred_batch_raw = model(X_batch)
            pred_batch, _ = handler.adapt_output_for_loss(pred_batch_raw, y_batch)
            all_predictions.append(pred_batch.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())
    
    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = np.concatenate(all_actuals, axis=0)

    # Inverse transform slopes and durations
    if handler.is_graph_based():
        # for both actual and predicted, take the first two features (slope and duration) and flatten them -> we know that the first two features are slope and duration
        actual_slopes = scalers['slope'].inverse_transform(actual_scaled[..., 0]).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_scaled[..., 1]).flatten()
        predicted_slopes = scalers['slope'].inverse_transform(predicted_scaled[..., 0]).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[..., 1]).flatten()
        # repeat the stock IDs for each sample in the flattened array
        stock_ids_flat = np.tile(test_stock_ids, actual_scaled.shape[0])
    else: # For TCN/MLP
        #same process as above, but we know that the first two features are slope and duration ->we just take the first two features and flatten them because this is a 2d array
        actual_slopes = scalers['slope'].inverse_transform(actual_scaled[:, 0].reshape(-1, 1)).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_scaled[:, 1].reshape(-1, 1)).flatten()
        predicted_slopes = scalers['slope'].inverse_transform(predicted_scaled[:, 0].reshape(-1, 1)).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_scaled[:, 1].reshape(-1, 1)).flatten()
        stock_ids_flat = test_stock_ids
        
    # Slope metrics (in degrees)
    actual_angles = np.degrees(np.arctan(actual_slopes))
    predicted_angles = np.degrees(np.arctan(predicted_slopes))
    rmse_angle = np.sqrt(mean_squared_error(actual_angles, predicted_angles))
    mae_angle = mean_absolute_error(actual_angles, predicted_angles)
    print(f"\n  --- Slope Angle Prediction (degrees) ---")
    print(f"  - RMSE: {rmse_angle:.4f}, MAE: {mae_angle:.4f}")

    # Duration metrics
    rmse_duration = np.sqrt(mean_squared_error(actual_durations, predicted_durations))
    mae_duration = mean_absolute_error(actual_durations, predicted_durations)
    print(f"\n  --- Duration Prediction (days) ---")
    print(f"  - RMSE: {rmse_duration:.2f}, MAE: {mae_duration:.2f}")

    # Save results
    results_df = pd.DataFrame({
        'StockID': stock_ids_flat,
        'Actual_Slope_Angle': actual_angles,
        'Predicted_Slope_Angle': predicted_angles,
        'Actual_Duration': actual_durations,
        'Predicted_Duration': predicted_durations
    })
    save_path = f"{settings.RESULTS_DIR}/{settings.MODEL_TYPE}/evaluation_TREND_{settings.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nTrend evaluation results saved to '{save_path}'.")

def _calculate_graph_metrics(adj_matrix, stock_ids, settings, percentile_threshold):
    """
    Calculates and prints graph-based network metrics after pruning weak edges.
    
    Args:
        adj_matrix (torch.Tensor): The learned (N, N) adjacency matrix.
        stock_ids (list): List of stock names/IDs for labeling.
        settings: Your settings object (contains RESULT_DIR, MODEL_TYPE, etc.).
        percentile_threshold (int): Percentile to determine pruning threshold for edge weights.
    """
    print_header("Network Analysis Metrics")
    
    # Convert adjacency matrix from PyTorch tensor to NumPy array
    if hasattr(adj_matrix, 'cpu'):
        adj_numpy = adj_matrix.cpu().numpy()
    else:
        adj_numpy = adj_matrix  # Already a numpy array
    
    # Compute the weight threshold: all values below this percentile will be removed
    threshold = np.percentile(adj_numpy, percentile_threshold)
    print(f"Pruning graph: Keeping top {100 - percentile_threshold}% of edges.")
    print(f"Calculated threshold value: {threshold:.6f}")

    # Make a copy of the matrix and zero out all values below the threshold
    pruned_adj_numpy = np.copy(adj_numpy)
    pruned_adj_numpy[pruned_adj_numpy < 0.1] = 0

    # Build a directed graph (DiGraph) from the pruned adjacency matrix
    G = nx.from_numpy_array(pruned_adj_numpy, create_using=nx.DiGraph)
    
    # Count edges before and after pruning
    num_edges_before = np.count_nonzero(adj_numpy)
    num_edges_after = G.number_of_edges()
    print(f"Pruning reduced edges from {num_edges_before} to {num_edges_after}")

    # If pruning removed all edges, exit early
    if num_edges_after == 0:
        print("WARNING: Threshold is too high, no edges remain. Try a lower percentile (e.g., 80 or 70).")
        return

    # Relabel node indices with stock names
    mapping = {i: name for i, name in enumerate(stock_ids)}
    nx.relabel_nodes(G, mapping, copy=False)
    
    # Calculate various centrality and graph metrics
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Try to calculate eigenvector centrality; fallback to PageRank if it fails
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=5000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        print("WARNING: Eigenvector centrality failed. Using PageRank instead.")
        eigenvector_centrality = nx.pagerank(G)

    # Compute local clustering coefficients
    local_clustering_coeff = nx.clustering(G)

    # Detect communities (on undirected version of the graph)
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)

    # Create a DataFrame to store all calculated metrics for each stock
    analysis_df = pd.DataFrame(index=G.nodes())
    analysis_df['in_degree_centrality'] = analysis_df.index.map(in_degree_centrality)
    analysis_df['out_degree_centrality'] = analysis_df.index.map(out_degree_centrality)
    analysis_df['betweenness_centrality'] = analysis_df.index.map(betweenness_centrality)
    analysis_df['eigenvector_centrality_or_pagerank'] = analysis_df.index.map(eigenvector_centrality)
    analysis_df['local_clustering_coeff'] = analysis_df.index.map(local_clustering_coeff)
    analysis_df['community_id'] = analysis_df.index.map(partition)

    # Print the DataFrame to console
    print("\n--- Network Metrics per Stock (on Pruned Graph) ---")
    print(analysis_df)

    # Save the results to a CSV file
    save_path = f"{settings.RESULTS_DIR}/{settings.MODEL_TYPE}/evaluation_GRAPH_{settings.PREDICTION_MODE}__{settings.MODEL_TYPE}.csv"
    analysis_df.to_csv(save_path, index=True)
    print(f"\nGRAPH evaluation results saved to '{save_path}'.")

def run(model, X_test_t, y_test_t, test_stock_ids, test_dates,scalers, handler: BaseModelHandler, settings, adj_matrix=None):
    """
    Stage 5: Assesses final model performance and calculates network metrics if applicable.
    """
    print_header("Stage 5: Model Evaluation")
    
    if settings.PREDICTION_MODE == "POINT":
        _evaluate_point_prediction(model, X_test_t, y_test_t, test_stock_ids, test_dates,scalers, handler, settings)
    elif settings.PREDICTION_MODE == "TREND":
        _evaluate_trend_prediction(model, X_test_t, y_test_t, test_stock_ids, scalers, handler, settings)
          
    if handler.is_graph_based() and adj_matrix is not None:
            _calculate_graph_metrics(adj_matrix, test_stock_ids,settings,settings.EVAL_THRESHOLD)