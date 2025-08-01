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
    """
    # This function remains unchanged.
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
    Evaluates the model for trend prediction. This function is assumed correct from
    previous steps and remains unchanged.
    """
    # This function remains unchanged.
    print("\nEvaluating TREND prediction performance...")
    model.eval()
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    all_predictions, all_actuals = [], []
    with torch.no_grad():
        for X_batch_test, y_batch_test in test_loader:
            pred_batch = model(X_batch_test)
            # --- START OF PROPOSED FIX ---
            # Reshape HSDGNN output to match the expected (B, N, D) format,
            # just like in the training loop in stage_4.
            if config.MODEL_TYPE == 'HSDGNN' and config.PREDICTION_MODE == 'TREND':
                # Raw output shape: (Batch, Features, Nodes, 1) -> e.g., (64, 2, 32, 1)
                # Squeeze and permute to: (Batch, Nodes, Features) -> e.g., (64, 32, 2)
                pred_batch = pred_batch.squeeze(-1).permute(0, 2, 1)
            # --- END OF PROPOSED FIX ---
            
            """
            
            RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does
            not match the length of the desired ordering of dimensions 
            i.e. input.dim() = 4 is not equal to len(dims) = 3
            """
            """"
            if config.MODEL_TYPE == 'GraphWaveNet':
                 pred_batch = pred_batch[..., -1:]
                 if config.PREDICTION_MODE == 'TREND':
                     pred_batch = pred_batch.squeeze(-1).permute(0, 2, 1)
            """

            if config.MODEL_TYPE == 'GraphWaveNet' and config.PREDICTION_MODE == 'TREND':
                 # The model output is (B, 1, N, 2). Squeeze the horizon dimension.
                 #pred_batch = pred_batch.squeeze(1)
                 pred_batch = pred_batch.squeeze(-1).permute(0, 2, 1)
            elif config.MODEL_TYPE == 'GraphWaveNet': # This handles POINT mode
                 pred_batch = pred_batch[..., -1:]
            
            all_predictions.append(pred_batch.cpu().numpy())
            all_actuals.append(y_batch_test.cpu().numpy())
    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = np.concatenate(all_actuals, axis=0)
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        """
        actual_slopes_scaled = actual_scaled[..., 0]
        actual_durations_scaled = actual_scaled[..., 1]
        predicted_slopes_scaled = predicted_scaled[..., 0]
        predicted_durations_scaled = predicted_scaled[..., 1]
        actual_raw_slopes = scalers['slope'].inverse_transform(actual_slopes_scaled.reshape(-1, 1)).flatten()
        actual_durations = scalers['duration'].inverse_transform(actual_durations_scaled.reshape(-1, 1)).flatten()
        predicted_raw_slopes = scalers['slope'].inverse_transform(predicted_slopes_scaled.reshape(-1, 1)).flatten()
        predicted_durations = scalers['duration'].inverse_transform(predicted_durations_scaled.reshape(-1, 1)).flatten()
        stock_ids_test_flat = np.tile(stock_ids_test, actual_scaled.shape[0])
        dates_test_flat = np.repeat(dates_test, len(stock_ids_test))
        """

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
    
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        # For graph models, tile and repeat to match the flattened prediction length.
        stock_ids_test_flat = np.tile(stock_ids_test, actual_scaled.shape[0])
        dates_test_flat = np.repeat(dates_test, len(stock_ids_test))
    else: 
        # For non-graph models, the arrays are already the correct length.
        stock_ids_test_flat = stock_ids_test
        dates_test_flat = dates_test
    
    
    
    results_df = pd.DataFrame({'Date': dates_test_flat, 'StockID': stock_ids_test_flat, 'Actual_Slope_Angle': actual_slope_angles, 'Predicted_Slope_Angle': predicted_slope_angles, 'Actual_Duration': actual_durations, 'Predicted_Duration': predicted_durations})
    
    results_df = pd.DataFrame({
        'Date': dates_test_flat, 
        'StockID': stock_ids_test_flat, 
        'Actual_Slope_Angle': actual_slope_angles, 
        'Predicted_Slope_Angle': predicted_slope_angles, 
        'Actual_Duration': actual_durations, 
        'Predicted_Duration': predicted_durations
    })
    
    
    #save_path = f"evaluation_results_TREND_{config.MODEL_TYPE}.csv"
    #save_path = f"evaluation_results_TREND_{config.MODEL_TYPE}_{config.TREND_INPUT_WINDOW_SIZE}.csv"
    save_path = (
        f"evaluation_results_TREND_{config.MODEL_TYPE}_"
        f"IN{config.TREND_INPUT_WINDOW_SIZE}_"
        f"ERR{config.MAX_SEGMENTATION_ERROR}.csv"
    )
    results_df.to_csv(save_path, index=False)
    print(f"\nTrend evaluation results saved to '{save_path}'.")


def _evaluate_point_prediction1(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config):
    print("\nEvaluating POINT prediction performance...")
    model.eval()

    # Use a DataLoader for efficient batching of test data
    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for X_batch_test in test_loader:
            pred_batch = model(X_batch_test[0])
            
            # Reshape model output to match target shape if necessary
            if config.MODEL_TYPE == 'HSDGNN':
                B, _, N, _ = pred_batch.shape
                H = config.POINT_OUTPUT_WINDOW_SIZE
                D_out = y_test_t.shape[-1]
                pred_batch = pred_batch.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
            elif config.MODEL_TYPE == 'GraphWaveNet':
                pred_batch = pred_batch[..., -1:]
            
            all_predictions.append(pred_batch.cpu().numpy())

    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = y_test_t.cpu().numpy()
    """
    # --- START OF FIX ---
    # This block correctly aligns dates, stock IDs, and prices for the final report.
    
    # Get dimensions from the data
    num_samples = actual_scaled.shape[0]
    horizon = config.POINT_OUTPUT_WINDOW_SIZE
    
    # Flatten the multi-dimensional actual and predicted arrays into long 1D lists of values.
    actual_flat = actual_scaled.reshape(-1, actual_scaled.shape[-1])
    predicted_flat = predicted_scaled.reshape(-1, predicted_scaled.shape[-1])

    # Create corresponding Date, StockID, and Horizon Step arrays that align with the flattened prices.
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        num_nodes = actual_scaled.shape[2]
        stock_ids_ordered = stock_ids_test  # This is the full list of nodes

        # Flatten the 2D dates_test array and repeat each date for every node.
        final_dates = np.repeat(dates_test.flatten(), num_nodes)
        
        # Tile the list of stock IDs to match the flattened data structure.
        final_stock_ids = np.tile(stock_ids_ordered, num_samples * horizon)
        
        # Create and tile the horizon steps array.
        horizon_steps_pattern = np.repeat(np.arange(1, horizon + 1), num_nodes)
        final_horizon_steps = np.tile(horizon_steps_pattern, num_samples)

    else:  # Logic for TCN, MLP
        # Flatten the 2D dates_test array.
        #final_dates = dates_test.flatten()
        
        # Repeat each stock ID for every step in its prediction horizon.
        #final_stock_ids = np.repeat(stock_ids_test, horizon)

        # Repeat each date for every step in the prediction horizon.
        final_dates = np.repeat(dates_test, horizon) # <--- FIX APPLIED HERE
        
        # Repeat each stock ID for every step in its prediction horizon.
        final_stock_ids = np.repeat(stock_ids_test, horizon)

        # Tile the horizon steps for each sample.
        final_horizon_steps = np.tile(np.arange(1, horizon + 1), num_samples)

    # Inverse transform the prices one by one, using the correct scaler for each stock.
    all_actual_prices, all_predicted_prices = [], []
    target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    num_features = len(config.FEATURE_COLUMNS)

    for i in tqdm(range(len(actual_flat)), desc="Inverse-transforming prices"):
        stock_id = final_stock_ids[i]
        scaler = scalers.get(stock_id)
        if scaler is None: continue

        # Create dummy arrays with the shape the scaler expects (num_features,)
        dummy_actual = np.zeros(num_features)
        dummy_predicted = np.zeros(num_features)

        # Place the scaled value in the correct target column position
        dummy_actual[target_col_idx] = actual_flat[i, 0]
        dummy_predicted[target_col_idx] = predicted_flat[i, 0]

        # Inverse transform and append the single price value
        all_actual_prices.append(scaler.inverse_transform(dummy_actual.reshape(1, -1))[0, target_col_idx])
        all_predicted_prices.append(scaler.inverse_transform(dummy_predicted.reshape(1, -1))[0, target_col_idx])

    actual_prices = np.array(all_actual_prices)
    predicted_prices = np.array(all_predicted_prices)
    """
    
    # Get dimensions from the data
    num_samples = actual_scaled.shape[0]
    horizon = config.POINT_OUTPUT_WINDOW_SIZE
    
    # Flatten the multi-dimensional actual and predicted arrays into long 1D lists of values.
    actual_flat = actual_scaled.reshape(-1, actual_scaled.shape[-1])
    predicted_flat = predicted_scaled.reshape(-1, predicted_scaled.shape[-1])

    # Create corresponding Date, StockID, and Horizon Step arrays that align with the flattened prices.
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        num_nodes = actual_scaled.shape[2]
        stock_ids_ordered = stock_ids_test  # This is the full list of nodes

        # Each date corresponds to a sample and must be repeated for each horizon step and each node.
        #final_dates = np.repeat(dates_test.flatten(), horizon * num_nodes)
        #Proposed fix
        final_dates = np.repeat(dates_test.flatten(), num_nodes)
        # The list of stocks must be tiled to match the flattened structure.
        final_stock_ids = np.tile(stock_ids_ordered, num_samples * horizon)
        
        # The horizon steps pattern must be created and tiled for each sample.
        horizon_steps_pattern = np.repeat(np.arange(1, horizon + 1), num_nodes)
        final_horizon_steps = np.tile(horizon_steps_pattern, num_samples)

    else:  # Logic for TCN, MLP
        # Defensively flatten per-sample arrays to ensure they are 1D.
        dates_1d = dates_test.flatten()
        stock_ids_1d = stock_ids_test.flatten()
        
        # Repeat each date for every step in its prediction horizon.
        final_dates = np.repeat(dates_1d, horizon)
        
        # Repeat each stock ID for every step in its prediction horizon.
        final_stock_ids = np.repeat(stock_ids_1d, horizon)
        
        # Tile the horizon steps for each sample.
        final_horizon_steps = np.tile(np.arange(1, horizon + 1), num_samples)

    # Inverse transform the prices one by one, using the correct scaler for each stock.
    all_actual_prices, all_predicted_prices = [], []
    target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    num_features = len(config.FEATURE_COLUMNS)

    for i in tqdm(range(len(actual_flat)), desc="Inverse-transforming prices"):
        stock_id = final_stock_ids[i]
        scaler = scalers.get(stock_id)
        if scaler is None: continue

        # Create dummy arrays with the shape the scaler expects (num_features,)
        dummy_actual = np.zeros(num_features)
        dummy_predicted = np.zeros(num_features)

        # Place the scaled value in the correct target column position
        dummy_actual[target_col_idx] = actual_flat[i, 0]
        dummy_predicted[target_col_idx] = predicted_flat[i, 0]

        # Inverse transform and append the single price value
        all_actual_prices.append(scaler.inverse_transform(dummy_actual.reshape(1, -1))[0, target_col_idx])
        all_predicted_prices.append(scaler.inverse_transform(dummy_predicted.reshape(1, -1))[0, target_col_idx])

    actual_prices = np.array(all_actual_prices)
    predicted_prices = np.array(all_predicted_prices)


    # --- END OF FIX ---

    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-8))) * 100

    print("\nQuantitative Metrics on Combined Test Set (All Stocks):")
    print(f"  - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # --- START OF FIX ---
    # Create the final DataFrame with the correctly aligned and sized arrays.
    results_df = pd.DataFrame({
        'Date': final_dates,
        'StockID': final_stock_ids,
        'Horizon_Step': final_horizon_steps,
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })
    # --- END OF FIX ---
    
    save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint prediction evaluation results saved to '{save_path}'.")

def _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config):
    print("\nEvaluating POINT prediction performance...")
    model.eval()

    # Use a DataLoader for efficient batching of test data
    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for X_batch_test in test_loader:
            pred_batch = model(X_batch_test[0])
            
            # Reshape model output to match target shape if necessary
            if config.MODEL_TYPE == 'HSDGNN':
                B, _, N, _ = pred_batch.shape
                H = config.POINT_OUTPUT_WINDOW_SIZE
                D_out = y_test_t.shape[-1]
                pred_batch = pred_batch.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
            elif config.MODEL_TYPE == 'GraphWaveNet':
                pred_batch = pred_batch[..., -1:]
            
            all_predictions.append(pred_batch.cpu().numpy())

    predicted_scaled = np.concatenate(all_predictions, axis=0)
    actual_scaled = y_test_t.cpu().numpy()
    
    # Get dimensions from the data
    num_samples = actual_scaled.shape[0]
    horizon = config.POINT_OUTPUT_WINDOW_SIZE
    
    # --- START OF FIX ---
    # Flatten the multi-dimensional actual and predicted arrays into long 1D lists of values.
    # Using .flatten() ensures we get a 1D array of length (num_samples * horizon).
    actual_flat = actual_scaled.flatten()
    predicted_flat = predicted_scaled.flatten()
    # --- END OF FIX ---

    # Create corresponding Date, StockID, and Horizon Step arrays that align with the flattened prices.
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        num_nodes = actual_scaled.shape[2]
        stock_ids_ordered = stock_ids_test
        final_dates = np.repeat(dates_test.flatten(), num_nodes)
        #final_dates = np.repeat(dates_test.flatten(), horizon * num_nodes)
        final_stock_ids = np.tile(stock_ids_ordered, num_samples * horizon)
        horizon_steps_pattern = np.repeat(np.arange(1, horizon + 1), num_nodes)
        final_horizon_steps = np.tile(horizon_steps_pattern, num_samples)

    else:  # Logic for TCN, MLP
        dates_1d = dates_test.flatten()
        stock_ids_1d = stock_ids_test.flatten()
        final_dates = dates_test.flatten()
        
        # The logic for stock IDs and horizon steps was already correct for this structure.
        final_stock_ids = np.repeat(stock_ids_test.flatten(), horizon)
        final_horizon_steps = np.tile(np.arange(1, horizon + 1), num_samples)

        #final_dates = np.repeat(dates_1d, horizon)
        #final_stock_ids = np.repeat(stock_ids_1d, horizon)
        #final_horizon_steps = np.tile(np.arange(1, horizon + 1), num_samples)

    # Inverse transform the prices one by one, using the correct scaler for each stock.
    all_actual_prices, all_predicted_prices = [], []
    target_col_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)
    num_features = len(config.FEATURE_COLUMNS)

    # The loop now correctly iterates (num_samples * horizon) times.
    for i in tqdm(range(len(actual_flat)), desc="Inverse-transforming prices"):
        stock_id = final_stock_ids[i]
        scaler = scalers.get(stock_id)
        if scaler is None: continue

        dummy_actual = np.zeros(num_features)
        dummy_predicted = np.zeros(num_features)

        # --- START OF FIX ---
        # Indexing is now 1D (e.g., actual_flat[i]) because the arrays are properly flattened.
        dummy_actual[target_col_idx] = actual_flat[i]
        dummy_predicted[target_col_idx] = predicted_flat[i]
        # --- END OF FIX ---

        all_actual_prices.append(scaler.inverse_transform(dummy_actual.reshape(1, -1))[0, target_col_idx])
        all_predicted_prices.append(scaler.inverse_transform(dummy_predicted.reshape(1, -1))[0, target_col_idx])

    actual_prices = np.array(all_actual_prices)
    predicted_prices = np.array(all_predicted_prices)

    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-8))) * 100

    print("\nQuantitative Metrics on Combined Test Set (All Stocks):")
    print(f"  - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    results_df = pd.DataFrame({
        'Date': final_dates,
        'StockID': final_stock_ids,
        'Horizon_Step': final_horizon_steps,
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })
    
    #save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}.csv"
    save_path = f"evaluation_results_POINT_{config.MODEL_TYPE}_{config.POINT_INPUT_WINDOW_SIZE}_{config.POINT_OUTPUT_WINDOW_SIZE}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nPoint prediction evaluation results saved to '{save_path}'.")

def run(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config, adj_matrix=None):
    print_header("Stage 5: Model Evaluation")
    print(f"Purpose: Assess final model performance for {config.PREDICTION_MODE} mode.")

    if config.PREDICTION_MODE == "TREND":
        _evaluate_trend_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config)
    elif config.PREDICTION_MODE == "POINT":
        _evaluate_point_prediction(model, X_test_t, y_test_t, stock_ids_test, dates_test, scalers, config)
    else:
        raise ValueError(f"Invalid PREDICTION_MODE '{config.PREDICTION_MODE}'.")

    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN'] and adj_matrix is not None:
        stock_ids_ordered = sorted(scalers.keys())
        _calculate_graph_metrics(adj_matrix, stock_ids_ordered, config)
        # This block specifically handles the dynamic adjacency matrix from HSDGNN.
    elif config.MODEL_TYPE == 'HSDGNN':
        print_header("Extracting Dynamic Adjacency Matrix from HSDGNN")
        print("Purpose: To analyze the graph structure learned by the HSDGNN model on the test set.")
        
        model.eval()
        with torch.no_grad():
            # Create a temporary DataLoader to get one batch of test data.
            # We only need the features (X) for the forward pass.
            test_dataset = TensorDataset(X_test_t)
            temp_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            X_batch_sample = next(iter(temp_loader))[0]
            
            # Call the model with return_adjs=True to get the learned matrix.
            # The first return value (predictions) is not needed here.
            _, dynamic_adj_batch = model(X_batch_sample, return_adjs=True)
            
            # The matrix shape is (Batch, Time, Nodes, Nodes). We'll take the matrix
            # from the last time step of the first sample in the batch as a representative example.
            representative_adj = dynamic_adj_batch[0, -1, :, :].cpu()
            
            print("Successfully extracted a representative adjacency matrix from the model.")
            
            # Now, call the graph metrics calculation function with the extracted matrix.
            stock_ids_ordered = sorted(scalers.keys())
            #_calculate_graph_metrics(representative_adj, stock_ids_ordered, config)