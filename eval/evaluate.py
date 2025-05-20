import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_model(checkpoint_path, device):
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model onto
        
    Returns:
        model: The loaded PyTorch model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {})
    input_size = config.get("model", {}).get("input_size", 1)
    hidden_size = config.get("model", {}).get("hidden_size", 64)
    output_size = config.get("model", {}).get("output_size", 1)
    
    # Import here to avoid circular imports
    from models.mlp_model import MLPModel
    
    model = MLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, config

def load_test_data(config):
    """
    Load only the test data needed for evaluation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        test_loader: DataLoader for test data
        scaler: Data scaler for inverse transformation
    """
    # Import here to avoid circular imports
    from data.data import load_data
    
    # We only need the test data, but load_data returns all data splits
    _, _, test_loader, scaler = load_data(config)
    
    # Log the dtype of the first batch for debugging
    for inputs, targets in test_loader:
        print(f"Test data dtype: {inputs.dtype}, shape: {inputs.shape}")
        break
        
    return test_loader, scaler

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true (np.ndarray): Array of true values
        y_pred (np.ndarray): Array of predicted values
        
    Returns:
        float: MAPE value
    """
    # Avoid division by zero
    mask = y_true != 0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def evaluate_model(model, test_loader, scaler, device):
    """
    Evaluate the model on test data and calculate metrics.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        scaler: Scaler to transform predictions back to original scale
        device: Device to run evaluation on
        
    Returns:
        tuple: (y_true, y_pred, metrics_dict)
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    # Identify the model's expected data type from its parameters
    model_dtype = next(model.parameters()).dtype
    print(f"Using model dtype for evaluation: {model_dtype}")
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Convert inputs to the same dtype as the model
            inputs = inputs.to(device).to(model_dtype)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    if not all_targets or not all_predictions:
        print("Warning: No data to evaluate!")
        return np.array([]), np.array([]), {"error": "No data to evaluate"}
    
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    
    print(f"Raw predictions shape: {y_pred.shape}, dtype: {y_pred.dtype}")
    print(f"Raw targets shape: {y_true.shape}, dtype: {y_true.dtype}")
    
    # Inverse transform using the scaler
    if scaler:
        # Reshape for inverse transformation (scaler expects 2D)
        y_true_2d = y_true.reshape(-1, 1)
        y_pred_2d = y_pred.reshape(-1, 1)
        
        # Ensure we have the inverse_transform method
        if hasattr(scaler, 'inverse_transform'):
            print("Applying inverse scaling to convert back to original values")
            y_true = scaler.inverse_transform(y_true_2d).flatten()
            y_pred = scaler.inverse_transform(y_pred_2d).flatten()
        else:
            print("Warning: Scaler doesn't have inverse_transform method")
    else:
        print("No scaler provided, using raw values")
    
    print(f"After inverse transform - predictions shape: {y_pred.shape}, targets shape: {y_true.shape}")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }
    
    return y_true, y_pred, metrics

def save_results(y_true, y_pred, metrics, results_dir, model_name):
    """
    Save evaluation results to files.
    
    Args:
        y_true (np.ndarray): Array of true values
        y_pred (np.ndarray): Array of predicted values
        metrics (dict): Dictionary of evaluation metrics
        results_dir (str): Directory to save results
        model_name (str): Name of the model
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to a text file
    metrics_file = os.path.join(results_dir, f"{model_name}_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    # Save predictions and true values to a CSV file
    predictions_file = os.path.join(results_dir, f"{model_name}_predictions.csv")
    
    # Reshape if needed
    y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    
    df = pd.DataFrame({
        'Actual': y_true_flat,
        'Predicted': y_pred_flat
    })
    df.to_csv(predictions_file, index=False)
    
    print(f"Results saved to {results_dir}")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Predictions: {predictions_file}")

def evaluate_trained_model(checkpoint_path, project_root_dir):
    """
    Main function to evaluate a trained model.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        project_root_dir (str): Path to the project root directory
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load the model
        model, config = load_model(checkpoint_path, device)
        
        # Ensure project_root_dir is in config
        if 'project_root_dir' not in config:
            config['project_root_dir'] = project_root_dir
        
        # Load test data
        test_loader, scaler = load_test_data(config)
        
        if len(test_loader.dataset) == 0:
            print(f"Test dataset is empty. Cannot evaluate.")
            return None
            
        # Evaluate the model
        y_true, y_pred, metrics = evaluate_model(model, test_loader, scaler, device)
        
        if len(y_true) == 0 or "error" in metrics:
            print("Evaluation failed: No data processed")
            return None
        
        # Print results to console
        print("\n========== Evaluation Results ==========")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
        
        # Save results to files
        results_dir = os.path.join(project_root_dir, "results")
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        save_results(y_true, y_pred, metrics, results_dir, model_name)
        
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None