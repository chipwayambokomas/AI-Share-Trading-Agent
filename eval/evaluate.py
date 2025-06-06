import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.utils import load_config_from_path, setup_logging
from train.trainTCN import prepare_stock_data


def load_model_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Dictionary containing model and related information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def detect_model_type(checkpoint: Dict) -> str:
    """
    Detect the type of model from checkpoint
    
    Args:
        checkpoint: Model checkpoint dictionary
        
    Returns:
        Model type string ('TCN' or 'MLP')
    """
    model_info = checkpoint.get('model_info', {})
    model_type = model_info.get('model_type', 'Unknown')
    
    if model_type != 'Unknown':
        return model_type
    
    # Try to infer from model state dict
    state_dict_keys = list(checkpoint.get('model_state_dict', {}).keys())
    
    # Check for TCN-specific layers
    tcn_indicators = ['tcn.network', 'network.', 'conv1d', 'temporal']
    if any(indicator in key.lower() for key in state_dict_keys for indicator in tcn_indicators):
        return 'TCN'
    
    # Check for MLP-specific layers
    mlp_indicators = ['linear', 'fc', 'layers.']
    if any(indicator in key.lower() for key in state_dict_keys for indicator in mlp_indicators):
        return 'MLP'
    
    return 'Unknown'


def load_tcn_model(checkpoint: Dict, device: torch.device):
    """Load TCN model from checkpoint"""
    from models.tcn_model import TCNStockPredictor
    
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    model = TCNStockPredictor(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('feature_scaler'), checkpoint.get('target_scaler')


def load_mlp_model(checkpoint: Dict, device: torch.device):
    """Load MLP model from checkpoint"""
    try:
        from models.mlp_model import MLPStockPredictor
        
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        model = MLPStockPredictor(model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint.get('feature_scaler'), checkpoint.get('target_scaler')
    except ImportError:
        raise ImportError("MLP model not available. Please ensure mlp_model.py exists.")


def prepare_data_for_model(data_path: str, model_type: str, config: Dict, scalers: Tuple = None):
    """
    Prepare data based on model type
    
    Args:
        data_path: Path to data file
        model_type: Type of model ('TCN' or 'MLP')
        config: Model configuration
        scalers: Tuple of (feature_scaler, target_scaler) if available
        
    Returns:
        Prepared data loaders and scalers
    """
    if model_type == 'TCN':
        return prepare_stock_data(
            data_path=data_path,
            sequence_length=config.get('model', {}).get('sequence_length', 60),
            target_column=config.get('data', {}).get('target_column', 'Close'),
            feature_columns=config.get('data', {}).get('feature_columns', ['Open', 'High', 'Low', 'Close', 'Volume']),
            train_ratio=config.get('data', {}).get('train_ratio', 0.8),
            val_ratio=config.get('data', {}).get('val_ratio', 0.1)
        )
    elif model_type == 'MLP':
        # Assuming similar structure for MLP data preparation
        try:
            from train.trainMLP import prepare_stock_data as prepare_mlp_data
            return prepare_mlp_data(
                data_path=data_path,
                sequence_length=config.get('model', {}).get('sequence_length', 60),
                target_column=config.get('data', {}).get('target_column', 'Close'),
                feature_columns=config.get('data', {}).get('feature_columns', ['Open', 'High', 'Low', 'Close', 'Volume']),
                train_ratio=config.get('data', {}).get('train_ratio', 0.8),
                val_ratio=config.get('data', {}).get('val_ratio', 0.1)
            )
        except ImportError:
            raise ImportError("MLP data preparation not available. Please ensure trainMLP.py exists.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate R-squared
    r2 = r2_score(y_true, y_pred)
    
    # # Calculate directional accuracy
    # y_true_direction = np.diff(y_true.flatten()) > 0
    # y_pred_direction = np.diff(y_pred.flatten()) > 0
    # directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        # 'directional_accuracy': float(directional_accuracy)
    }


def create_evaluation_plots(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          save_dir: str, 
                          model_type: str,
                          target_scaler=None) -> List[str]:
    """
    Create comprehensive evaluation plots
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_dir: Directory to save plots
        model_type: Type of model for plot titles
        target_scaler: Scaler for inverse transformation
        
    Returns:
        List of saved plot paths
    """
    os.makedirs(save_dir, exist_ok=True)
    plot_paths = []
    
    # Inverse transform if scaler available
    if target_scaler is not None:
        y_true_orig = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig = y_true.flatten()
        y_pred_orig = y_pred.flatten()
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Time Series Plot
    plt.figure(figsize=(15, 8))
    time_steps = np.arange(len(y_true_orig))
    
    plt.plot(time_steps, y_true_orig, label='Actual', color='blue', alpha=0.7, linewidth=1)
    plt.plot(time_steps, y_pred_orig, label='Predicted', color='red', alpha=0.7, linewidth=1)
    
    plt.title(f'{model_type} Model: Actual vs Predicted Stock Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'{model_type.lower()}_time_series.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot_path)
    
    # 2. Scatter Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'{model_type} Model: Predicted vs Actual Scatter Plot', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Stock Price', fontsize=12)
    plt.ylabel('Predicted Stock Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'{model_type.lower()}_scatter.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot_path)
    
    # 3. Residuals Plot
    residuals = y_pred_orig - y_true_orig
    
    plt.figure(figsize=(15, 6))
    
    # Residuals over time
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, residuals, color='purple', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title(f'{model_type} Model: Residuals Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Residuals histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title(f'{model_type} Model: Residuals Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'{model_type.lower()}_residuals.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot_path)
    
    # 4. Error Distribution
    plt.figure(figsize=(12, 8))
    
    # Absolute errors
    abs_errors = np.abs(residuals)
    
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, abs_errors, color='orange', alpha=0.7)
    plt.title('Absolute Errors Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(abs_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Absolute Errors Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Percentage errors
    pct_errors = (residuals / y_true_orig) * 100
    
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, pct_errors, color='green', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Percentage Errors Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Percentage Error (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(pct_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Percentage Errors Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_type} Model: Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'{model_type.lower()}_error_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(plot_path)
    
    return plot_paths


def evaluate_trained_model(checkpoint_path: str, project_root_dir: str):
    """
    Evaluate a trained model (TCN or MLP)
    
    Args:
        checkpoint_path: Path to the model checkpoint
        project_root_dir: Project root directory
    """
    # Setup logging
    logger = setup_logging(__name__)
    logger.info(f"Starting model evaluation for checkpoint: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load checkpoint
        checkpoint = load_model_checkpoint(checkpoint_path, device)
        
        # Detect model type
        model_type = detect_model_type(checkpoint)
        logger.info(f"Detected model type: {model_type}")
        
        if model_type == 'Unknown':
            raise ValueError("Could not determine model type from checkpoint")
        
        # Load model and scalers
        if model_type == 'TCN':
            model, feature_scaler, target_scaler = load_tcn_model(checkpoint, device)
        elif model_type == 'MLP':
            model, feature_scaler, target_scaler = load_mlp_model(checkpoint, device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Get configuration
        config = checkpoint.get('config', {})
        
        # Prepare data
        data_path = os.path.join(project_root_dir, config.get('data', {}).get('file_path', ''))
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        _, _, test_loader, _, _ = prepare_data_for_model(data_path, model_type, config)
        
        # Make predictions
        logger.info("Making predictions on test set...")
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics on normalized data
        normalized_metrics = calculate_metrics(all_targets, all_predictions)
        
        # Calculate metrics on original scale if scalers available
        if target_scaler is not None:
            all_predictions_orig = target_scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
            all_targets_orig = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
            original_metrics = calculate_metrics(all_targets_orig.reshape(-1, 1), all_predictions_orig.reshape(-1, 1))
        else:
            original_metrics = normalized_metrics
        
        # Create evaluation plots
        results_dir = os.path.join(project_root_dir, "results", "evaluation_plots")
        plot_paths = create_evaluation_plots(
            all_targets, all_predictions, results_dir, model_type, target_scaler
        )
        
        # Save predictions
        predictions_dir = os.path.join(project_root_dir, "results")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Determine model type from checkpoint filename for consistent naming
        checkpoint_filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        # Create consistent naming like MLP
        if "hpo" in checkpoint_filename.lower():
            base_name = f"{model_type}_best_hpo_model"
        elif "standalone" in checkpoint_filename.lower():
            base_name = f"{model_type}_best_model_standalone"
        else:
            base_name = f"{model_type}_model"
        
        # Save metrics in .txt format (matching MLP format)
        metrics_file = os.path.join(predictions_dir, f"{base_name}_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"{model_type} Model Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            for metric, value in original_metrics.items():
                if metric == 'directional_accuracy':
                    f.write(f"{metric.upper()}: {value:.2f}%\n")
                elif metric == 'mape':
                    f.write(f"{metric.upper()}: {value:.2f}%\n")
                elif metric == 'r2':
                    f.write(f"R²: {value:.4f}\n")
                else:
                    f.write(f"{metric.upper()}: {value:.4f}\n")
        
        # Save predictions in .csv format (matching MLP format)
        predictions_file = os.path.join(predictions_dir, f"{base_name}_predictions.csv")
        predictions_df = pd.DataFrame({
            'Actual_Normalized': all_targets.flatten(),
            'Predicted_Normalized': all_predictions.flatten()
        })
        
        if target_scaler is not None:
            # Preserve original precision - no rounding
            predictions_df['Actual_Stock_Price'] = all_targets_orig
            predictions_df['Predicted_Stock_Price'] = all_predictions_orig
        
        predictions_df.to_csv(predictions_file, index=False, float_format='%.4f')
        
        # Save evaluation results (comprehensive JSON for reference)
        evaluation_results = {
            'model_type': model_type,
            'checkpoint_path': checkpoint_path,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': {
                'normalized': normalized_metrics,
                'original_scale': original_metrics
            },
            'model_info': checkpoint.get('model_info', {}),
            'data_info': {
                'test_samples': len(all_targets),
                'data_path': data_path
            },
            'plots': plot_paths,
            'metrics_file': metrics_file,
            'predictions_file': predictions_file
        }
        
        results_path = os.path.join(project_root_dir, "results", f"{base_name}_evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print(f"{model_type} MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Model Type: {model_type}")
        print(f"Test Samples: {len(all_targets):,}")
        print()
        print("ORIGINAL SCALE METRICS:")
        for metric, value in original_metrics.items():
            if metric == 'directional_accuracy':
                print(f"{metric.upper()}: {value:.2f}%")
            elif metric == 'mape':
                print(f"{metric.upper()}: {value:.2f}%")
            elif metric == 'r2':
                print(f"R²: {value:.4f}")
            else:
                print(f"{metric.upper()}: {value:.4f}")
        print()
        print(f"Evaluation results saved:")
        print(f"- Metrics: {metrics_file}")
        print(f"- Predictions: {predictions_file}")
        print(f"- Plots: {results_dir}")
        print(f"- Full results: {results_path}")
        print("="*60)
        
        logger.info(f"Evaluation completed successfully for {model_type} model")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise e


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--project_root', type=str, default=PROJECT_ROOT,
                       help='Project root directory')
    
    args = parser.parse_args()
    
    evaluate_trained_model(args.checkpoint, args.project_root)