import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import json
import time

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.tcn_model import TCNStockPredictor
from utils.utils import load_config_from_path, setup_logging


def prepare_stock_data(data_path: str, 
                      sequence_length: int, 
                      target_column: str = 'Value',
                      feature_columns: list = None,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      logger=None) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:
    """
    Prepare stock data for TCN training
    
    Args:
        data_path: Path to the CSV data file
        sequence_length: Length of input sequences
        target_column: Column to predict
        feature_columns: List of feature columns to use
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        
    Returns:
        train_loader, val_loader, test_loader, feature_scaler, target_scaler
    """
    # Load data
    df = pd.read_csv(data_path)
    
    if logger:
        logger.info(f"Loaded {len(df)} rows from {data_path}")
        logger.info(f"Original first row: Date={df.iloc[0]['Date'] if 'Date' in df.columns else 'N/A'}, Value={df.iloc[0][target_column]}")
        logger.info(f"Original last row: Date={df.iloc[-1]['Date'] if 'Date' in df.columns else 'N/A'}, Value={df.iloc[-1][target_column]}")
    
    # Sort by date if Date column exists (oldest first for time series)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # Handle DD/MM/YYYY format
        original_order = df['Date'].iloc[0] > df['Date'].iloc[-1]  # True if reverse order
        df = df.sort_values('Date').reset_index(drop=True)
        
        if logger:
            logger.info(f"Data sorted chronologically. Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"Original order was: {'REVERSE (newest→oldest)' if original_order else 'CHRONOLOGICAL (oldest→newest)'}")
            logger.info(f"After sorting first row: Date={df.iloc[0]['Date']}, Value={df.iloc[0][target_column]}")
            logger.info(f"After sorting last row: Date={df.iloc[-1]['Date']}, Value={df.iloc[-1][target_column]}")
    
    # Use default feature columns if not specified
    if feature_columns is None:
        feature_columns = ['Value']
    
    # Ensure target column is in feature columns
    if target_column not in feature_columns:
        feature_columns.append(target_column)
    
    # Select relevant columns
    data = df[feature_columns].values
    target_data = df[target_column].values.reshape(-1, 1)
    
    # Normalize features and target separately
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(data)
    scaled_target = target_scaler.fit_transform(target_data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_target[i, 0])  # Predict next day's value
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    total_samples = len(X)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_scaler, target_scaler


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device) -> float:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model: nn.Module, 
                  val_loader: DataLoader, 
                  criterion: nn.Module, 
                  device: torch.device) -> float:
    """Validate model for one epoch"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(config: Dict[str, Any], 
          trial_name_for_temp_checkpoint: str = "tcn_training",
          trial=None) -> Tuple[Dict[str, float], str]:
    """
    Train TCN model
    
    Args:
        config: Configuration dictionary
        trial_name_for_temp_checkpoint: Name for temporary checkpoint
        trial: Optuna trial object (for HPO)
        
    Returns:
        Tuple of (final_metrics, checkpoint_path)
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract training parameters
    train_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    epochs = train_config.get('epochs', 100)
    learning_rate = float(train_config.get('learning_rate', 0.001))
    weight_decay = float(train_config.get('weight_decay', 1e-5))
    patience = train_config.get('patience', 10)
    
    # Setup logging
    logger = setup_logging(__name__)
    logger.info(f"Starting TCN training with trial: {trial_name_for_temp_checkpoint}")
    
    try:
        # Prepare data
        data_path = os.path.join(config.get('project_root_dir', ''), data_config.get('file_path', ''))
        
        train_loader, val_loader, test_loader, feature_scaler, target_scaler = prepare_stock_data(
            data_path=data_path,
            sequence_length=model_config.get('sequence_length', 60),
            target_column=data_config.get('target_column', 'Value'),
            feature_columns=data_config.get('feature_columns', ['Value']),
            train_ratio=data_config.get('train_ratio', 0.8),
            val_ratio=data_config.get('val_ratio', 0.1),
            logger=logger
        )
        
        # Update model config with actual input size
        model_config['input_size'] = len(data_config.get('feature_columns', ['Value']))
        
        # Initialize model
        model = TCNStockPredictor(model_config).to(device)
        
        # Setup loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        early_stop_counter = 0
        
        # Create temporary checkpoint path
        temp_checkpoint_dir = os.path.join(config.get('project_root_dir', ''), 'temp_checkpoints')
        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        temp_checkpoint_path = os.path.join(temp_checkpoint_dir, f"tcn_{trial_name_for_temp_checkpoint}.pt")
        
        logger.info(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'config': config,
                    'feature_scaler': feature_scaler,
                    'target_scaler': target_scaler,
                    'model_info': model.get_model_info()
                }
                torch.save(checkpoint, temp_checkpoint_path)
                logger.info(f"New best model saved at epoch {epoch}")
            else:
                early_stop_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s | "
                          f"Best: {best_val_loss:.6f} @ epoch {best_epoch+1}")
            
            # Early stopping
            if early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model for final evaluation
        if os.path.exists(temp_checkpoint_path):
            checkpoint = torch.load(temp_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Final evaluation on test set
        test_loss = validate_epoch(model, test_loader, criterion, device)
        
        # Calculate additional metrics
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
        
        # Inverse transform for real-scale metrics
        all_predictions_real = target_scaler.inverse_transform(all_predictions)
        all_targets_real = target_scaler.inverse_transform(all_targets)
        
        mse = mean_squared_error(all_targets_real, all_predictions_real)
        mae = mean_absolute_error(all_targets_real, all_predictions_real)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE
        mape = np.mean(np.abs((all_targets_real - all_predictions_real) / all_targets_real)) * 100
        
        final_metrics = {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1
        }
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Test metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        return final_metrics, temp_checkpoint_path
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e


if __name__ == "__main__":
    # For standalone testing
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/tcnconfig.yaml')
    args = parser.parse_args()
    
    config = load_config_from_path(args.config)
    config['project_root_dir'] = PROJECT_ROOT
    
    metrics, checkpoint_path = train(config)
    print(f"Training completed. Metrics: {metrics}")
    print(f"Model saved to: {checkpoint_path}")