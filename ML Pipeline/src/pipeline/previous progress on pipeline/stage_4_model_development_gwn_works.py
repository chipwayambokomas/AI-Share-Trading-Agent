# src/pipeline/stage_4_model_development.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.utils import print_header
from src import models
from tqdm import tqdm

def _trend_loss_function(predicted, actual, mse_criterion):
    """Calculates a combined MSE loss for non-graph trend prediction."""
    loss_slope = mse_criterion(predicted[:, 0], actual[:, 0])
    loss_duration = mse_criterion(predicted[:, 1], actual[:, 1])
    return (loss_slope + loss_duration) / 2.0

def run(train_loader, val_loader, config, supports=None):
    print_header("Stage 4: Model Development")
    print(f"""
Purpose: Train a model for {config.PREDICTION_MODE} prediction.
- Model Architecture: {config.MODEL_TYPE}
Executing Code: Initializing and training the {config.MODEL_TYPE} model...
    """)

    model_args = config.MODEL_ARGS[config.MODEL_TYPE].copy()
    
    # --- Determine input/output sizes and base loss function ---
    if config.PREDICTION_MODE == "TREND":
        input_features = 2 # slope, duration
        output_size = 2
        base_loss_fn = lambda pred, act: _trend_loss_function(pred, act, nn.MSELoss())
    else: # "POINT" mode
        input_features = len(config.FEATURE_COLUMNS)
        output_size = config.POINT_OUTPUT_WINDOW_SIZE
        base_loss_fn = nn.MSELoss()

    # --- Model Initialization ---
    model = None
    loss_fn = base_loss_fn # Default loss function

    if config.MODEL_TYPE == "GraphWaveNet":
        if supports is None:
            raise ValueError("GraphWaveNet model requires an adjacency matrix (supports).")
        
        model_args['num_nodes'] = supports[0].shape[0]
        model_args['supports'] = supports
        model_args['in_dim'] = input_features
        model_args['out_dim'] = output_size

        model_args.pop('output_size', None)
        model_args.pop('input_channels', None)
        model_args.pop('num_channels', None)
        
        model = models.GraphWaveNet(**model_args)

        # Define specific loss functions for GraphWaveNet output shapes
        if config.PREDICTION_MODE == "TREND":
            def graph_trend_loss(y_pred, y_true):
                y_pred_permuted = y_pred.permute(0, 2, 1, 3).squeeze(-1)
                return F.mse_loss(y_pred_permuted, y_true)
            loss_fn = graph_trend_loss
        else: # POINT mode for GraphWaveNet
            loss_fn = nn.MSELoss()

    elif config.MODEL_TYPE == "TCN":
        model_args['input_channels'] = input_features
        model_args['output_size'] = output_size
        model = models.TCN_Forecaster(**model_args)

    elif config.MODEL_TYPE == "MLP":
        input_window = config.POINT_INPUT_WINDOW_SIZE if config.PREDICTION_MODE == "POINT" else config.TREND_INPUT_WINDOW_SIZE
        input_size = input_window * input_features
        model_args['input_size'] = input_size
        model_args['output_size'] = output_size
        model_args.pop('input_channels', None) 
        model = models.MLP_Forecaster(**model_args)
        
    else:
        raise ValueError(f"Model type '{config.MODEL_TYPE}' not recognized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("\nStarting model training...")
    start_time = time.time()
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1:2d}/{config.EPOCHS}") as tepoch:
            for X_batch, y_batch in tepoch:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                
                # --- START OF FIX ---
                # Slicing logic is only needed for POINT mode with GraphWaveNet
                if config.MODEL_TYPE == 'GraphWaveNet' and config.PREDICTION_MODE == 'POINT':
                    y_pred = y_pred[..., -y_batch.size(3):]
                # --- END OF FIX ---

                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                
                # --- START OF FIX ---
                # Apply the same conditional slicing to the validation loop
                if config.MODEL_TYPE == 'GraphWaveNet' and config.PREDICTION_MODE == 'POINT':
                    y_pred = y_pred[..., -y_batch.size(3):]
                # --- END OF FIX ---
                
                val_loss += loss_fn(y_pred, y_batch).item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nModel training complete in {training_time:.2f} seconds.")
    return model, training_time
