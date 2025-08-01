# src/pipeline/stage_4_model_development.py

import torch
import torch.nn as nn
import time
from src.utils import print_header
from src import models
from tqdm import tqdm
from argparse import Namespace # Import Namespace for HSDGNN's constructor

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
    
    # --- Determine input/output sizes from a sample batch ---
    X_sample, y_sample = next(iter(train_loader))
    
    if config.PREDICTION_MODE == "TREND":
        input_features = 2 # slope, duration
        output_size = 2
        base_loss_fn = lambda pred, act: _trend_loss_function(pred, act, nn.MSELoss())
    else: # "POINT" mode
        # For graph models, features are at the last dimension
        if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
            input_features = X_sample.shape[3]
        else: # For TCN/MLP, features are at dimension 2
            input_features = X_sample.shape[2]
        output_size = config.POINT_OUTPUT_WINDOW_SIZE
        base_loss_fn = nn.MSELoss()
        #base_loss_fn = nn.HuberLoss() #Attempt to reduce MSE 

    # --- Model Initialization ---
    model = None
    loss_fn = base_loss_fn

    # --- START OF FIX ---
    # Add an 'elif' block to handle HSDGNN model initialization.
    if config.MODEL_TYPE == "HSDGNN":
        # HSDGNN's constructor expects an 'args' object, not a dictionary.
        # We build it using Namespace from our config values.
        hsdgnn_constructor_args = Namespace(
            num_nodes=X_sample.shape[2],
            input_dim=input_features,
            output_dim=y_sample.shape[3], # Get output dim from target tensor
            rnn_units=model_args['rnn_units'],
            embed_dim=model_args['embed_dim'],
            steps_per_day=model_args['steps_per_day'],
            lag=config.POINT_INPUT_WINDOW_SIZE,
            horizon=config.POINT_OUTPUT_WINDOW_SIZE,
            batch_size=config.BATCH_SIZE
        )
        model = models.HSDGNN(hsdgnn_constructor_args)
        loss_fn = nn.MSELoss() # Point prediction uses standard MSE
    # --- END OF FIX ---
    
    elif config.MODEL_TYPE == "GraphWaveNet":
        if supports is None:
            raise ValueError("GraphWaveNet model requires an adjacency matrix (supports).")
        model_args['num_nodes'] = supports[0].shape[0]
        model_args['supports'] = supports
        model_args['in_dim'] = input_features
        model_args['out_dim'] = output_size
        model = models.GraphWaveNet(**model_args)
        loss_fn = nn.MSELoss()

    elif config.MODEL_TYPE == "DSTAGNN":
        if supports is None:
            raise ValueError("DSTAGNN model requires an adjacency matrix (supports).")
        input_window = config.TREND_INPUT_WINDOW_SIZE if config.PREDICTION_MODE == "TREND" else config.POINT_INPUT_WINDOW_SIZE
        model = models.make_dstagnn_model(
            num_nodes=supports[0].shape[0],
            in_dim=input_features,
            out_dim=output_size,
            input_window=input_window,
            adj_matrix=supports[0].cpu().numpy(),
            dstagnn_args=model_args
        )
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
        model = models.MLP_Forecaster(**model_args)
        
    else:
        raise ValueError(f"Model type '{config.MODEL_TYPE}' not recognized.")
    
    if model is not None:
        model.double()

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
                # Add logic to reshape HSDGNN's unique output tensor before loss calculation.
                if config.MODEL_TYPE == 'HSDGNN':
                    # HSDGNN Output: (B, H*D_out, N, 1) -> Reshape to (B, H, N, D_out) to match y_batch
                    B, _, N, _ = y_pred.shape
                    H = config.POINT_OUTPUT_WINDOW_SIZE
                    D_out = y_batch.shape[-1]
                    y_pred = y_pred.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
                # --- END OF FIX ---
                
                elif config.MODEL_TYPE == 'GraphWaveNet':
                    y_pred = y_pred[..., -1:]
                    if config.PREDICTION_MODE == 'TREND':
                        y_pred = y_pred.squeeze(-1).permute(0, 2, 1)
                
                elif config.MODEL_TYPE == 'DSTAGNN' and config.PREDICTION_MODE == 'POINT':
                    y_batch = y_batch.permute(0, 2, 1, 3).squeeze(-1)

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
                # Apply the same output reshaping for HSDGNN in the validation loop.
                if config.MODEL_TYPE == 'HSDGNN':
                    B, _, N, _ = y_pred.shape
                    H = config.POINT_OUTPUT_WINDOW_SIZE
                    D_out = y_batch.shape[-1]
                    y_pred = y_pred.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
                # --- END OF FIX ---

                elif config.MODEL_TYPE == 'GraphWaveNet':
                    y_pred = y_pred[..., -1:]
                    if config.PREDICTION_MODE == 'TREND':
                        y_pred = y_pred.squeeze(-1).permute(0, 2, 1)
                
                elif config.MODEL_TYPE == 'DSTAGNN' and config.PREDICTION_MODE == 'POINT':
                    y_batch = y_batch.permute(0, 2, 1, 3).squeeze(-1)
                
                val_loss += loss_fn(y_pred, y_batch).item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nModel training complete in {training_time:.2f} seconds.")
    return model, training_time