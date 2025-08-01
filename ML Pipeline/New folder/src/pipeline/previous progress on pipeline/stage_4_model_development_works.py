# src/pipeline/stage_4_model_development.py

import torch
import torch.nn as nn
import time
from src.utils import print_header
from src import models

def _trend_loss_function(predicted, actual, mse_criterion):
    loss_slope = mse_criterion(predicted[:, 0], actual[:, 0])
    loss_duration = mse_criterion(predicted[:, 1], actual[:, 1])
    return (loss_slope + loss_duration) / 2.0

def run(train_loader, val_loader, config):
    print_header("Stage 4: Model Development")
    print(f"""
Purpose: Train a model for {config.PREDICTION_MODE} prediction.
- Model Architecture: {config.MODEL_TYPE}
Executing Code: Initializing and training the {config.MODEL_TYPE} model...
    """)

    model_args = config.MODEL_ARGS[config.MODEL_TYPE].copy()
    if config.PREDICTION_MODE == "TREND":
        model_args['output_size'] = 2
        input_window = config.TREND_INPUT_WINDOW_SIZE
        loss_fn = lambda pred, act: _trend_loss_function(pred, act, nn.MSELoss())
    else: # "POINT" mode
        model_args['output_size'] = config.POINT_OUTPUT_WINDOW_SIZE
        input_window = config.POINT_INPUT_WINDOW_SIZE
        loss_fn = nn.MSELoss()

    model = None
    if config.MODEL_TYPE == "TCN":
        model = models.TCN_Forecaster(**model_args)
    elif config.MODEL_TYPE == "MLP":
        input_size = input_window * model_args.pop('input_channels')
        model_args['input_size'] = input_size
        model = models.MLP_Forecaster(**model_args)
    else:
        raise ValueError(f"Model type '{config.MODEL_TYPE}' not recognized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("\nStarting model training...")
    start_time = time.time()
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            if config.PREDICTION_MODE == "POINT": y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                if config.PREDICTION_MODE == "POINT": y_pred = y_pred.squeeze()
                val_loss += loss_fn(y_pred, y_batch).item()
        
        if (epoch + 1) % 10 == 0:
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS}, Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"Model training complete.")
    return model, training_time
