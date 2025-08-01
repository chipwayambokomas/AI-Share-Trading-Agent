# run_tuning.py

import torch
import torch.nn as nn
import optuna
import copy
import importlib
import time
import logging
from tqdm import tqdm

from pipeline import stage_3_data_partitioning1
from src.pipeline import (
    stage_1_feature_extraction,
)
from src import config, models
from src.utils import print_header


# --- NEW: Set up logging ---
# Tqdm-aware handler to prevent progress bars from interfering with logs
class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())
logger.addHandler(logging.FileHandler("tuning.log")) # Log to a file


# --- FIX: Helper class to make the config serializable ---
class ConfigObject:
    """A simple class to hold configuration attributes."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))

def to_config_object(config_module):
    """Converts a module to a ConfigObject instance."""
    config_dict = {key: getattr(config_module, key) for key in dir(config_module) if not key.startswith('__')}
    return ConfigObject(**config_dict)


# --- 1. Objective Function Definition ---
def objective(trial, base_config, train_loader, val_loader, supports):
    """
    The objective function for Optuna.
    """
    logger.info(f"--- Starting Trial {trial.number} ---")
    
    trial_config = copy.deepcopy(base_config)

    # --- Suggest Hyperparameters ---
    trial_config.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    trial_config.BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    model_type = trial_config.MODEL_TYPE
    if model_type == "GraphWaveNet":
        trial_config.MODEL_ARGS["GraphWaveNet"]["dropout"] = trial.suggest_float("gwn_dropout", 0.1, 0.5)
        trial_config.MODEL_ARGS["GraphWaveNet"]["residual_channels"] = trial.suggest_categorical("gwn_residual_channels", [16, 32])
        trial_config.MODEL_ARGS["GraphWaveNet"]["dilation_channels"] = trial.suggest_categorical("gwn_dilation_channels", [16, 32])
        trial_config.MODEL_ARGS["GraphWaveNet"]["skip_channels"] = trial.suggest_categorical("gwn_skip_channels", [64, 128, 256])
        trial_config.MODEL_ARGS["GraphWaveNet"]["end_channels"] = trial.suggest_categorical("gwn_end_channels", [128, 256, 512])
        trial_config.MODEL_ARGS["GraphWaveNet"]["blocks"] = trial.suggest_int("gwn_blocks", 2, 5)
        trial_config.MODEL_ARGS["GraphWaveNet"]["layers"] = trial.suggest_int("gwn_layers", 2, 4)

    elif model_type == "DSTAGNN":
        trial_config.MODEL_ARGS["DSTAGNN"]["nb_block"] = trial.suggest_int("dstagnn_nb_block", 1, 3)
        trial_config.MODEL_ARGS["DSTAGNN"]["K"] = trial.suggest_int("dstagnn_K", 2, 4)
        trial_config.MODEL_ARGS["DSTAGNN"]["nb_chev_filter"] = trial.suggest_categorical("dstagnn_nb_chev_filter", [32, 64])
        trial_config.MODEL_ARGS["DSTAGNN"]["d_model"] = trial.suggest_categorical("dstagnn_d_model", [32, 64])
        trial_config.MODEL_ARGS["DSTAGNN"]["d_k"] = trial.suggest_categorical("dstagnn_d_k", [16, 32])
        trial_config.MODEL_ARGS["DSTAGNN"]["d_v"] = trial.suggest_categorical("dstagnn_d_v", [16, 32])
        trial_config.MODEL_ARGS["DSTAGNN"]["n_heads"] = trial.suggest_categorical("dstagnn_n_heads", [2, 4, 8])

    logger.info(f"Trial {trial.number} Parameters: {trial.params}")

    # --- Re-create DataLoaders with the new batch size ---
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    trial_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=trial_config.BATCH_SIZE, shuffle=False)
    trial_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=trial_config.BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    model_args = trial_config.MODEL_ARGS[model_type]
    
    if trial_config.PREDICTION_MODE == "TREND":
        input_features, output_size = 2, 2
    else: # "POINT"
        input_features = len(trial_config.FEATURE_COLUMNS)
        output_size = trial_config.POINT_OUTPUT_WINDOW_SIZE
    
    loss_fn = nn.MSELoss()
    model = None

    if model_type == "GraphWaveNet":
        model_args.update({'num_nodes': supports[0].shape[0], 'supports': supports, 'in_dim': input_features, 'out_dim': output_size})
        model = models.GraphWaveNet(**model_args)
    elif model_type == "DSTAGNN":
        input_window = trial_config.TREND_INPUT_WINDOW_SIZE if trial_config.PREDICTION_MODE == "TREND" else trial_config.POINT_INPUT_WINDOW_SIZE
        model = models.make_dstagnn_model(
            num_nodes=supports[0].shape[0], in_dim=input_features, out_dim=output_size,
            input_window=input_window, adj_matrix=supports[0].cpu().numpy(), dstagnn_args=model_args
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=trial_config.LEARNING_RATE)

    # --- Training & Validation Loop ---
    best_val_loss = float('inf')
    epochs_for_tuning = 30

    for epoch in range(epochs_for_tuning):
        model.train()
        # Wrap the loader with tqdm for a progress bar
        train_pbar = tqdm(trial_train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{epochs_for_tuning} Training", leave=False)
        for X_batch, y_batch in train_pbar:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            if model_type == 'GraphWaveNet':
                y_pred = y_pred[..., -1:]
                if trial_config.PREDICTION_MODE == 'TREND':
                    y_pred = y_pred.squeeze(-1).permute(0, 2, 1)
            elif model_type == 'DSTAGNN' and trial_config.PREDICTION_MODE == 'POINT':
                y_batch = y_batch.permute(0, 2, 1, 3).squeeze(-1)

            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=loss.item())

        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in trial_val_loader:
                y_pred = model(X_batch)
                
                if model_type == 'GraphWaveNet':
                    y_pred = y_pred[..., -1:]
                    if trial_config.PREDICTION_MODE == 'TREND':
                        y_pred = y_pred.squeeze(-1).permute(0, 2, 1)
                elif model_type == 'DSTAGNN' and trial_config.PREDICTION_MODE == 'POINT':
                    y_batch = y_batch.permute(0, 2, 1, 3).squeeze(-1)
                
                current_val_loss += loss_fn(y_pred, y_batch).item()
        
        avg_val_loss = current_val_loss / len(trial_val_loader)
        logger.info(f"Trial {trial.number} - Epoch {epoch+1}: Validation Loss = {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        trial.report(best_val_loss, epoch)
        if trial.should_prune():
            logger.warning(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"--- Trial {trial.number} Finished. Final Validation Loss: {best_val_loss:.6f} ---")
    return best_val_loss


# --- 2. Main Execution Block ---
def main():
    print_header("Hyperparameter Optimization for Graph-Based Models")

    if config.MODEL_TYPE not in ["GraphWaveNet", "DSTAGNN"]:
        raise ValueError(f"This script is for GraphWaveNet or DSTAGNN. Update config.py.")

    serializable_config = to_config_object(config)

    logger.info("--- Running Initial Data Preparation (Stages 1-3) ---")
    combined_df = stage_1_feature_extraction.run(serializable_config)

    preprocessing_module_mapping = {
        ("POINT", "GraphWaveNet"): "src.pipeline.stage_2_data_preprocessing.gnn_point_prediction",
        ("POINT", "DSTAGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_point_prediction",
        ("TREND", "GraphWaveNet"): "src.pipeline.stage_2_data_preprocessing.gnn_trendline_prediction",
        ("TREND", "DSTAGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_trendline_prediction",
    }
    module_path = preprocessing_module_mapping[(serializable_config.PREDICTION_MODE, serializable_config.MODEL_TYPE)]
    stage_2_module = importlib.import_module(module_path)
    X, y, stock_ids, _, adj_matrix = stage_2_module.run(combined_df, serializable_config)

    train_loader, val_loader, _, _, _ = stage_3_data_partitioning1.run(X, y, stock_ids, serializable_config)

    if adj_matrix is None:
        raise ValueError("Graph models require an adjacency matrix.")
    supports = [adj_matrix]
    
    logger.info("--- Data Preparation Complete ---")

    print_header(f"Starting Optuna Study for {serializable_config.MODEL_TYPE} in {serializable_config.PREDICTION_MODE} mode")
    
    objective_with_data = lambda trial: objective(trial, serializable_config, train_loader, val_loader, supports)
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    n_trials = 2
    timeout_seconds = 3600
    logger.info(f"Starting study with {n_trials} trials and a timeout of {timeout_seconds} seconds.")
    study.optimize(objective_with_data, n_trials=n_trials, timeout=timeout_seconds)

    print_header("Hyperparameter Optimization Finished")
    logger.info("--- STUDY RESULTS ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    logger.info(f"Best trial:")
    logger.info(f"  Value (Min Validation Loss): {best_trial.value:.6f}")
    
    logger.info("  Best Hyperparameters (update config.py with these):")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

if __name__ == '__main__':
    main()
