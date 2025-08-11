import optuna
import torch
from src.config import settings
from src.utils import print_header, setup_logging
from src.pipeline import data_loader, data_partitioner, model_trainer
from src.models.model_factory import ModelFactory
from src.data_processing.preprocessor_factory import PreprocessorFactory

def objective(trial):
    """
    Defines the HPO search space, configures a trial, and kicks off the
    model_trainer, which now handles pruning and returns the best validation loss.
    """
    print_header(f"Starting Trial {trial.number}")

    # 1. DEFINE THE SCIENTIFICALLY-BACKED SEARCH SPACE
    # Architectural Hyperparameters
    agcrn_params = {
        'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3]),
        'embed_dim': trial.suggest_categorical('embed_dim', [5, 10, 15]),
        'rnn_units': trial.suggest_categorical('rnn_units', [32, 64, 128]),
        'cheb_k': 2,  # Fixed common default
    }

    # Training and Data Hyperparameters
    trial_epochs = trial.suggest_categorical('epochs', [10, 20, 30, 40, 50])
    trial_batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    trial_lr = trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.0001])
    
    # 2. CREATE A TEMPORARY CONFIGURATION FOR THIS TRIAL
    class TrialSettings:
        pass
    
    temp_settings = TrialSettings()
    for key, value in settings.__dict__.items():
        if not key.startswith('__'):
            setattr(temp_settings, key, value)
    
    temp_settings.EPOCHS = trial_epochs
    temp_settings.BATCH_SIZE = trial_batch_size
    temp_settings.LEARNING_RATE = trial_lr
    # Create a copy of MODEL_ARGS to avoid modifying the global settings dict
    temp_settings.MODEL_ARGS = settings.MODEL_ARGS.copy()
    temp_settings.MODEL_ARGS['AGCRN'] = agcrn_params

    try:
        # 3. RUN THE PIPELINE FOR THIS TRIAL
        global train_dataset, val_dataset, model_handler, static_adj_matrix

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=temp_settings.BATCH_SIZE, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=temp_settings.BATCH_SIZE, shuffle=False)

        # The trainer now handles all training, validation, and pruning.
        # We pass the `trial` object directly to it.
        _, _, _, best_val_loss = model_trainer.run(
            train_loader=train_loader,
            val_loader=val_loader,
            model_handler=model_handler,
            settings=temp_settings,
            supports=[static_adj_matrix] if static_adj_matrix is not None else None,
            verbose=False,
            trial=trial  # Pass the trial object for pruning
        )

        print(f"Trial {trial.number} finished with Best Validation Loss: {best_val_loss:.6f}")
        return best_val_loss

    except optuna.exceptions.TrialPruned:
        # This is not an error, just Optuna stopping a bad trial.
        # Return the special value to let the study know.
        return optuna.trial.TrialState.PRUNED
    except Exception as e:
        print(f"Trial {trial.number} failed with a critical error: {e}")
        return float('inf')  # Penalize critical failures

# --- Main Execution Block ---
if __name__ == "__main__":
    setup_logging()
    
    # 1. SETUP: PRE-LOAD AND PROCESS DATA ONCE
    print_header("HPO Setup: Pre-loading and Processing Data")
    
    # The HPO is fixed for AGCRN POINT mode. The input window is taken from settings.py
    settings.MODEL_TYPE = "AGCRN"
    settings.PREDICTION_MODE = "POINT"
    
    model_handler = ModelFactory.create(settings.MODEL_TYPE, settings)
    preprocessor = PreprocessorFactory.create(model_handler, settings)

    combined_df = data_loader.run(settings)
    X, y, stock_ids, dates, scalers, static_adj_matrix = preprocessor.process(combined_df)
    
    train_loader, val_loader, _, _, _, _ = data_partitioner.run(
        X, y, stock_ids, dates, model_handler.is_graph_based(), settings
    )
    # Store datasets globally for trials to access
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # 2. RUN THE OPTIMIZATION STUDY
    print_header("Starting Bayesian Optimization for AGCRN")
    print(f"Using fixed Input Window = {settings.TREND_INPUT_WINDOW_SIZE}")
    print(f"Optimizing over search space from paper for {settings.MODEL_TYPE} in {settings.PREDICTION_MODE} mode.")
    print("Pruning enabled: Unpromising trials will be stopped early.")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner() # Pruner is enabled here
    )
    
    study.optimize(objective, n_trials=50) # You can change n_trials

    # 3. DISPLAY THE FINAL RESULTS
    print_header("Optimization Finished")
    print(f"Number of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    
    print(f"  - Pruned trials: {len(pruned_trials)}")
    print(f"  - Completed trials: {len(complete_trials)}")
    
    print(f"\nBest trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    
    print("\n--- Best Hyperparameters Found ---")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    print("\nNext step: Manually update the 'AGCRN' section in 'src/config/settings.py' with these values.")