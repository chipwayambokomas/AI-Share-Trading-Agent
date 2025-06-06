import os
import sys
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import json
import shutil
from typing import Dict, Any

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train.trainTCN import train as train_tcn
from utils.utils import load_config_from_path, setup_logging


def suggest_tcn_hyperparameters(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest hyperparameters for TCN model optimization
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary
        
    Returns:
        Modified configuration with suggested hyperparameters
    """
    # Create a copy of the base config
    config = base_config.copy()
    
    # Ensure nested dictionaries exist
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}
    
    # TCN-specific hyperparameters
    
    # Number of channels (layers and their sizes)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    base_channels = trial.suggest_categorical('base_channels', [32, 64, 128])
    
    # Generate channel configuration
    num_channels = []
    for i in range(num_layers):
        if i == 0:
            channels = base_channels
        else:
            # Each layer can have same, double, or half the channels of previous layer
            multiplier = trial.suggest_categorical(f'layer_{i}_multiplier', [0.5, 1.0, 2.0])
            channels = int(num_channels[i-1] * multiplier)
            channels = max(16, min(512, channels))  # Keep within reasonable bounds
        num_channels.append(channels)
    
    config['model']['num_channels'] = num_channels
    
    # Kernel size
    config['model']['kernel_size'] = trial.suggest_int('kernel_size', 2, 8)
    
    # Dropout rate
    config['model']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Sequence length
    config['model']['sequence_length'] = trial.suggest_categorical('sequence_length', [30, 60, 90, 120])
    
    # Training hyperparameters
    config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Early stopping patience
    config['training']['patience'] = trial.suggest_int('patience', 10, 30)
    
    # Maximum epochs (can be pruned)
    config['training']['epochs'] = trial.suggest_int('epochs', 50, 200)
    
    return config


def objective(trial: optuna.Trial, base_config: Dict[str, Any], logger) -> float:
    """
    Objective function for Optuna optimization
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        logger: Logger instance
        
    Returns:
        Validation loss to minimize
    """
    try:
        # Suggest hyperparameters
        config = suggest_tcn_hyperparameters(trial, base_config)
        
        # Generate unique trial name
        trial_name = f"trial_{trial.number:04d}"
        
        logger.info(f"Starting trial {trial.number} with config: {json.dumps(config['model'], indent=2)}")
        
        # Train model with suggested hyperparameters
        metrics, temp_checkpoint_path = train_tcn(
            config=config,
            trial_name_for_temp_checkpoint=trial_name,
            trial=trial
        )
        
        # Clean up temporary checkpoint
        if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
            try:
                os.remove(temp_checkpoint_path)
            except OSError:
                pass
        
        # Return validation loss (metric to minimize)
        val_loss = metrics.get('best_val_loss', float('inf'))
        
        # Report additional metrics for monitoring
        trial.set_user_attr('test_mse', metrics.get('mse', 0))
        trial.set_user_attr('test_mae', metrics.get('mae', 0))
        trial.set_user_attr('test_rmse', metrics.get('rmse', 0))
        trial.set_user_attr('test_mape', metrics.get('mape', 0))
        trial.set_user_attr('best_epoch', metrics.get('best_epoch', 0))
        trial.set_user_attr('total_epochs', metrics.get('total_epochs', 0))
        
        logger.info(f"Trial {trial.number} completed. Validation loss: {val_loss:.6f}")
        
        return val_loss
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}")
        # Return a large value to indicate failure
        return float('inf')


def run_hpo(config_path: str, project_root_dir: str):
    """
    Run hyperparameter optimization for TCN model
    
    Args:
        config_path: Path to configuration file
        project_root_dir: Project root directory
    """
    # Setup logging
    logger = setup_logging(__name__)
    logger.info("Starting TCN Hyperparameter Optimization")
    
    try:
        # Load base configuration
        base_config = load_config_from_path(config_path)
        base_config['project_root_dir'] = project_root_dir
        
        # HPO configuration
        hpo_config = base_config.get('hpo', {})
        n_trials = hpo_config.get('n_trials', 50)
        timeout = hpo_config.get('timeout', 3600)  # 1 hour default
        
        # Study configuration
        study_name = hpo_config.get('study_name', 'tcn_optimization')
        storage = hpo_config.get('storage', None)  # Can be a database URL
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        
        logger.info(f"Starting optimization with {n_trials} trials")
        logger.info(f"Study name: {study_name}")
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, base_config, logger),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best trial
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best validation loss: {best_trial.value:.6f}")
        logger.info(f"Best parameters: {json.dumps(best_trial.params, indent=2)}")
        
        # Create best configuration
        best_config = suggest_tcn_hyperparameters(best_trial, base_config)
        
        # Train final model with best hyperparameters
        logger.info("Training final model with best hyperparameters...")
        
        final_metrics, temp_checkpoint_path = train_tcn(
            config=best_config,
            trial_name_for_temp_checkpoint="best_hpo_model"
        )
        
        # Save best model to permanent location
        best_model_path = os.path.join(project_root_dir, "checkpoints", "TCN", "TCN_best_hpo_model.pt")
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
            shutil.copy(temp_checkpoint_path, best_model_path)
            logger.info(f"Best model saved to: {best_model_path}")
            
            # Clean up temporary checkpoint
            try:
                os.remove(temp_checkpoint_path)
            except OSError:
                pass
        
        # Save optimization results
        results_dir = os.path.join(project_root_dir, "results", "hpo")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best configuration
        best_config_path = os.path.join(results_dir, "tcn_best_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        # Save optimization study results
        study_results = {
            'best_trial_number': best_trial.number,
            'best_validation_loss': best_trial.value,
            'best_parameters': best_trial.params,
            'best_metrics': final_metrics,
            'total_trials': len(study.trials),
            'study_name': study_name
        }
        
        study_results_path = os.path.join(results_dir, "tcn_hpo_results.json")
        with open(study_results_path, 'w') as f:
            json.dump(study_results, f, indent=2, default=str)
        
        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_csv_path = os.path.join(results_dir, "tcn_hpo_trials.csv")
        trials_df.to_csv(trials_csv_path, index=False)
        
        logger.info(f"HPO results saved to: {results_dir}")
        logger.info(f"Final model metrics: {json.dumps(final_metrics, indent=2, default=str)}")
        
        # Print summary
        print("\n" + "="*50)
        print("TCN HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("="*50)
        print(f"Best validation loss: {best_trial.value:.6f}")
        print(f"Best test MSE: {final_metrics.get('mse', 'N/A'):.4f}")
        print(f"Best test RMSE: {final_metrics.get('rmse', 'N/A'):.4f}")
        print(f"Best test MAPE: {final_metrics.get('mape', 'N/A'):.2f}%")
        print(f"Total trials completed: {len(study.trials)}")
        print(f"Best model saved to: {best_model_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"HPO failed with error: {str(e)}")
        raise e


def analyze_hpo_results(results_path: str):
    """
    Analyze and visualize HPO results
    
    Args:
        results_path: Path to HPO results directory
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Load trials data
    trials_csv_path = os.path.join(results_path, "tcn_hpo_trials.csv")
    if not os.path.exists(trials_csv_path):
        print(f"Trials data not found at {trials_csv_path}")
        return
    
    trials_df = pd.read_csv(trials_csv_path)
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Objective value over trials
    plt.subplot(2, 2, 1)
    plt.plot(trials_df['number'], trials_df['value'])
    plt.title('Optimization History')
    plt.xlabel('Trial')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    
    # Subplot 2: Parameter importance (if enough trials)
    if len(trials_df) > 10:
        plt.subplot(2, 2, 2)
        # Plot distribution of learning rates
        lr_col = [col for col in trials_df.columns if 'learning_rate' in col]
        if lr_col:
            plt.hist(trials_df[lr_col[0]].dropna(), bins=20, alpha=0.7)
            plt.title('Learning Rate Distribution')
            plt.xlabel('Learning Rate')
            plt.ylabel('Frequency')
    
    # Subplot 3: Best trial metrics
    plt.subplot(2, 2, 3)
    best_idx = trials_df['value'].idxmin()
    best_trial = trials_df.loc[best_idx]
    
    metrics = ['test_mse', 'test_mae', 'test_rmse', 'test_mape']
    values = [best_trial.get(f'user_attrs_{metric}', 0) for metric in metrics]
    
    plt.bar(metrics, values)
    plt.title('Best Trial Metrics')
    plt.xticks(rotation=45)
    plt.ylabel('Value')
    
    # Subplot 4: Trial durations
    plt.subplot(2, 2, 4)
    if 'duration' in trials_df.columns:
        plt.hist(trials_df['duration'].dropna(), bins=20, alpha=0.7)
        plt.title('Trial Duration Distribution')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_path, "tcn_hpo_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"HPO analysis plot saved to: {plot_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TCN Hyperparameter Optimization")
    parser.add_argument('--config', type=str, default='configs/tcnconfig.yaml',
                       help='Path to configuration file')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing HPO results')
    parser.add_argument('--results_path', type=str, default='results/hpo',
                       help='Path to HPO results for analysis')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_hpo_results(args.results_path)
    else:
        run_hpo(args.config, PROJECT_ROOT)