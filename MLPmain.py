
import argparse
import os
import shutil
import sys

# Add project root to sys.path to allow imports from subdirectories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.utils import load_config_from_path # Centralized config loading
from train.trainMLP import train as train_mlp
from hpo.hpoMLP import run_hpo as run_mlp_hpo
from eval.evaluate import evaluate_trained_model 

# Define standard paths for models relative to project root
MAIN_STANDALONE_MODEL_PATH = "checkpoints/MLP/MLP_best_model_standalone.pt"
HPO_MODEL_PATH = "checkpoints/MLP/MLP_best_hpo_model.pt" # Path used by hpoMLP.py

def setup_directories():
    """Create necessary directories if they don't exist, relative to project root."""
    dirs = ["configs", "data", "results/evaluation_plots", "models"]
    for d in dirs:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
    # Ensure parent dirs for specific final checkpoint paths also exist
    if MAIN_STANDALONE_MODEL_PATH: os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, MAIN_STANDALONE_MODEL_PATH)), exist_ok=True)
    if HPO_MODEL_PATH: os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, HPO_MODEL_PATH)), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Time Series Price Prediction MLP Pipeline")
    parser.add_argument("--config", type=str, default="configs\mlpconfig.yaml",
                        help="Path to the base YAML configuration file (relative to project root).")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "hpo", "evaluate", "full"],
                        help="Mode of operation: 'train' (standalone), 'hpo', 'evaluate' a specific model, or 'full' pipeline.")
    parser.add_argument("--eval_ckpt", type=str, default=None,
                        help="Path to checkpoint file to evaluate (relative to project root, used in 'evaluate' mode). If None, tries HPO then standalone model.")
    args = parser.parse_args()

    setup_directories()
    
    abs_config_path = os.path.join(PROJECT_ROOT, args.config)
    try:
        base_config = load_config_from_path(abs_config_path) # From utils
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Base configuration file {abs_config_path} not found. Exiting.")
        return
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load or parse base configuration {abs_config_path}: {e}. Exiting.")
        return
    
    # Add project_root_dir to config for consistent path handling in sub-modules
    base_config["project_root_dir"] = PROJECT_ROOT


    if args.mode == "train":
        print("\n=== Mode: Training Model (Standalone) ===")
        training_results = train_mlp(base_config, trial_name_for_temp_checkpoint="main_standalone_run")
        for i, (_,temp_ckpt_path,sheet_name) in enumerate(training_results):
            abs_main_standalone_models_path = os.path.join(PROJECT_ROOT, f"checkpoints/MLP/standalone/{sheet_name}_best_model.pt")
            if temp_ckpt_path and os.path.exists(temp_ckpt_path):
                os.makedirs(os.path.dirname(abs_main_standalone_models_path), exist_ok=True)
                shutil.copy(temp_ckpt_path, abs_main_standalone_models_path)
                print(f"Standalone training finished. Model saved to: {abs_main_standalone_models_path}")
                try: os.remove(temp_ckpt_path); # print(f"Cleaned up temp file: {temp_ckpt_path}")
                except OSError as e: print(f"Warning: Could not remove temp file {temp_ckpt_path}: {e}")
            else:
                print("Standalone training did not produce a model checkpoint.")

    elif args.mode == "hpo":
        print("\n=== Mode: Hyperparameter Optimization ===")
        run_mlp_hpo(config_path=abs_config_path, project_root_dir=PROJECT_ROOT)

    elif args.mode == "evaluate":
        print("\n=== Mode: Evaluating Model ===")
        model_to_evaluate_rel_path = args.eval_ckpt
        
        abs_hpo_model_path = os.path.join(PROJECT_ROOT, HPO_MODEL_PATH)
        abs_main_standalone_model_path = os.path.join(PROJECT_ROOT, MAIN_STANDALONE_MODEL_PATH)

        if model_to_evaluate_rel_path is None:
            if os.path.exists(abs_hpo_model_path):
                model_to_evaluate_rel_path = HPO_MODEL_PATH # Use relative path for the function
                print(f"No specific checkpoint. Evaluating HPO best model: {model_to_evaluate_rel_path}")
            elif os.path.exists(abs_main_standalone_model_path):
                model_to_evaluate_rel_path = MAIN_STANDALONE_MODEL_PATH
                print(f"No specific checkpoint, no HPO model. Evaluating standalone model: {model_to_evaluate_rel_path}")
            else:
                print(f"No checkpoint specified and no default models found. Cannot evaluate.")
                return
        
        if model_to_evaluate_rel_path:
            abs_model_to_evaluate_path = os.path.join(PROJECT_ROOT, model_to_evaluate_rel_path)
            if os.path.exists(abs_model_to_evaluate_path):
                evaluate_trained_model(checkpoint_path=abs_model_to_evaluate_path, project_root_dir=PROJECT_ROOT) # Now this function is defined
            else:
                print(f"Error: Checkpoint file for evaluation not found at {abs_model_to_evaluate_path}")

    elif args.mode == "full":
        print("\n=== Mode: Full Pipeline Run ===")
        models_path_for_evaluation_rel = "checkpoints/MLP/standalone" # Relative path to project root
        
        if base_config.get("hpo", {}).get("enabled", False):
            print("\n--- Running HPO (Full Pipeline) ---")
            run_mlp_hpo(config_path=abs_config_path, project_root_dir=PROJECT_ROOT)
            if os.path.exists(os.path.join(PROJECT_ROOT, HPO_MODEL_PATH)):
                model_path_for_evaluation_rel = HPO_MODEL_PATH
            else:
                 print(f"HPO finished, but best model not found at {HPO_MODEL_PATH}. Cannot evaluate.")
                 return
        else:
            print("\n--- Training Model (Standalone, HPO disabled, Full Pipeline) ---")
            training_results = train_mlp(base_config, trial_name_for_temp_checkpoint="main_standalone_run")
            for i, (_,temp_ckpt_path,sheet_name) in enumerate(training_results):
                abs_main_standalone_models_path = os.path.join(PROJECT_ROOT, f"checkpoints/MLP/standalone/{sheet_name}_best_model.pt")
                if temp_ckpt_path and os.path.exists(temp_ckpt_path):
                    os.makedirs(os.path.dirname(abs_main_standalone_models_path), exist_ok=True)
                    shutil.copy(temp_ckpt_path, abs_main_standalone_models_path)
                    try: os.remove(temp_ckpt_path); # print(f"Cleaned up temp file: {temp_ckpt_path}")
                    except OSError as e: print(f"Warning: Could not remove temp file {temp_ckpt_path}: {e}")
                else:
                    print("Standalone training (full pipeline) did not produce a model. Cannot evaluate.")
                    return
            

        if models_path_for_evaluation_rel:
            abs_models_path_for_evaluation = os.path.join(PROJECT_ROOT, models_path_for_evaluation_rel)
            if os.path.exists(abs_models_path_for_evaluation):
                print(f"\n--- Evaluating Model: {models_path_for_evaluation_rel} (Full Pipeline) ---")
                evaluate_trained_model(checkpoint_path=abs_models_path_for_evaluation, project_root_dir=PROJECT_ROOT) 
            else: # Should not happen if logic is correct
                print(f"Model path {model_path_for_evaluation_rel} determined but file not found at {abs_models_path_for_evaluation}.")
        else:
            print("No model was successfully trained or found from the pipeline steps for evaluation.")

    print("\nMain script execution finished.")

if __name__ == "__main__":
    main()