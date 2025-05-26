# train/trainMLP.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml

from data.data import load_data
from models.mlp_model import MLPModel


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config_from_path(path="configs/mlpconfig.yaml"):
    """Load configuration from a specific YAML file path"""
    config_path = Path(__file__).resolve().parents[1] / path
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train(config: dict, trial_name_for_temp_checkpoint: str):
    set_seed(config["experiment"]["seed"])
    data_loaders, _ = load_data(config)
    training_results = [[None for _ in range(3)] for _ in range(38)]
    project_root = Path(config.get("project_root_dir", "."))

    count = 0
    for i, (sheet_name,train_loader, val_loader,_) in enumerate(data_loaders):
    
        temp_checkpoint_dir = project_root / f"checkpoints/MLP/{sheet_name}/temp_trial_models"
        temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        current_trial_temp_checkpoint_path = temp_checkpoint_dir / f"temp_model_{trial_name_for_temp_checkpoint}.pt"
        

        if len(train_loader.dataset) == 0:
            print(f"Warning: Training dataset for trial '{trial_name_for_temp_checkpoint}' is empty. Skipping training.")
            return float('inf'), None

        if len(val_loader.dataset) == 0 and config["data"]["split"]["val"] > 0:
            print(f"Warning: Validation dataset for trial '{trial_name_for_temp_checkpoint}' is empty. Proceeding without validation.")

        model = MLPModel(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            output_size=config["model"]["output_size"]
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.0)
        )

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        early_stopping_patience = config["training"].get("early_stopping_patience", 10)
        total_val_samples = len(val_loader.dataset)

        for epoch in range(config["training"]["epochs"]):
            model.train()
            running_train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device).float()
                y_batch = y_batch.to(device).float()
                if y_batch.ndim == 1:
                    y_batch = y_batch.unsqueeze(1)

                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_train_loss += loss.item() * x_batch.size(0)


            avg_val_loss = float('inf')
            if total_val_samples > 0:
                model.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(device).float()
                        y_batch = y_batch.to(device).float()
                        if y_batch.ndim == 1:
                            y_batch = y_batch.unsqueeze(1)

                        output = model(x_batch)
                        loss = criterion(output, y_batch)
                        running_val_loss += loss.item() * x_batch.size(0)

                avg_val_loss = running_val_loss / total_val_samples

            # Save best model checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config
                }, current_trial_temp_checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                break
        
        training_results[count] = [best_val_loss,str(current_trial_temp_checkpoint_path),str(sheet_name)]
        count += 1

    return training_results