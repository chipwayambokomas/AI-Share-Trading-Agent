# src/pipeline/stage_3_data_partitioning.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
# -- MODIFICATION: train_test_split is no longer needed --
# from sklearn.model_selection import train_test_split
from src.utils import print_header

# --- MODIFICATION START: The function now accepts pre-partitioned data ---
def run(train_data, val_data, test_data, config):
    """
    Takes pre-partitioned training, validation, and test sets and creates
    PyTorch DataLoaders for the training and validation sets.
    """
    print_header("Stage 3: Data Loader Creation")
    print(f"""
Purpose: Convert pre-partitioned datasets into PyTorch Tensors and DataLoaders.
- Method: The data was already split chronologically in Stage 2.
- Batch Size: {config.BATCH_SIZE}
Executing Code: Creating PyTorch DataLoaders...
    """)

    # Unpack the data tuples passed from Stage 2
    X_train, y_train, _ = train_data
    X_val, y_val, _ = val_data
    X_test, y_test, stock_ids_test = test_data
# --- MODIFICATION END ---

# --- MODIFICATION START: All splitting logic has been removed ---
    # The train_test_split calls are no longer necessary as this functionality
    # was moved to Stage 2 of the pipeline.
# --- MODIFICATION END ---

    print(f"\nData received:")
    #print(f"  - Training set sequences:   {len(X_train)}")
    #print(f"  - Validation set sequences: {len(X_val)}")
    #print(f"  - Test set sequences:       {len(X_test)}")

    # This part remains the same: convert numpy arrays to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # This part remains the same: reshape tensor for POINT mode
    if config.PREDICTION_MODE == "POINT":
        if y_train_t.dim() == 3 and y_train_t.shape[2] == 1:
            y_train_t = y_train_t.squeeze(-1)
            y_val_t = y_val_t.squeeze(-1)
            y_test_t = y_test_t.squeeze(-1)

    # This part remains the same: create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nPyTorch DataLoaders created with batch size: {config.BATCH_SIZE}")

    # The return signature is the same, as the final required components are identical.
    return train_loader, val_loader, X_test_t, y_test_t, stock_ids_test