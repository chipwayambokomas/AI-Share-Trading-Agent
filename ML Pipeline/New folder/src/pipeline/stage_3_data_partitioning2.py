# src/pipeline/stage_3_data_partitioning.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils import print_header

def run(X, y, stock_ids, config):
    """
    Partitions the data into training, validation, and test sets, then creates
    PyTorch DataLoaders.
    - For graph models (GraphWaveNet, DSTAGNN, HSDGNN), it performs a chronological split.
    - For other models (TCN, MLP), it uses a stratified split by stock ID.
    """
    print_header("Stage 3: Data Partitioning")

    # --- START OF FIX ---
    # Add 'HSDGNN' to the list of models that require a chronological, time-based split.
    # This is the main fix for the error you are seeing.
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        print(f"""
Purpose: Split data chronologically for a graph-based model ({config.MODEL_TYPE}).
- Method: Time-based split.
- Train Split: First {int(config.TRAIN_SPLIT * 100)}% of samples
- Validation Split: Next {int(config.VAL_SPLIT * 100)}% of samples
- Test Split: Last {int((1 - config.TRAIN_SPLIT - config.VAL_SPLIT) * 100)}% of samples
Executing Code: Partitioning data and creating PyTorch DataLoaders...
        """)

        num_samples = X.shape[0]
        train_end_idx = int(num_samples * config.TRAIN_SPLIT)
        val_end_idx = train_end_idx + int(num_samples * config.VAL_SPLIT)

        X_train, y_train = X[:train_end_idx], y[:train_end_idx]
        X_val, y_val = X[train_end_idx:val_end_idx], y[train_end_idx:val_end_idx]
        X_test, y_test = X[val_end_idx:], y[val_end_idx:]
        
        # For graph models, stock_ids_test is the full ordered list of nodes.
        stock_ids_test = stock_ids

    else: # Original logic for TCN, MLP
        # This block remains unchanged and will function as it did before for TCN and MLP.
        print(f"""
Purpose: Split data into training, validation, and test sets.
- Method: Stratified split based on StockID.
- Train Split: {config.TRAIN_SPLIT * 100:.0f}%
- Validation Split: {config.VAL_SPLIT * 100:.0f}%
- Test Split: {(1 - config.TRAIN_SPLIT - config.VAL_SPLIT) * 100:.0f}%
Executing Code: Partitioning data and creating PyTorch DataLoaders...
        """)
        
        # Note: The error occurs when HSDGNN is incorrectly routed here.
        # The 'stratify' argument expects an array with the same number of samples
        # as X and y, which is why the error occurred.
        test_split_size = 1.0 - (config.TRAIN_SPLIT + config.VAL_SPLIT)
        
        X_train_val, X_test, y_train_val, y_test, stock_ids_train_val, stock_ids_test = train_test_split(
            X, y, stock_ids,
            test_size=test_split_size,
            random_state=config.RANDOM_SEED,
            stratify=stock_ids
        )

        relative_val_split = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=relative_val_split,
            random_state=config.RANDOM_SEED,
            stratify=stock_ids_train_val
        )
    # --- END OF FIX ---


    print(f"\nData partitioning complete:")
    print(f"  - Training set size:   {len(X_train)}")
    print(f"  - Validation set size: {len(X_val)}")
    print(f"  - Test set size:       {len(X_test)}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # --- START OF FIX ---
    # Add 'HSDGNN' here to prevent its target tensor from being squeezed.
    # Graph models expect a 4D tensor (Batch, Horizon, Nodes, Features).
    if config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN'] and config.PREDICTION_MODE == "POINT":
        if y_train_t.dim() == 3 and y_train_t.shape[2] == 1:
            y_train_t = y_train_t.squeeze(-1)
            y_val_t = y_val_t.squeeze(-1)
            y_test_t = y_test_t.squeeze(-1)
    # --- END OF FIX ---

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # --- START OF FIX ---
    # Add 'HSDGNN' here to disable shuffling for the training DataLoader,
    # as time order is critical for the model.
    shuffle_train = config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']
    # --- END OF FIX ---
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nPyTorch DataLoaders created with batch size: {config.BATCH_SIZE}")
    print(f"  - Training data shuffling: {shuffle_train}")

    return train_loader, val_loader, X_test_t, y_test_t, stock_ids_test