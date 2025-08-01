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
    - For graph models (GraphWaveNet, DSTAGNN), it performs a chronological split.
    - For other models (TCN, MLP), it uses a stratified split by stock ID.
    """
    print_header("Stage 3: Data Partitioning")

    # --- START OF FIX ---
    # Check if we are using a graph model and partition accordingly
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        print("""
Purpose: Split data chronologically for a graph-based model.
- Method: Time-based split.
- Train Split: First {}% of samples
- Validation Split: Next {}% of samples
- Test Split: Last {}% of samples
Executing Code: Partitioning data and creating PyTorch DataLoaders...
        """.format(
            int(config.TRAIN_SPLIT * 100),
            int(config.VAL_SPLIT * 100),
            int((1 - config.TRAIN_SPLIT - config.VAL_SPLIT) * 100)
        ))

        num_samples = X.shape[0]
        train_end_idx = int(num_samples * config.TRAIN_SPLIT)
        val_end_idx = train_end_idx + int(num_samples * config.VAL_SPLIT)

        X_train, y_train = X[:train_end_idx], y[:train_end_idx]
        X_val, y_val = X[train_end_idx:val_end_idx], y[train_end_idx:val_end_idx]
        X_test, y_test = X[val_end_idx:], y[val_end_idx:]
        
        # For graph models, stock_ids_test is the full ordered list of nodes
        stock_ids_test = stock_ids

    else: # Original logic for TCN, MLP
        print(f"""
Purpose: Split data into training, validation, and test sets.
- Method: Stratified split based on StockID.
- Train Split: {config.TRAIN_SPLIT * 100:.0f}%
- Validation Split: {config.VAL_SPLIT * 100:.0f}%
- Test Split: {(1 - config.TRAIN_SPLIT - config.VAL_SPLIT) * 100:.0f}%
Executing Code: Partitioning data and creating PyTorch DataLoaders...
        """)
        
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

    # For POINT mode with TCN/MLP, y has shape (samples, output_window, 1). Squeeze the last dim.
    if config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN'] and config.PREDICTION_MODE == "POINT":
        if y_train_t.dim() == 3 and y_train_t.shape[2] == 1:
            y_train_t = y_train_t.squeeze(-1)
            y_val_t = y_val_t.squeeze(-1)
            y_test_t = y_test_t.squeeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # For graph models, we don't want to shuffle the data as the time order matters
    shuffle_train = config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN']
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nPyTorch DataLoaders created with batch size: {config.BATCH_SIZE}")
    print(f"  - Training data shuffling: {shuffle_train}")

    return train_loader, val_loader, X_test_t, y_test_t, stock_ids_test
