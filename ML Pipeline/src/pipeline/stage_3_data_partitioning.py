# src/pipeline/stage_3_data_partitioning.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils import print_header

def run(X, y, stock_ids, dates, config):
    """
    Partitions the data and dates into training, validation, and test sets,
    then creates PyTorch DataLoaders.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        stock_ids (np.ndarray): Array of stock IDs for stratification (non-graph)
                                or list of nodes (graph).
        dates (np.ndarray): Array of dates corresponding to the samples in X and y.
        config (Namespace): Configuration object with model and data parameters.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - X_test_t (torch.Tensor): Test set features.
            - y_test_t (torch.Tensor): Test set targets.
            - stock_ids_test (np.ndarray): Test set stock IDs.
            - dates_test (np.ndarray): Test set dates.
    """
    print_header("Stage 3: Data Partitioning")

    # --- START OF MODIFICATIONS ---
    # Validation logic to ensure data consistency
    if not (len(X) == len(y) == len(dates)):
        raise ValueError(
            f"Inconsistent sample dimensions found:\n"
            f"  - X samples: {len(X)}\n"
            f"  - y samples: {len(y)}\n"
            f"  - dates samples: {len(dates)}\n"
            "These lengths must always be identical."
        )

    if config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        if len(X) != len(stock_ids):
            raise ValueError(
                f"Inconsistent input lengths for non-graph model '{config.MODEL_TYPE}':\n"
                f"  - X samples: {len(X)}\n"
                f"  - stock_ids samples: {len(stock_ids)}\n"
                "For TCN/MLP models, the number of stock_ids must match the number of samples."
            )

    print(f"""
Purpose: Split data chronologically for all models.
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
    
    dates_test = dates[val_end_idx:]
    
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN']:
        # For graph models, stock_ids is the list of all nodes
        stock_ids_test = stock_ids
    else:
        # For non-graph models, slice the stock_ids array as well
        stock_ids_test = stock_ids[val_end_idx:]
    # --- END OF MODIFICATIONS ---

    print(f"\nData partitioning complete:")
    print(f"  - Training set size:   {len(X_train)}")
    print(f"  - Validation set size: {len(X_val)}")
    print(f"  - Test set size:       {len(X_test)}")
    print(f"  - Test dates found:    {len(dates_test)}")

    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_val_t = torch.tensor(X_val, dtype=torch.float64)
    y_val_t = torch.tensor(y_val, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    if config.MODEL_TYPE not in ['GraphWaveNet', 'DSTAGNN', 'HSDGNN'] and config.PREDICTION_MODE == "POINT":
        if y_train_t.dim() == 3 and y_train_t.shape[2] == 1:
            y_train_t = y_train_t.squeeze(-1)
            y_val_t = y_val_t.squeeze(-1)
            y_test_t = y_test_t.squeeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Shuffling is disabled for all models to maintain chronological order
    shuffle_train = False
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nPyTorch DataLoaders created with batch size: {config.BATCH_SIZE}")
    print(f"  - Training data shuffling: {shuffle_train}")

    return train_loader, val_loader, X_test_t, y_test_t, stock_ids_test, dates_test