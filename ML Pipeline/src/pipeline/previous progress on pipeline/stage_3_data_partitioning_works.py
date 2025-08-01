# src/pipeline/stage_3_data_partitioning.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils import print_header

def run(X, y, stock_ids, config):
    """
    Partitions the data into training, validation, and test sets, then creates
    PyTorch DataLoaders. This version uses stratified splitting to ensure each
    stock is represented proportionally in each data split.

    Args:
        X (np.ndarray): The input features.
        y (np.ndarray): The target labels.
        stock_ids (np.ndarray): Array of stock IDs corresponding to each sequence.
        config: The configuration module.

    Returns:
        Tuple: A tuple containing:
            - train_loader (DataLoader)
            - val_loader (DataLoader)
            - X_test_t (torch.Tensor)
            - y_test_t (torch.Tensor)
            - stock_ids_test (np.ndarray)
    """
    print_header("Stage 3: Data Partitioning")
    print(f"""
Purpose: Split data into training, validation, and test sets.
- Method: Stratified split based on StockID.
- Train Split: {config.TRAIN_SPLIT * 100:.0f}%
- Validation Split: {config.VAL_SPLIT * 100:.0f}%
- Test Split: {(1 - config.TRAIN_SPLIT - config.VAL_SPLIT) * 100:.0f}%
Executing Code: Partitioning data and creating PyTorch DataLoaders...
    """)

    # --- 1. First Split: Separate Test Set ---
    # We first split off the test set. The remaining data will be split into train/val.
    test_split_size = 1.0 - config.TRAIN_SPLIT
    
    # Create indices for splitting
    indices = np.arange(len(X))
    
    X_train_val, X_test, y_train_val, y_test, stock_ids_train_val, stock_ids_test = train_test_split(
        X, y, stock_ids,
        test_size=test_split_size,
        random_state=config.RANDOM_SEED,
        stratify=stock_ids  # Stratify ensures proportional representation of stocks
    )

    # --- 2. Second Split: Separate Train and Validation Sets ---
    # The validation split size needs to be recalculated relative to the new train_val set.
    relative_val_split = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_split,
        random_state=config.RANDOM_SEED,
        stratify=stock_ids_train_val # Stratify again on the smaller set
    )

    print(f"\nData partitioning complete:")
    print(f"  - Training set size:   {len(X_train)}")
    print(f"  - Validation set size: {len(X_val)}")
    print(f"  - Test set size:       {len(X_test)}")

    # --- 3. Convert to PyTorch Tensors ---
    # Using float32 is standard for deep learning.
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # For POINT mode, the y-tensors might have an extra dimension. Squeeze it.
    if config.PREDICTION_MODE == "POINT":
        y_train_t = y_train_t.squeeze(-1)
        y_val_t = y_val_t.squeeze(-1)
        y_test_t = y_test_t.squeeze(-1)

    # --- 4. Create DataLoaders ---
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nPyTorch DataLoaders created with batch size: {config.BATCH_SIZE}")

    return train_loader, val_loader, X_test_t, y_test_t, stock_ids_test
