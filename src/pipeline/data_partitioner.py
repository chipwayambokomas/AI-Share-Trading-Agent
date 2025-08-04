import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from ..utils import print_header

def run(X, y, stock_ids,dates, is_graph_model: bool, settings):
    """
    Partitions data into training, validation, and test sets.
    - Uses chronological split for graph models.
    - Uses stratified split for non-graph (DNN) models.
    """
    print_header("Stage 3: Data Partitioning")
    dates_test = None
    if is_graph_model:
        print("Method: Chronological split for graph-based model.")
        num_samples = X.shape[0]
        #get the indices for train, val, and test splits
        train_end_idx = int(num_samples * settings.TRAIN_SPLIT)
        val_end_idx = train_end_idx + int(num_samples * settings.VAL_SPLIT)
        # Split the data
        X_train, y_train = X[:train_end_idx], y[:train_end_idx]
        X_val, y_val = X[train_end_idx:val_end_idx], y[train_end_idx:val_end_idx]
        X_test, y_test = X[val_end_idx:], y[val_end_idx:]
        
        if dates is not None: 
            dates_train, dates_val, dates_test = dates[:train_end_idx], dates[train_end_idx:val_end_idx], dates[val_end_idx:]
        
        # For graph models, test_stock_ids is the full ordered list of nodes
        test_stock_ids = stock_ids

    else: # Stratified split for TCN, MLP
        print("Method: Stratified split by StockID for non-graph model.")
        # Calculate the test split size based on the remaining data after train and validation splits
        test_split_size = 1.0 - (settings.TRAIN_SPLIT + settings.VAL_SPLIT)
        if dates is not None: 
            #First, split the data into train+val and test, we stratify to ensures that all stock types are proportionally represented in both sets
            X_train_val, X_test, y_train_val, y_test, stock_ids_train_val, test_stock_ids,dates_train_val,dates_test = train_test_split(
                X, y, stock_ids,dates,
                test_size=test_split_size,
                random_state=settings.RANDOM_SEED,
                stratify=stock_ids
            )
            # Now split the train+val into train and validation sets
            relative_val_split = settings.VAL_SPLIT / (settings.TRAIN_SPLIT + settings.VAL_SPLIT)
            X_train, X_val, y_train, y_val,dates_train,dates_val = train_test_split(
                X_train_val, y_train_val, dates_train_val,
                test_size=relative_val_split,
                random_state=settings.RANDOM_SEED,
                stratify=stock_ids_train_val
            )
        else: 
                #First, split the data into train+val and test, we stratify to ensures that all stock types are proportionally represented in both sets
                X_train_val, X_test, y_train_val, y_test, stock_ids_train_val, test_stock_ids= train_test_split(
                X, y, stock_ids,
                test_size=test_split_size,
                random_state=settings.RANDOM_SEED,
                stratify=stock_ids
                )
                # Now split the train+val into train and validation sets
                relative_val_split = settings.VAL_SPLIT / (settings.TRAIN_SPLIT + settings.VAL_SPLIT)
                X_train, X_val, y_train, y_val= train_test_split(
                X_train_val, y_train_val,
                test_size=relative_val_split,
                random_state=settings.RANDOM_SEED,
                stratify=stock_ids_train_val
                ) 
            

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

    # For non-graph point prediction, y has shape (samples, out_win, 1). Squeeze it.
    #Some models output a target shape like (samples, out_window, 1) — the last dimension is unnecessary for MLP or TCN -> squeeze(-1) removes that extra dimension if it's size 1, making it easier to train.
    if not is_graph_model and settings.PREDICTION_MODE == "POINT":
        if y_train_t.dim() == 3 and y_train_t.shape[2] == 1:
            y_train_t, y_val_t, y_test_t = [t.squeeze(-1) for t in (y_train_t, y_val_t, y_test_t)]

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    #create data loaders, if it is not a graph model, we shuffle the training data ->
    shuffle_train = not is_graph_model
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    print(f"\nPyTorch DataLoaders created with batch size: {settings.BATCH_SIZE}")
    print(f"  - Training data shuffling: {shuffle_train}")

    return train_loader, val_loader, X_test_t, y_test_t, test_stock_ids,dates_test