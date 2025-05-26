# data/data.py

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from dataset import TimeSeriesDataset
import numpy as np
import os


def load_data(config: dict):
    """
    Loads and preprocesses data based on the configuration.
    Scaler is fit ONLY on the training data.
    """
    
    # data_loader variable removed as it was unused and invalid
    # Example: Initialize a 2D array (numpy) of zeros with shape (rows, cols)
    data_loaders = [[None for _ in range(4)] for _ in range(38)]
       
    # --- 1. Load Raw Data ---
    data_path = config["data"]["path"]
    project_root = config.get("project_root_dir", ".")
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    try:
        df = pd.read_excel(data_path, sheet_name=None)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
    
    count = 0  
    for sheet_name, df in df.items():
        # --- 2. Select Target Column ---
        if df.columns[1] not in df.columns:
            print(f"Warning: Target column not found.")
            return None, None, None, None
        
        # Extract raw values (before scaling)
        raw_values = df.iloc[:, 1].values.reshape(-1, 1) # Keep as 2D for scaler

        # --- 3. Split Data into Training, Validation, and Test Sets (RAW values) ---
        total_len = len(raw_values)
        train_split = config["data"]["split"]["train"]
        val_split = config["data"]["split"]["val"]
        
        if not (0 < train_split < 1 and 0 <= val_split < 1 and (train_split + val_split) <= 1):
            raise ValueError("Train and Val splits must be valid proportions (0 to 1), and their sum at most 1.")

        train_end_idx = int(total_len * train_split)
        val_end_idx = int(total_len * (train_split + val_split))

        raw_train_series = raw_values[:train_end_idx]
        raw_val_series = raw_values[train_end_idx:val_end_idx]
        raw_test_series = raw_values[val_end_idx:]

        # --- 4. Normalize Data (Fit on Train, Transform All) ---
        scaler = MinMaxScaler()

        # Fit the scaler ONLY on the training data
        # Ensure raw_train_series is not empty before fitting
        if len(raw_train_series) == 0:
            raise ValueError("Training data is empty after split. Cannot fit scaler. Check data size and splits.")
        
        scaler.fit(raw_train_series)

        # Transform all sets using the scaler fitted on the training data
        scaled_train_series = scaler.transform(raw_train_series).flatten()
        
        # Handle empty validation or test sets before transforming
        if len(raw_val_series) > 0:
            scaled_val_series = scaler.transform(raw_val_series).flatten()
        else:
            scaled_val_series = np.array([]) # Empty array if raw_val_series is empty
            
        if len(raw_test_series) > 0:
            scaled_test_series = scaler.transform(raw_test_series).flatten()
        else:
            scaled_test_series = np.array([]) # Empty array if raw_test_series is empty

        # --- 5. Create PyTorch Datasets ---
        input_size = config["model"]["input_size"]

        train_ds = TimeSeriesDataset(scaled_train_series, input_size)
        val_ds = TimeSeriesDataset(scaled_val_series, input_size) # Will be empty if scaled_val_series is empty
        test_ds = TimeSeriesDataset(scaled_test_series, input_size) # Will be empty if scaled_test_series is empty

        if len(train_ds) == 0: 
            print(f"Warning: Training dataset is empty after sequence creation. "
                f"Raw train points: {len(raw_train_series)}, Input size: {input_size}.")
        if len(val_ds) == 0 and len(raw_val_series) > 0: # Warn if raw val data existed but no sequences formed
            print(f"Warning: Validation dataset is empty after sequence creation. "
                f"Raw val points: {len(raw_val_series)}, Input size: {input_size}.")
        # Similar warning for test_ds could be added if desired, or handled in eval

        # --- 6. Create PyTorch DataLoaders ---
        batch_size = config["training"]["batch_size"]
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                drop_last=True if len(train_ds) >= batch_size else False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=1, drop_last=False)
        
        data_loaders[count] = [sheet_name,train_loader, val_loader, test_loader]
        count += 1
        print(f"Loaded and processed data from sheet: {sheet_name}")

    # --- 7. Return Processed Data ---
    return data_loaders, scaler