# src/utils.py

def print_header(title):
    """Prints a formatted header for each pipeline stage."""
    print("\n" + "="*80)
    print(f"## {title.upper()} ##")
    print("="*80)

import pandas as pd
import numpy as np
import torch

def create_adjacency_matrix(df: pd.DataFrame, target_column: str, threshold: float = 0.75) -> torch.Tensor:
    """
    Creates a sparse adjacency matrix from stock data based on price correlation.

    Args:
        df (pd.DataFrame): DataFrame containing cleaned stock data with 'Date', 
                           'StockID', and the target_column.
        target_column (str): The name of the column to use for correlation 
                             (e.g., 'close').
        threshold (float): The absolute correlation value above which stocks are 
                           considered connected.

    Returns:
        torch.Tensor: A PyTorch tensor representing the adjacency matrix.
    """
    print_header("Adjacency Matrix Creation")
    print(f"Creating adjacency matrix based on '{target_column}' correlation...")
    print(f"Correlation threshold: {threshold}")

    # 1. Pivot the DataFrame to have stocks as columns and dates as index
    price_pivot = df.pivot(index='Date', columns='StockID', values=target_column)

    # 2. Calculate the Pearson correlation matrix
    correlation_matrix = price_pivot.corr()

    # 3. Create the adjacency matrix based on the threshold
    # Set values to 1 if absolute correlation is >= threshold, else 0
    adj_matrix = (correlation_matrix.abs() >= threshold).astype(int)

    # 4. Set the diagonal to 0 (a stock is not a neighbor of itself in this context)
    np.fill_diagonal(adj_matrix.values, 0)
    
    # Convert the pandas DataFrame to a PyTorch tensor
    adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float64)#changed from float32 to float64 for consistency with other tensors

    num_nodes = adj_matrix_tensor.shape[0]
    num_edges = torch.sum(adj_matrix_tensor).item() / 2 # Each edge is counted twice
    print(f"Adjacency matrix created with shape: {adj_matrix_tensor.shape}")
    print(f"Number of nodes (stocks): {num_nodes}")
    print(f"Number of edges (connections): {int(num_edges)}")
    
    return adj_matrix_tensor

