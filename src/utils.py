import pandas as pd
import numpy as np
import torch
import logging

def print_header(title: str):
    """Prints a formatted header."""
    print("\n" + "=" * 80)
    print(f"## {title.upper()} ##")
    print("=" * 80)

def setup_logging():
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocessing.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured.")

def create_adjacency_matrix(df: pd.DataFrame, target_column: str, threshold: float) -> torch.Tensor:
    """
    Creates a sparse adjacency matrix from stock data based on price correlation.
    """
    print_header("Adjacency Matrix Creation")
    print(f"Creating adjacency matrix based on '{target_column}' correlation with threshold {threshold}...")

    # Pivot the DataFrame to have stocks as columns and dates as index
    price_pivot = df.pivot(index='Date', columns='StockID', values=target_column)

    # Calculate the Pearson correlation matrix
    correlation_matrix = price_pivot.corr()

    # Create the adjacency matrix based on the threshold
    adj_matrix = (correlation_matrix.abs() >= threshold).astype(int)

    # Set the diagonal to 0
    np.fill_diagonal(adj_matrix.values, 0)
    
    # Convert to a PyTorch tensor, this is a format used for machine learning models
    adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float32)

    num_nodes = adj_matrix_tensor.shape[0]
    num_edges = torch.sum(adj_matrix_tensor).item() / 2 # Each edge is counted twice
    print(f"Adjacency matrix created with shape: {adj_matrix_tensor.shape}")
    print(f"Number of nodes (stocks): {num_nodes}")
    print(f"Number of edges (connections): {int(num_edges)}")
    
    return adj_matrix_tensor