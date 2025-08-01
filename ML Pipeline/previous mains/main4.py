# main.py

import torch
from pipeline import stage_1_feature_extraction_re, stage_3_data_partitioning1
from src.pipeline import (
    stage_2_data_preprocessing,
    stage_4_model_development,
    stage_5_model_evaluation
)
from src import config # Your configuration file

def main():
    # Stage 1: Load and combine data
    combined_df = stage_1_feature_extraction_re.run(config)

    # Stage 2: Preprocess data. This now returns the adjacency matrix.
    X, y, stock_ids, scalers, adj_matrix = stage_2_data_preprocessing.run(combined_df, config)

    # Stage 3: Split data into train/val/test sets
    train_loader, val_loader, X_test_t, y_test_t, stock_ids_test = stage_3_data_partitioning1.run(X, y, stock_ids, config)

    # --- START OF FIX ---
    # The adjacency matrix needs to be passed to stage 4 for any graph-based model
    supports = None
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        if adj_matrix is not None:
            supports = [adj_matrix] # Wrap the matrix in a list
        else:
            raise ValueError(f"{config.MODEL_TYPE} requires an adjacency matrix, but none was generated.")
    # --- END OF FIX ---

    # Stage 4: Train the model, passing the supports
    model, training_time = stage_4_model_development.run(train_loader, val_loader, config, supports=supports)

    # Stage 5: Evaluate the model
    stage_5_model_evaluation.run(model, X_test_t, y_test_t, stock_ids_test, scalers, config, adj_matrix=adj_matrix)

if __name__ == '__main__':
    main()
