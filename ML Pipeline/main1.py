# main.py (Updated for new folder name)

import torch
import importlib
from pipeline import stage_1_feature_extraction_re, stage_3_data_partitioning1
from src.pipeline import (
    stage_4_model_development,
    stage_5_model_evaluation
)
from src import config

def main():
    # --- START OF CHANGE ---
    # Update the path from "src.preprocessing" to "src.stage_2_data_preprocessing"
    preprocessing_module_mapping = {
        ("POINT", "TCN"): "src.pipeline.stage_2_data_preprocessing.dnn_point_prediction",
        ("POINT", "MLP"): "src.pipeline.stage_2_data_preprocessing.dnn_point_prediction",
        ("POINT", "GraphWaveNet"): "src.pipeline.stage_2_data_preprocessing.gnn_point_prediction",
        ("POINT", "DSTAGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_point_prediction",
        ("TREND", "TCN"): "src.pipeline.stage_2_data_preprocessing.dnn_trendline_prediction",
        ("TREND", "MLP"): "src.pipeline.stage_2_data_preprocessing.dnn_trendline_prediction",
        ("TREND", "GraphWaveNet"): "src.pipeline.stage_2_data_preprocessing.gnn_trendline_prediction",
        ("TREND", "DSTAGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_trendline_prediction",
    }
    # --- END OF CHANGE ---

    # Stage 1: Load and combine data
    combined_df = stage_1_feature_extraction_re.run(config)

    # Stage 2: Dynamically select and run the correct preprocessing module
    try:
        module_path = preprocessing_module_mapping[(config.PREDICTION_MODE, config.MODEL_TYPE)]
        stage_2_module = importlib.import_module(module_path)
    except KeyError:
        raise ValueError(f"No preprocessing module found for MODE='{config.PREDICTION_MODE}' and MODEL='{config.MODEL_TYPE}'")

    X, y, stock_ids, scalers, adj_matrix = stage_2_module.run(combined_df, config)

    # The rest of the file remains exactly the same.
    # Stage 3: Split data into train/val/test sets
    train_loader, val_loader, X_test_t, y_test_t, stock_ids_test = stage_3_data_partitioning1.run(X, y, stock_ids, config)

    # Prepare adjacency matrix for graph models
    supports = None
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        if adj_matrix is not None:
            supports = [adj_matrix]
        else:
            raise ValueError(f"{config.MODEL_TYPE} requires an adjacency matrix, but none was generated.")

    # Stage 4: Train the model
    model, training_time = stage_4_model_development.run(train_loader, val_loader, config, supports=supports)

    # Stage 5: Evaluate the model
    stage_5_model_evaluation.run(model, X_test_t, y_test_t, stock_ids_test, scalers, config, adj_matrix=adj_matrix)

if __name__ == '__main__':
    main()