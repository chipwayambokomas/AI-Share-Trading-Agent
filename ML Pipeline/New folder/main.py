# main.py (Corrected with if/else logic)

import torch
import importlib
import warnings
from src.pipeline import (
    stage_1_feature_extraction,
    stage_3_data_partitioning,
    stage_4_model_development,
    stage_5_model_evaluation
)
from src import config
torch.set_default_dtype(torch.float64) #For floating point precision
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """
    Executes the entire machine learning pipeline from data ingestion to evaluation.
    """
    # --- Define Preprocessing Modules ---
    preprocessing_module_mapping = {
        ("POINT", "TCN"): "src.pipeline.stage_2_data_preprocessing.dnn_point_prediction",
        ("POINT", "MLP"): "src.pipeline.stage_2_data_preprocessing.dnn_point_prediction",
        ("POINT", "GraphWaveNet"): "src.pipeline.stage_2_data_preprocessing.gnn_point_prediction",
        ("POINT", "DSTAGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_point_prediction",
        ("POINT", "HSDGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_point_prediction",
        ("TREND", "TCN"): "src.pipeline.stage_2_data_preprocessing.dnn_trendline_prediction",
        ("TREND", "MLP"): "src.pipeline.stage_2_data_preprocessing.dnn_trendline_prediction",
        ("TREND", "GraphWaveNet"): "src.pipeline.stage_2_data_preprocessing.gnn_trendline_prediction",
        ("TREND", "DSTAGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_trendline_prediction",
        ("TREND", "HSDGNN"): "src.pipeline.stage_2_data_preprocessing.stgnn_trendline_prediction",
    }

    # --- Stage 1: Data Ingestion & Feature Extraction ---
    combined_df = stage_1_feature_extraction.run(config)

    # --- Stage 2: Dynamic Data Preprocessing ---
    try:
        module_path = preprocessing_module_mapping[(config.PREDICTION_MODE, config.MODEL_TYPE)]
        stage_2_module = importlib.import_module(module_path)
    except KeyError:
        raise ValueError(f"No preprocessing module found for MODE='{config.PREDICTION_MODE}' and MODEL='{config.MODEL_TYPE}'")

    # --- START OF FIX ---
    # Conditionally unpack variables based on the model type.
    # Non-graph models (TCN, MLP) return 5 values.
    # Graph models return 6 values.
    if config.MODEL_TYPE in ['TCN', 'MLP']:
        # This module returns 5 values, so we unpack 5 and set adj_matrix to None.
        X, y, stock_ids, dates, scalers = stage_2_module.run(combined_df, config)
        adj_matrix = None # Manually set the 6th variable
    else:
        # This module returns all 6 values.
        X, y, stock_ids, dates, scalers, adj_matrix = stage_2_module.run(combined_df, config)
    # --- END OF FIX ---

    # --- Stage 3: Data Partitioning ---
    # Pass the dates array into the partitioning function and get the test dates back.
    train_loader, val_loader, X_test_t, y_test_t, stock_ids_test, dates_test = stage_3_data_partitioning.run(
        X, y, stock_ids, dates, config
    )

    # --- Stage 4: Model Development ---
    # Prepare the adjacency matrix ('supports') only for the models that need it.
    supports = None
    if config.MODEL_TYPE in ['GraphWaveNet', 'DSTAGNN']:
        if adj_matrix is not None:
            supports = [torch.tensor(adj_matrix, dtype=torch.float64)] # Ensure it's a tensor and changed from float32 to float64
        else:
            raise ValueError(f"{config.MODEL_TYPE} requires an adjacency matrix, but none was generated.")

    model, training_time = stage_4_model_development.run(train_loader, val_loader, config, supports=supports)

    # --- Stage 5: Model Evaluation ---
    stage_5_model_evaluation.run(
        model,
        X_test_t,
        y_test_t,
        stock_ids_test,
        dates_test,  # Pass the test dates for the final report
        scalers,
        config,
        adj_matrix=adj_matrix
    )

if __name__ == '__main__':
    main()