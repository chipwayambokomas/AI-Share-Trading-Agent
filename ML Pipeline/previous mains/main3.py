# main.py

import torch
import numpy as np
import time
import multiprocessing as mp
import joblib # <-- Import joblib to load the scaler in main if needed, though it's passed directly
from pipeline import stage_1_feature_extraction_re, stage_2_data_preprocessing3, stage_3_data_partitioning1
from src import config
from src.pipeline import (
    stage_4_model_development,
    stage_5_model_evaluation
)

def main():
    """
    Orchestrates the entire machine learning pipeline.
    This pipeline can be configured for two modes via config.py:
    1. TREND PREDICTION: Segments data into trends and predicts the next
       trend's slope and duration for all stocks simultaneously.
    2. POINT PREDICTION: Predicts the next price point using a sliding window
       of multiple features for all stocks simultaneously.
    
    This pipeline is designed to work with both standard sequential models (TCN, MLP)
    and graph-based models (GraphWaveNet) by structuring the data as a panel/graph.
    """
    pipeline_start_time = time.time()

    print("="*80)
    print("## JSE STOCK PREDICTION PIPELINE ##")
    print("="*80)
    print(f"Running pipeline in '{config.PREDICTION_MODE}' mode with model: {config.MODEL_TYPE}")
    print(f"Using device: {config.DEVICE}")

    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    print(f"\n[INFO] Random seed set to {config.RANDOM_SEED} for reproducibility.")

    # --- TIME: DATA LOADING & PREPROCESSING ---
    data_start_time = time.time()

    # --- STAGE 1: Feature Extraction ---
    df = stage_1_feature_extraction_re.run(config)

    # --- STAGE 2: Data Preprocessing ---
    # This stage now produces RAW, unscaled panel data. The scaler is no longer returned here.
    X, y, stock_ids, _ = stage_2_data_preprocessing3.run(df, config)

    data_duration = time.time() - data_start_time
    # --- END TIME ---

    # --- STAGE 3: Data Partitioning & Normalization ---
    # This stage now handles the split and returns the fitted scaler.
    # --- START OF FIX ---
    # We now expect SIX return values from stage_3, so we add 'scalers' to the assignment.
    train_loader, val_loader, X_test_t, y_test_t, stock_ids_ordered, scalers = stage_3_data_partitioning1.run(X, y, stock_ids, config)
    # --- END OF FIX ---

    # --- STAGE 4: Model Development & Training ---
    # Pass stock_ids (node list) and config for model initialization
    model, training_duration = stage_4_model_development2.run(train_loader, val_loader, stock_ids_ordered, config)

    # --- STAGE 5: Model Evaluation ---
    # Pass the scalers dictionary from Stage 3 to Stage 5 for inverse transformation.
    stage_5_model_evaluation.run(model, X_test_t, y_test_t, stock_ids_ordered, scalers, config)

    pipeline_duration = time.time() - pipeline_start_time

    # --- FINAL PERFORMANCE SUMMARY ---
    print("\n" + "="*80)
    print("## PIPELINE PERFORMANCE SUMMARY ##")
    print("="*80)
    print(f"Data Loading & Preprocessing Time: {data_duration:.2f} seconds.")
    print(f"Model Training Time:               {training_duration:.2f} seconds.")
    print("-" * 40)
    print(f"Total Pipeline Execution Time:     {pipeline_duration:.2f} seconds.")
    print("="*80)


if __name__ == "__main__":
    # Set the start method to 'spawn' to avoid potential deadlocks on some systems.
    # This must be done inside the __name__ == '__main__' block.
    try:
        mp.set_start_method('spawn', force=True)
        print("\n[INFO] Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass # The context can only be set once.
    main()