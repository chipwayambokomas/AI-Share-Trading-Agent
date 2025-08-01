# main.py

import torch
import numpy as np
import time
import multiprocessing as mp
#from pipeline import stage_1_feature_extraction, stage_2_data_preprocessing, stage_3_data_partitioning, stage_4_model_development, stage_5_model_evaluation
#from pipeline import stage_2_data_preprocessing3
from pipeline import stage_1_feature_extraction_re, stage_3_data_partitioning_gwn_working, stage_4_model_development_gwn_works, stage_5_model_evaluation_trend_line_works
from src import config
from src.pipeline import (
    stage_2_data_preprocessing
)

def main():
    """
    Orchestrates the entire machine learning pipeline.
    This pipeline can be configured for two modes via config.py:
    1. TREND PREDICTION: Segments data into trends and predicts the next
       trend's slope and duration.
    2. POINT PREDICTION: Predicts the next price point using a sliding window
       of multiple features (e.g., OHLCV).
    """
    pipeline_start_time = time.time()

    print("="*80)
    print("## JSE STOCK PREDICTION PIPELINE ##")
    print("="*80)
    print(f"Running pipeline in '{config.PREDICTION_MODE}' mode with model: {config.MODEL_TYPE}")

    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    print(f"\n[INFO] Random seed set to {config.RANDOM_SEED} for reproducibility.")

    # --- TIME: DATA LOADING & PREPROCESSING ---
    data_start_time = time.time()

    # --- STAGE 1: Feature Extraction ---
    df = stage_1_feature_extraction_re.run(config)

    # --- STAGE 2: Data Preprocessing ---
    X, y, stock_ids, scalers = stage_2_data_preprocessing.run(df, config)

    data_duration = time.time() - data_start_time
    # --- END TIME ---

    # --- STAGE 3: Data Partitioning ---
    train_loader, val_loader, X_test_t, y_test_t, stock_ids_test = stage_3_data_partitioning_gwn_working.run(X, y, stock_ids, config)

    # --- STAGE 4: Model Development & Training ---
    model, training_duration = stage_4_model_development_gwn_works.run(train_loader, val_loader, config)

    # --- STAGE 5: Model Evaluation ---
    #stage_5_model_evaluation1.run(model, X_test_t, y_test_t, stock_ids_test, scalers, config)
    test_data = (X_test_t, y_test_t, stock_ids_test)
    stage_5_model_evaluation_trend_line_works.run(model, *test_data, scalers, config)

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