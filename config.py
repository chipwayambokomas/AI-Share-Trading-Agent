# src/config.py

# --- Global Configuration ---
FILE_PATH = "JSE_Top40_OHLCV_2014_2024.xlsx"
# Define the target we want to predict
TARGET_COLUMN = 'close'
# --- NEW: Define all columns to be used as features ---
# The TARGET_COLUMN must be one of these.
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
RANDOM_SEED = 42

# --- MASTER SWITCH: Choose the prediction mode ---
# "POINT" -> Predict the next price point using multiple features.
# "TREND" -> Segment data into trends and predict the next trend's slope/duration.
PREDICTION_MODE = "POINT" #"TREND" #"POINT" # <-- CHANGE THIS to "POINT" or "TREND"

# --- POINT PREDICTION Configuration (used if PREDICTION_MODE is "POINT") ---
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 1

# --- TREND PREDICTION Configuration (used if PREDICTION_MODE is "TREND") ---
# We define how many data points, on average, should make up one trend.
# This still operates on the single TARGET_COLUMN.
AVG_POINTS_PER_TREND = 75
# This defines how many past trends the model looks at to predict the next one.
TREND_INPUT_WINDOW_SIZE = 10

# --- Data Partitioning ---
# 60 training, 20 validation, 20 test split
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20

# --- Model Development ---
MODEL_TYPE = "TCN" # "TCN" or "MLP"

# --- Training Hyperparameters ---
LEARNING_RATE = 0.002
EPOCHS = 10
BATCH_SIZE = 32

# --- Model-Specific Hyperparameters ---
# The `input_channels` are now set dynamically based on the prediction mode and feature count.
MODEL_ARGS = {
    "TCN": {
        "input_channels": len(FEATURE_COLUMNS) if PREDICTION_MODE == "POINT" else 2,
        "num_channels": [32, 32, 32, 32],
        "kernel_size": 3,
        "dropout": 0.2
    },
    "MLP": {
        "input_channels": len(FEATURE_COLUMNS) if PREDICTION_MODE == "POINT" else 2,
        "hidden_layers": [128, 64],
        "dropout": 0.2
    }
}