# src/config.py

# --- Global Configuration ---
FILE_PATH = "JSE_Top40_OHLCV_2014_2024.xlsx"
TARGET_COLUMN = 'close'
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
RANDOM_SEED = 42

# --- MASTER SWITCH: Choose the prediction mode ---
PREDICTION_MODE = "POINT" # "POINT" or "TREND"

# --- POINT PREDICTION Configuration ---
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 1 

# --- TREND PREDICTION Configuration ---
AVG_POINTS_PER_TREND = 75
TREND_INPUT_WINDOW_SIZE = 10
SEGMENTATION_PENALTY = 1.0

# --- Data Partitioning ---
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20

# --- Model Development ---
MODEL_TYPE = "GraphWaveNet" # "TCN", "MLP", "GraphWaveNet", "DSTAGNN"

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001 # Adjusted for DSTAGNN
EPOCHS = 2
BATCH_SIZE = 32

# --- Model-Specific Hyperparameters ---
MODEL_ARGS = {
    "TCN": {
        "num_channels": [32, 32, 32, 32],
        "kernel_size": 3,
        "dropout": 0.2
    },
    "MLP": {
        "hidden_layers": [128, 64],
        "dropout": 0.2
    },
    "GraphWaveNet": {
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    },
    # --- ADD THIS SECTION FOR DSTAGNN ---
    "DSTAGNN": {
        "nb_block": 2,
        "K": 3,
        "nb_chev_filter": 64,
        "time_strides": 1,
        "d_model": 64,
        "d_k": 32,
        "d_v": 32,
        "n_heads": 4
    }
}
