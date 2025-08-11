import os

# --- Global Configuration ---
FILE_PATH = "data/JSE_Top40_OHLCV_2014_2024.xlsx"

# Results directory - can be overridden by environment variable for unique job outputs
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

TARGET_COLUMN = 'vwap'
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
RANDOM_SEED = 42

# --- MASTER SWITCHES: Can be overridden by HPC job environment variables ---
PREDICTION_MODE = os.getenv("PREDICTION_MODE", "POINT")  # "POINT" or "TREND"
MODEL_TYPE = os.getenv("MODEL_TYPE", "AGCRN")  # "TCN", "MLP", "GraphWaveNet", "AGCRN", "DSTAGNN"

# --- Data & Preprocessing ---
# Point prediction windows - can be overridden by environment variables
POINT_INPUT_WINDOW_SIZE = int(os.getenv("POINT_INPUT_WINDOW_SIZE", "60"))
POINT_OUTPUT_WINDOW_SIZE = int(os.getenv("POINT_OUTPUT_WINDOW_SIZE", "1"))

# Trend prediction windows - can be overridden by environment variables  
TREND_INPUT_WINDOW_SIZE = int(os.getenv("TREND_INPUT_WINDOW_SIZE", "60"))

# Other preprocessing parameters
MAX_SEGMENTATION_ERROR = 60.0
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
EPOCHS = int(os.getenv("EPOCHS", "50"))  # Can be overridden by environment variable
BATCH_SIZE = 64

# --- Adjacency Matrix ---
CORRELATION_THRESHOLD = 0.75 # For static adjacency matrix
EVAL_THRESHOLD = 90

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
        "aptinit": None,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    },
     "AGCRN": {
        "rnn_units": 64,
        "num_layers": 2,
        "cheb_k": 2,
        "embed_dim": 10       
    },
     "DSTAGNN": {
        "nb_block": 4,           # Paper value
        "K": 3,                  # Paper value  
        "nb_chev_filter": 32,    # Paper value
        "nb_time_filter": 32,    # Paper value
        "n_heads": 3,            # Paper value
        "d_k": 32,               # Paper value
        "d_model": 512,          # Paper value
    },
}

# --- Debug information for HPC runs ---
if __name__ == "__main__":
    print("=== Current Configuration ===")
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"PREDICTION_MODE: {PREDICTION_MODE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"POINT_INPUT_WINDOW_SIZE: {POINT_INPUT_WINDOW_SIZE}")
    print(f"POINT_OUTPUT_WINDOW_SIZE: {POINT_OUTPUT_WINDOW_SIZE}")
    print(f"TREND_INPUT_WINDOW_SIZE: {TREND_INPUT_WINDOW_SIZE}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print("=============================")