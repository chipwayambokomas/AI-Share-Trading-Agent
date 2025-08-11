# --- Global Configuration ---
FILE_PATH = "data/JSE_Top40_OHLCV_2014_2024.xlsx"
RESULTS_DIR = "results"
TARGET_COLUMN = 'vwap'
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
RANDOM_SEED = 42

# --- MASTER SWITCH: Choose the prediction mode ---
PREDICTION_MODE = "POINT" # "POINT" or "TREND"

# --- MASTER SWITCH: Choose the model type ---
MODEL_TYPE = "AGCRN" # "TCN", "MLP", "GraphWaveNet", "AGCRN", "DSTAGNN"

# --- Data & Preprocessing ---
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 5
TREND_INPUT_WINDOW_SIZE = 60
TREND_OUTPUT_WINDOW_SIZE = 1
MAX_SEGMENTATION_ERROR = 60.0
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 32

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
        "nb_block": 2,           # Reduced from 4 - less blocks for 60 timesteps to avoid overfitting
        "K": 3,                  # Keep as 3 (good for spatial attention heads)
        "nb_chev_filter": 16,    # Increased from 5 to 16 (more stable)
        "nb_time_filter": 16,    # Increased from 5 to 16 (matches chev_filter)
        "n_heads": 4,            # Changed from 3 to 4 (divides evenly into d_model)
        "d_k": 16,               # Reduced from 32 to 16 (smaller attention dimensions)
        "d_model": 64,           # Drastically reduced from 512 to 64 (much more reasonable)
    },
}
