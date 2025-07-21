# --- Global Configuration ---
FILE_PATH = "data/JSE_Top40_OHLCV_2014_2024.xlsx"
RESULTS_DIR = "results"
TARGET_COLUMN = 'close'
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
RANDOM_SEED = 42

# --- MASTER SWITCH: Choose the prediction mode ---
PREDICTION_MODE = "POINT" # "POINT" or "TREND"

# --- MASTER SWITCH: Choose the model type ---
MODEL_TYPE = "MLP" # "TCN", "MLP", "GraphWaveNet",

# --- Data & Preprocessing ---
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 1
TREND_INPUT_WINDOW_SIZE = 10
SEGMENTATION_PENALTY = 1.0
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 32

# --- Adjacency Matrix ---
CORRELATION_THRESHOLD = 0.75 # For static adjacency matrix

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
}