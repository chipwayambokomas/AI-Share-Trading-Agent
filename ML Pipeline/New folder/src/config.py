# src/config.py

# --- Global Configuration ---
FILE_PATH = "JSE_Top40_OHLCV_2014_2024.xlsx"
TARGET_COLUMN = 'vwap'
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
RANDOM_SEED = 42

# --- MASTER SWITCH: Choose the prediction mode ---
PREDICTION_MODE = "POINT" # "POINT" or "TREND"

# --- POINT PREDICTION Configuration ---
POINT_INPUT_WINDOW_SIZE = 20
POINT_OUTPUT_WINDOW_SIZE = 5

#Default 
"""
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 1

#Single Step and Multi Step Prediction Experiments
"""
#Experiment 1 
"""
POINT_INPUT_WINDOW_SIZE = 30
POINT_OUTPUT_WINDOW_SIZE = 1
"""
#Experiment 2 
"""
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 1
"""
#Experiment 3 
"""
POINT_INPUT_WINDOW_SIZE = 120
POINT_OUTPUT_WINDOW_SIZE = 1
"""
#Experiment 4 
"""
POINT_INPUT_WINDOW_SIZE = 20
POINT_OUTPUT_WINDOW_SIZE = 5
"""
#Experiment 5
"""
POINT_INPUT_WINDOW_SIZE = 20
POINT_OUTPUT_WINDOW_SIZE = 10
"""
#Experiment 6
"""
POINT_INPUT_WINDOW_SIZE = 20
POINT_OUTPUT_WINDOW_SIZE = 20
"""
#Experiment 7
"""
POINT_INPUT_WINDOW_SIZE = 40
POINT_OUTPUT_WINDOW_SIZE = 5
"""
#Experiment 8
"""
POINT_INPUT_WINDOW_SIZE = 40
POINT_OUTPUT_WINDOW_SIZE = 10
"""
#Experiment 9
"""
POINT_INPUT_WINDOW_SIZE = 40
POINT_OUTPUT_WINDOW_SIZE = 20
"""
#Experiment 10
"""
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 5
"""
#Experiment 11
"""
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 10
"""
#Experiment 12
"""
POINT_INPUT_WINDOW_SIZE = 60
POINT_OUTPUT_WINDOW_SIZE = 20
"""

# --- TREND PREDICTION Configuration ---
#AVG_POINTS_PER_TREND = 75
TREND_INPUT_WINDOW_SIZE = 10 #10 
#SEGMENTATION_PENALTY = 1.0
MAX_SEGMENTATION_ERROR = 1.0 #20000000.0 #600000.0 = 4 days 

# --- Data Partitioning ---
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20

# --- Model Development ---
MODEL_TYPE = "MLP" # "TCN", "MLP", "GraphWaveNet", "DSTAGNN", "HSDGNN"

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001 # same as kialan paper 0.001 # Adjusted for DSTAGNN
#EPOCHS = 2
#BATCH_SIZE = 32
#TCN 
EPOCHS = 2
BATCH_SIZE = 64

# --- Model-Specific Hyperparameters ---
MODEL_ARGS = {
    "TCN": {
        "num_channels": [8, 8, 8, 8], # "num_channels": [32, 32, 32, 32],
        "kernel_size": 4, # "kernel_size": 3,
        "dropout": 0.2 # "dropout": 0.2
    },
    "MLP": {
        "hidden_layers": [16, 10],  # Changed from [128, 64] to match paper
        "dropout": 0.0,             # Changed from 0.2 to 0.0 (no dropout as per paper)
        "activation": "tanh",       # Specify tanh activation for hidden layers
        "output_activation": "linear" # Linear activation for output layer
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
    "DSTAGNN": {
        "nb_block": 2,
        "K": 3,
        "nb_chev_filter": 64,
        "time_strides": 1,
        "d_model": 64,
        "d_k": 32,
        "d_v": 32,
        "n_heads": 4
    },
    # --- ADD THIS SECTION FOR HSDGNN ---
    "HSDGNN": {
         # rnn_units: The number of hidden units in the GRU cells within the model.
        'rnn_units': 64,
        
        # embed_dim: The dimensionality of the node and time embeddings.
        'embed_dim': 10,
        
        # steps_per_day: The number of time steps in a single day. This is crucial
        # for creating the time-of-day embedding.
        # Example: For 5-minute data, there are (24 hours * 60 minutes) / 5 = 288 steps per day.
        'steps_per_day': 288 
    }
}