#!/bin/bash

# ==============================================================================
#  Master Experiment Automation Script
# ==============================================================================
#
#  This script automates the process of running a series of machine learning
#  experiments by:
#    1. Defining sets of parameters to test.
#    2. Modifying the 'src/config.py' file for each experiment.
#    3. Executing the main Python script ('main.py').
#    4. Saving the console output for each run to a unique log file.
#    5. Automatically backing up and restoring the original config file.
#
# ==============================================================================

# --- Script Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Experiment Parameters ---

# Define the models you want to test
MODELS=("TCN" "MLP" "GraphWaveNet" "DSTAGNN" "HSDGNN")

# --- POINT PREDICTION PARAMETERS ---
# Each element is a "INPUT_WINDOW/OUTPUT_WINDOW" pair
POINT_PARAMS=("30/1" "60/1" "120/1" "20/5" "20/10" "20/20" "40/5" "40/10" "40/20" "60/5" "60/10" "60/20")

# --- TREND PREDICTION PARAMETERS ---
# Define the input window sizes to test for trend prediction
TREND_INPUT_WINDOWS=("10") # Example: ("10" "20")

# Define the segmentation error thresholds to test
TREND_SEGMENTATION_ERRORS=("0.5" "1.0" "2.5") # Example: ("0.5" "1.0" "2.5")

# --- File and Environment Paths ---
CONFIG_FILE="src/config.py"
PYTHON_EXEC="myenv/bin/python" # IMPORTANT: Verify this path is correct

# --- Backup and Restore Logic ---

# Function to restore the original config file
restore_config() {
  if [ -f "${CONFIG_FILE}.bak" ]; then
    echo -e "\nRestoring original config file..."
    mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
  fi
}

# The 'trap' command ensures that the restore_config function is called
# when the script exits, whether normally, due to an error, or by interruption.
trap restore_config EXIT

# Backup the original config file before starting
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"
echo "Original config file backed up to ${CONFIG_FILE}.bak"


# ==============================================================================
#  Function to run POINT prediction experiments
# ==============================================================================
run_point_experiments() {
    echo -e "\n\n##################################################"
    echo "#         RUNNING POINT PREDICTION EXPERIMENTS         #"
    echo "##################################################"
    
    # Set PREDICTION_MODE to POINT in the config file
    sed -i 's/PREDICTION_MODE = .*/PREDICTION_MODE = "POINT"/' "$CONFIG_FILE"

    for model in "${MODELS[@]}"; do
        for params in "${POINT_PARAMS[@]}"; do
            # Split the params into input and output window sizes
            IFS='/' read -r in_win out_win <<< "$params"

            echo -e "\n--- Starting POINT: MODEL=${model}, IN=${in_win}, OUT=${out_win} ---"

            # 1. Modify config.py with the current parameters
            sed -i "s/MODEL_TYPE = .*/MODEL_TYPE = \"${model}\"/" "$CONFIG_FILE"
            sed -i "s/POINT_INPUT_WINDOW_SIZE = .*/POINT_INPUT_WINDOW_SIZE = ${in_win}/" "$CONFIG_FILE"
            sed -i "s/POINT_OUTPUT_WINDOW_SIZE = .*/POINT_OUTPUT_WINDOW_SIZE = ${out_win}/" "$CONFIG_FILE"
            
            # 2. Define the log file name
            LOG_FILE="experiment_log_POINT_${model}_${in_win}_${out_win}.log"
            
            # 3. Run the main script and redirect all output (stdout & stderr) to the log file
            echo "Running pipeline... Log will be saved to ${LOG_FILE}"
            $PYTHON_EXEC main.py > "$LOG_FILE" 2>&1
            
            echo "--- Finished: MODEL=${model}, IN=${in_win}, OUT=${out_win} ---"
        done
    done
}

# ==============================================================================
#  Function to run TREND prediction experiments
# ==============================================================================
run_trend_experiments() {
    echo -e "\n\n##################################################"
    echo "#         RUNNING TREND PREDICTION EXPERIMENTS         #"
    echo "##################################################"
    
    # Set PREDICTION_MODE to TREND in the config file
    sed -i 's/PREDICTION_MODE = .*/PREDICTION_MODE = "TREND"/' "$CONFIG_FILE"

    for model in "${MODELS[@]}"; do
        for in_win in "${TREND_INPUT_WINDOWS[@]}"; do
            for error_val in "${TREND_SEGMENTATION_ERRORS[@]}"; do
            
                echo -e "\n--- Starting TREND: MODEL=${model}, IN=${in_win}, MAX_ERROR=${error_val} ---"

                # 1. Modify config.py with the current parameters
                sed -i "s/MODEL_TYPE = .*/MODEL_TYPE = \"${model}\"/" "$CONFIG_FILE"
                sed -i "s/TREND_INPUT_WINDOW_SIZE = .*/TREND_INPUT_WINDOW_SIZE = ${in_win}/" "$CONFIG_FILE"
                sed -i "s/MAX_SEGMENTATION_ERROR = .*/MAX_SEGMENTATION_ERROR = ${error_val}/" "$CONFIG_FILE"
                
                # 2. Define the log file name
                LOG_FILE="experiment_log_TREND_${model}_IN${in_win}_ERR${error_val}.log"
                
                # 3. Run the main script and redirect output to the log file
                echo "Running pipeline... Log will be saved to ${LOG_FILE}"
                $PYTHON_EXEC main.py > "$LOG_FILE" 2>&1
                
                echo "--- Finished: MODEL=${model}, IN=${in_win}, MAX_ERROR=${error_val} ---"
            done
        done
    done
}


# ==============================================================================
#  Main Execution Block
# ==============================================================================
echo -e "\nStarting Experiment Automation Script at $(date)"

# Execute the experiment functions
run_point_experiments
run_trend_experiments

echo -e "\nExperiment Automation Script finished successfully at $(date)."
echo "====================================================="

# The 'trap' will automatically call 'restore_config' upon exit.