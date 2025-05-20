# utils/utils.py

import os
import yaml # For load_config_from_path

def load_config_from_path(path): # Updated default
    """Load configuration from a specific YAML file path"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r") as file:
        return yaml.safe_load(file)