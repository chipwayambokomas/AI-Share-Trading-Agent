# src/models/__init__.py

# Import the main model classes from their respective files
from .tcn import TCN_Forecaster
from .mlp import MLP_Forecaster
from .graphwavenet import GraphWaveNet
from .dstagnn import DSTAGNN, make_dstagnn_model
from .hsdgnn import HSDGNN  # Import HSDGNN model
# You can optionally define __all__ to specify what gets imported with 'from . import *'
__all__ = [
    'TCN_Forecaster',
    'MLP_Forecaster',
    'GraphWaveNet',
    'DSTAGNN',
    'HSDGNN'
]