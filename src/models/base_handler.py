# src/models/base_handler.py
from abc import ABC, abstractmethod
import torch.nn as nn

# Inherits from ABC to mark it as an abstract class.
class BaseModelHandler(ABC):
    def __init__(self, settings):
        # The handler stores the global settings and fetches its own specific arguments
        # from the `MODEL_ARGS` dictionary.
        self.settings = settings
        self.model_args = settings.MODEL_ARGS.get(self.name(), {})

    # The `@abstractmethod` decorator means any class that inherits from BaseModelHandler
    # MUST implement this method.
    @abstractmethod
    def name(self) -> str:
        # Must return the model's string name (e.g., "TCN").
        pass

    @abstractmethod
    def is_graph_based(self) -> bool:
        # Must return True or False. This is used by factories and pipeline stages.
        pass
        
    @abstractmethod
    def build_model(self, **kwargs) -> nn.Module:
        # Must return an instantiated PyTorch model (an `nn.Module` object).
        pass
    
    @abstractmethod
    def get_loss_function(self):
        """Returns the appropriate loss function for the model."""
        pass


    # These are helper methods with default implementations that can be used by all handlers.
    def get_input_dims(self):
        # Centralizes the logic for determining input/output features based on the mode.
        if self.settings.PREDICTION_MODE == "TREND":
            return 2, 2 #in trend line, you only wants to predict the trend and magnitude
        else:
            return len(self.settings.FEATURE_COLUMNS), self.settings.POINT_OUTPUT_WINDOW_SIZE #for point prediction, you want to predict the next price for a particlar amount of time steps for a specified input feature size in our case we are looking at 'open', 'high', 'low', 'close', 'vwap'
        
    def get_window_size(self):
        """Determines input window size based on prediction mode."""
        if self.settings.PREDICTION_MODE == "TREND":
            return self.settings.TREND_INPUT_WINDOW_SIZE
        # Defaults to POINT mode if not TREND
        return self.settings.POINT_INPUT_WINDOW_SIZE

    # Default methods that can be overridden by specific handlers if they have
    # unique requirements for handling data during the loss calculation.
    def adapt_output_for_loss(self, y_pred, y_batch):
        return y_pred, y_batch
        
    def adapt_y_for_loss(self, y_pred, y_batch):
        return y_pred, y_batch
    
    def adapt_input_for_model(self, X_batch):
        """
        Adapts the input batch to the shape expected by the specific model's forward pass.
        The default implementation does nothing.
        """
        return X_batch