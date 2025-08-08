from .base_handler import BaseModelHandler
from .architectures.tcn import TCN_Forecaster
from .architectures.mlp import MLP_Forecaster
import torch.nn as nn

def _trend_loss_function(predicted, actual, mse_criterion):
        """
        Calculates a combined MSE loss for the two trend features: slope and duration.
        This is specific to non-graph models in TREND mode.
        """
        # predicted/actual shape: (batch_size, 2) where columns are [slope, duration]
        loss_slope = mse_criterion(predicted[:, 0], actual[:, 0])
        loss_duration = mse_criterion(predicted[:, 1], actual[:, 1])
        return (loss_slope + loss_duration) / 2.0

class DNNHandler(BaseModelHandler):
    """Base handler for non-graph DNN models like TCN and MLP."""
    def is_graph_based(self) -> bool:
        return False

    def get_loss_function(self):
        """
        Gets the appropriate loss function for DNN models.
        - For TREND mode, it returns the custom combined loss function.
        - For POINT mode, it returns the standard MSE loss.
        """
        if self.settings.PREDICTION_MODE == "TREND":
            # For TREND mode, we use the special combined loss function.
            # We use a lambda function to wrap it with an nn.MSELoss instance.
            return lambda pred, act: _trend_loss_function(pred, act, nn.MSELoss())
        else: # POINT mode
            # For POINT mode, standard MSE is correct.
            return nn.MSELoss()
    def adapt_output_for_loss(self, y_pred, y_batch):
        print_once_loss = getattr(self, 'print_once_loss', True)
        if print_once_loss:
            print("\n--- Loss Shape Alignment Verification ---")
            print(f"Shape of y_pred (model output): {y_pred.shape}")
            print(f"Shape of y_batch (target):    {y_batch.shape}")
            setattr(self, 'print_once_loss', False)
        # The base handler just passes them through, which should be correct for DNNs
        return y_pred, y_batch

class TCNHandler(DNNHandler):
    def name(self) -> str:
        return "TCN"

# **kwargs means we can pass additional parameters like `supports` if needed.
    def build_model(self, **kwargs) -> TCN_Forecaster:
        #the input and output dimensions based on the prediction mode in the abstract base class -> how many features we have and how many outputs we want
        input_features, output_size = self.get_input_dims()
        
        #this updaes the model_args dictionary with the input and output sizes defined in the abstract base class based off of what we have set in the settings.py file
        self.model_args['input_channels'] = input_features
        
        self.model_args['output_size'] = output_size
        
        return TCN_Forecaster(**self.model_args)

class MLPHandler(DNNHandler):
    def name(self) -> str:
        return "MLP"
        
    def build_model(self, **kwargs) -> MLP_Forecaster:
        input_features, output_size = self.get_input_dims()
        input_window = self.get_window_size()
        self.model_args['input_size'] = input_window * input_features
        self.model_args['output_size'] = output_size
        return MLP_Forecaster(**self.model_args)