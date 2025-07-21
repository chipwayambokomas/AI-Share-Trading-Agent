from .base_handler import BaseModelHandler
from .architectures.tcn import TCN_Forecaster
from .architectures.mlp import MLP_Forecaster
import torch.nn as nn

class DNNHandler(BaseModelHandler):
    """Base handler for non-graph DNN models like TCN and MLP."""
    def is_graph_based(self) -> bool:
        return False

    def get_loss_function(self):
        """Gets the appropriate loss function for DNN models."""
        if self.settings.PREDICTION_MODE == "TREND":
            mse_criterion = nn.MSELoss()
            # For TREND mode, we average the MSE across two outputs (e.g., trend and magnitude).
            return lambda pred, act: (mse_criterion(pred[:, 0], act[:, 0]) + mse_criterion(pred[:, 1], act[:, 1])) / 2.0
        else: # POINT mode
            #the average of the squared differences between predicted and actual values
            return nn.MSELoss()

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