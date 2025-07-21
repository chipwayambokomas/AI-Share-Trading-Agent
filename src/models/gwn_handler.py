from .base_handler import BaseModelHandler
from .architectures.gwn import GraphWaveNet
import torch.nn as nn

class GraphWaveNetHandler(BaseModelHandler):
    def name(self) -> str:
        return "GraphWaveNet"

    def is_graph_based(self) -> bool:
        return True
        
    def build_model(self, **kwargs) -> GraphWaveNet:
        input_features, output_size = self.get_input_dims()
        
        #supports: This should be a list of adjacency matrices — they tell the model how the nodes (e.g., sensors or locations) are connected. If supports is missing or invalid, it throws an error because GraphWaveNet cannot run without this.
        
        supports = kwargs.get('supports')
        if supports is None or supports[0] is None:
            raise ValueError("GraphWaveNet model requires an adjacency matrix (supports).")
         
        # Update model_args with necessary parameters
        #how many nodes (i.e., sensors, locations, etc.) in your graph.
        self.model_args['num_nodes'] = supports[0].shape[0]
        # how many features each node has (e.g., temperature, humidity, etc.)
        self.model_args['in_dim'] = input_features
        # how many output features you want to predict (e.g., next temperature, humidity, etc.)
        self.model_args['out_dim'] = output_size
        # the adjacency matrices that define the graph structure
        self.model_args['supports'] = supports
        
        return GraphWaveNet(**self.model_args)

    def get_loss_function(self):
        """GraphWaveNet always uses MSE."""
        return nn.MSELoss()

    def adapt_output_for_loss(self, y_pred, y_batch):
        """Handles the specific output shape of GraphWaveNet."""
        #we only want the last time step's prediction, so we slice it
        #y_pred: (batch, nodes, features, only one time step from the prediction)
        y_pred = y_pred[..., -1:]
        if self.settings.PREDICTION_MODE == 'TREND':
            # Reshape from (batch, features, nodes, 1) -> (batch, nodes, features)
            #the features are the trend and magnitude in this case
            y_pred = y_pred.squeeze(-1).permute(0, 2, 1)
        return y_pred, y_batch