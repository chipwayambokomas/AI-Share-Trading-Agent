from .base_handler import BaseModelHandler
from .architectures.gwn import GraphWaveNet
import torch.nn as nn
import torch
import torch.nn.functional as F

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
    
    def extract_adjacency_matrix(self, model: GraphWaveNet):
        print("Extracting adjacency matrix from GraphWaveNet...")
        
        # Check if the model was configured to learn an adaptive matrix
        if not model.addaptadj:
            print("Model was not trained with 'addaptadj=True'. No matrix to extract.")
            return None
            
        # Use torch.no_grad() to ensure no gradients are computed
        with torch.no_grad():
            # Get the learned node embeddings from the trained model
            nodevec1 = model.nodevec1
            nodevec2 = model.nodevec2

            # Reconstruct the matrix using the same formula as in the forward pass
            raw_adj = torch.mm(nodevec1, nodevec2)
            activated_adj = F.relu(raw_adj)
            final_adj_matrix = F.softmax(activated_adj, dim=1)
        
        print("Successfully extracted adaptive adjacency matrix.")
        return final_adj_matrix
    
    