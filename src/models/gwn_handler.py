# src/models/graphwavenet_handler.py

import torch
import torch.nn as nn
from .base_handler import BaseModelHandler
from .architectures.gwn import GraphWaveNet

class GraphWaveNetHandler(BaseModelHandler):
    def name(self) -> str:
        return "GraphWaveNet"

    def is_graph_based(self) -> bool:
        return True
        
    def build_model(self, **kwargs) -> GraphWaveNet:
        """
        Builds the new GraphWaveNet model, configured for direct multi-step forecasting.
        """
        input_features, _ = self.get_input_dims()
        supports = kwargs.get('supports')
        num_nodes = kwargs.get('num_nodes')

        if supports is None or supports[0] is None:
            raise ValueError("GraphWaveNet requires a pre-computed adjacency matrix.")
        if num_nodes is None:
            raise ValueError("GraphWaveNet handler requires num_nodes to be passed.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Building GraphWaveNet model on device: {device}")
            
        self.model_args['node_cnt'] = num_nodes
        self.model_args['in_dim'] = input_features
        self.model_args['supports'] = supports
        self.model_args['device'] = device
        
        # --- KEY CHANGE FOR MULTI-STEP ---
        # The 'out_dim' of this new architecture corresponds to the forecast horizon.
        if self.settings.PREDICTION_MODE == "POINT":
            self.model_args['out_dim'] = 1
        else: # TREND mode
            # For trend, we predict 2 features (slope, duration) for a single step.
            # We must reshape this later to match the target.
            self.model_args['out_dim'] = 2
        # ---------------------------------
        
        return GraphWaveNet(**self.model_args)

    def get_loss_function(self):
        return nn.MSELoss()

    def adapt_output_for_loss(self, y_pred, y_batch):
        """
        Adapts the GraphWaveNet's output for loss calculation for both
        POINT (multi-step) and TREND (single-step) modes.
        """
        # Raw y_pred shape from GWN: (Batch, Output_Features, Nodes, Output_Seq_Len)
        
        if self.settings.PREDICTION_MODE == "POINT":
            # For POINT mode, we slice the horizon and create a 4D tensor.
            horizon = self.settings.POINT_OUTPUT_WINDOW_SIZE
            y_pred_sliced = y_pred[:, :, :, -horizon:]
            # Permute to target shape: (Batch, Horizon, Nodes, Features)
            y_pred_final = y_pred_sliced.permute(0, 3, 2, 1)

        else: # TREND Mode
            # For TREND mode, the target is 3D (Batch, Nodes, Features).
            # 1. Take the prediction from the very last time step of the model's output sequence.
            # Shape becomes: (Batch, Output_Features, Nodes)
            y_pred_sliced = y_pred[..., -1]
            
            # 2. Permute the dimensions to match the target.
            # (Batch, Output_Features, Nodes) -> (Batch, Nodes, Output_Features)
            y_pred_final = y_pred_sliced.permute(0, 2, 1)
            # --- END OF FIX ---

        return y_pred_final, y_batch

    def extract_adjacency_matrix(self, model: GraphWaveNet):
        # This method remains correct as it accesses the same parameter names.
        print("Attempting to extract learned adjacency matrix from GraphWaveNet...")
        if not self.model_args.get('addaptadj', False):
            print("Model not configured with 'addaptadj=True'. No learned matrix.")
            return None
            
        with torch.no_grad():
            nodevec1 = model.nodevec1
            nodevec2 = model.nodevec2
            raw_adj = torch.mm(nodevec1, nodevec2)
            final_adj_matrix = torch.nn.functional.softmax(torch.nn.functional.relu(raw_adj), dim=1)
            print("Successfully extracted learned adjacency matrix.")
            return final_adj_matrix
        
    def adapt_input_for_model(self, X_batch):
        """
        GraphWaveNet's architecture expects input in the shape (B, F, N, T).
        Our loader provides (B, T, N, F). This method performs the required permutation.
        """
        # (Batch, Time, Nodes, Features) -> (Batch, Features, Nodes, Time)
        return X_batch.permute(0, 3, 2, 1)