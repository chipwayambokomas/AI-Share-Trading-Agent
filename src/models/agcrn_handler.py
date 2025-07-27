# src/models/agcrn_handler.py

import torch.nn as nn
from .base_handler import BaseModelHandler
from .architectures.agcrn import AGCRN
import torch
import torch.nn.functional as F

class AGCRNHandler(BaseModelHandler):
    def name(self) -> str:
        """Returns the model's string identifier."""
        return "AGCRN"

    def is_graph_based(self) -> bool:
        """AGCRN is a graph-based model."""
        return True

    def get_loss_function(self):
        """AGCRN typically uses MAE for traffic forecasting, but MSE is also fine."""
        return nn.MSELoss()

    def build_model(self, **kwargs) -> AGCRN:
        """Implements the logic to build an instance of the AGCRN model."""
        # This model doesn't use a pre-computed adjacency matrix, so `supports` is ignored.
        # It learns the graph structure via node embeddings.
        
        # Get generic dimensions
        input_features, output_size = self.get_input_dims()
        
        # Add all necessary arguments from settings.py for the model's __init__ method
        # The `num_nodes` will be determined during data preprocessing.
        # We pass it in via kwargs from the trainer.
        self.model_args['num_nodes'] = kwargs['num_nodes']
        self.model_args['input_dim'] = input_features
        self.model_args['output_dim'] = output_size
        
        # AGCRN's output horizon is determined by our POINT_OUTPUT_WINDOW_SIZE
        self.model_args['horizon'] = self.settings.POINT_OUTPUT_WINDOW_SIZE
        
        print("Building AGCRN model with args:", self.model_args)
        return AGCRN(**self.model_args)
    
    def adapt_output_for_loss(self, y_pred, y_batch):
        """
        Robustly adapts model output and ground truth to a consistent shape for loss calculation,
        especially for single-step forecasting (horizon=1).

        - The goal is to make y_pred and y_batch have identical shapes before loss calculation.
        - For single-step forecasts, both should be flattened to a 3D tensor:
          (Batch, Num_Nodes, Features)
        """
        # Check if the model's prediction has a redundant horizon dimension of size 1.
        # This is typical for AGCRN's output.
        if y_pred.dim() == 4 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)

        # Adapt the ground truth tensor ONLY if it exists (i.e., during training).
        if y_batch is not None:
            if y_batch.dim() == 4 and y_batch.shape[1] == 1:
                y_batch = y_batch.squeeze(1)
            
        if print_once_loss:
            print(f"Shape of y_pred AFTER adaptation:  {y_pred.shape}")
            print(f"Shape of y_batch AFTER adaptation: {y_batch.shape}")
            setattr(self, 'print_once_loss', False)

        # After squeezing, both tensors should be 3D and have the same shape.
        # This now works for BOTH POINT and TREND modes.
        return y_pred, y_batch
    
    def extract_adjacency_matrix(self, model: AGCRN):
        """Extracts the learned adaptive adjacency matrix from a trained AGCRN model."""
        print("Attempting to extract learned adjacency matrix from AGCRN...")

        # AGCRN's graph is learned via its node_embeddings parameter.
        # Check if the model has this parameter.
        if not hasattr(model, 'node_embeddings'):
            print("Model does not have 'node_embeddings' attribute. Cannot extract matrix.")
            return None
            
        with torch.no_grad():
            # Get the learned node embeddings from the trained model
            node_embeddings = model.node_embeddings

            # Reconstruct the matrix using the same formula as in the AVWGCN forward pass
            raw_adj = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
            activated_adj = F.relu(raw_adj)
            final_adj_matrix = F.softmax(activated_adj, dim=1)

            print("Successfully extracted learned adjacency matrix.")
            return final_adj_matrix

    def adapt_input_for_model(self, X_batch):
        """
        AGCRN expects input of shape (Batch, Seq_Len, Num_Nodes, Features).
        Our dataloader provides this shape, so no change is needed.
        """
        return X_batch
