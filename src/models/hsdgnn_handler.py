# src/models/hsdgnn_handler.py

import torch
import torch.nn as nn
from argparse import Namespace
from torch.utils.data import TensorDataset, DataLoader
from .base_handler import BaseModelHandler
from .architectures.hsdgnn import HSDGNN

class HSDGNNHandler(BaseModelHandler):
    def name(self) -> str:
        """Returns the model's string identifier."""
        return "HSDGNN"

    def is_graph_based(self) -> bool:
        """HSDGNN is a graph-based model."""
        return True

    def get_loss_function(self):
        """HSDGNN uses standard MSE."""
        # For TREND mode, a combined loss on two features is needed.
        if self.settings.PREDICTION_MODE == "TREND":
            return lambda pred, act: nn.MSELoss()(pred[..., 0], act[..., 0]) + nn.MSELoss()(pred[..., 1], act[..., 1])
        return nn.MSELoss()

    def build_model(self, **kwargs) -> HSDGNN:
        """
        Implements the unique logic to build an HSDGNN model instance,
        now correctly handling both POINT and TREND modes.
        """
        # --- START OF CHANGE ---
        # Determine input and output dimensions based on the prediction mode.
        if self.settings.PREDICTION_MODE == "TREND":
            # In TREND mode, the input features are just [slope, duration].
            # The GraphProcessor for TREND does not add time features.
            hsdgnn_input_dim = 2
            # The output is also the next [slope, duration].
            output_size = 2
            horizon = 1 # Trend predicts a single next step
        else: # POINT mode
            # In POINT mode, we use the original features + 2 time-based features.
            hsdgnn_input_dim = len(self.settings.FEATURE_COLUMNS) + 2
            # The output size is the prediction horizon.
            output_size = 1 #self.settings.POINT_OUTPUT_WINDOW_SIZE
            horizon = self.settings.POINT_OUTPUT_WINDOW_SIZE
        # --- END OF CHANGE ---

        # Construct the special arguments object HSDGNN expects.
        hsdgnn_constructor_args = Namespace(
            num_nodes=kwargs['num_nodes'],
            input_dim=hsdgnn_input_dim,
            output_dim=output_size,
            rnn_units=self.model_args['rnn_units'],
            embed_dim=self.model_args['embed_dim'],
            steps_per_day=self.model_args['steps_per_day'],
            lag=self.get_window_size(),
            horizon=horizon,
            batch_size=self.settings.BATCH_SIZE
        )
        
        print("Building HSDGNN model with args:", hsdgnn_constructor_args)
        return HSDGNN(hsdgnn_constructor_args)

    def adapt_output_for_loss(self, y_pred, y_batch):
        """
        Reshapes the complex output of HSDGNN to match the ground truth tensor,
        now handling both POINT and TREND modes.
        """
        # --- START OF CHANGE ---
        if self.settings.PREDICTION_MODE == "TREND":
            # For TREND, model output is (B, H*D, N, 1) where H=1, D=2.
            # Shape is (B, 2, N, 1). We need (B, N, 2) to match y_batch.
            if y_pred.dim() == 4:
                y_pred = y_pred.squeeze(-1).permute(0, 2, 1)
            return y_pred, y_batch
        else: # POINT mode
            """
            # Original logic for POINT mode
            B, _, N, _ = y_pred.shape
            H = self.settings.POINT_OUTPUT_WINDOW_SIZE
            D_out = y_batch.shape[-1] # Output features (usually 1 for target column)
            y_pred = y_pred.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
            return y_pred, y_batch
        # --- END OF CHANGE ---
            """

            # =============================================================================
            # --- START OF CHANGE ---
            # =============================================================================
            B, _, N, _ = y_pred.shape
            H = self.settings.POINT_OUTPUT_WINDOW_SIZE
            
            # If y_batch is available (training), get D_out from it.
            # If y_batch is None (evaluation), we know we are predicting 1 target feature.
            if y_batch is not None:
                D_out = y_batch.shape[-1]
            else:
                D_out = 1
            
            y_pred = y_pred.squeeze(-1).view(B, H, D_out, N).permute(0, 1, 3, 2)
            return y_pred, y_batch
            # ===========================================================================
            # --- END OF CHANGE ---
            # ===========================================================================

    def extract_adjacency_matrix(self, model: HSDGNN, X_test_t: torch.Tensor):
        """
        Extracts a representative dynamic adjacency matrix from a trained HSDGNN model.
        This requires passing a batch of test data through the model.
        """
        print("Extracting dynamic adjacency matrix from HSDGNN on a test batch...")
        
        model.eval()
        with torch.no_grad():
            test_dataset = TensorDataset(X_test_t)
            temp_loader = DataLoader(test_dataset, batch_size=self.settings.BATCH_SIZE, shuffle=False)
            X_batch_sample = next(iter(temp_loader))[0]
            
            _, dynamic_adj_batch = model(X_batch_sample, return_adjs=True)
            
            representative_adj = dynamic_adj_batch[0, -1, :, :].cpu()
            
            print("Successfully extracted a representative adjacency matrix from the model.")
            return representative_adj