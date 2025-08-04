import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from .base_handler import BaseModelHandler
from .architectures.dstagnn import DSTAGNN

class DSTAGNNHandler(BaseModelHandler):
    def name(self) -> str:
        return "DSTAGNN"

    def is_graph_based(self) -> bool:
        return True

    def get_loss_function(self):
        return nn.SmoothL1Loss()

    def build_model(self, **kwargs):
        """
        Builds the DSTAGNN model using parameters from settings.py.
        Returns the model directly without wrapper.
        """
        input_features, output_size = self.get_input_dims()
        supports = kwargs.get('supports')
        num_nodes = kwargs.get('num_nodes')
        device = kwargs.get('device', torch.device('cpu'))
        
        # Get configuration from settings.py (passed through the pipeline)
        settings = kwargs.get('settings') or getattr(self, 'settings', None)
        self.settings = settings  # Store settings for later use
        
        if settings:
            model_config = settings.MODEL_ARGS.get('DSTAGNN', {})
            # Extract data parameters from settings
            if settings.PREDICTION_MODE == "POINT":
                num_timesteps_in = settings.POINT_INPUT_WINDOW_SIZE
                num_timesteps_out = settings.POINT_OUTPUT_WINDOW_SIZE
            else:  # TREND
                num_timesteps_in = settings.TREND_INPUT_WINDOW_SIZE
                num_timesteps_out = 2  # TREND prediction outputs 2 values: slope and duration
        else:
            # Fallback defaults if no settings available
            model_config = {
                'nb_block': 4, 'K': 3, 'nb_chev_filter': 32, 'nb_time_filter': 32,
                'n_heads': 3, 'd_k': 32, 'd_model': 512
            }
            num_timesteps_in = 60
            num_timesteps_out = 1

        # DSTAGNN expects the actual number of features per timestep
        # POINT mode: ['open', 'high', 'low', 'close', 'vwap'] = 5 features
        # TREND mode: ['slope', 'duration'] = 2 features
        if settings and settings.PREDICTION_MODE == "TREND":
            actual_features = 2  # TREND mode uses slope and duration only
        else:
            actual_features = len(settings.FEATURE_COLUMNS) if settings else 5
        print(f"Using actual feature count: {actual_features}")
            
        # Use the full temporal window from settings - no adjustment needed
        len_input = num_timesteps_in
        print(f"Using full temporal window: {len_input} timesteps")

        # Get the base adjacency matrix (already computed by pipeline)
        base_adj = supports[0].cpu().numpy() if supports and len(supports) > 0 else np.eye(num_nodes)
        
        # Create DSTAGNN matrices efficiently
        adj_mx = base_adj
        # 1. Create STAD matrix from base adjacency (simplified approach)
        adj_TMD = self._create_stad_from_base(base_adj)
        # 2. Create sparse STRG matrix from STAD
        adj_pa = self._create_strg_from_stad(adj_TMD, num_nodes)
        
        # DEBUG: Print adjacency matrix information
        print(f"\n=== DSTAGNN Adjacency Matrix Information ===")
        print(f"Base adjacency (adj_mx) shape: {adj_mx.shape}")
        print(f"Base adjacency connections: {np.count_nonzero(adj_mx)}")
        print(f"STAD matrix (adj_TMD) shape: {adj_TMD.shape}")
        print(f"STAD matrix connections: {np.count_nonzero(adj_TMD)}")
        print(f"STRG matrix (adj_pa) shape: {adj_pa.shape}")
        print(f"STRG matrix connections: {np.count_nonzero(adj_pa)}")
        print(f"===============================================\n")
        
        # SAVE: Export adjacency matrices for inspection
        import os
        adj_dir = "adjacency_matrices"
        os.makedirs(adj_dir, exist_ok=True)
        
        np.save(f"{adj_dir}/dstagnn_base_adj.npy", adj_mx)
        np.save(f"{adj_dir}/dstagnn_stad_adj.npy", adj_TMD) 
        np.save(f"{adj_dir}/dstagnn_strg_adj.npy", adj_pa)
        print(f"Saved adjacency matrices to {adj_dir}/ directory")
        
        # Build model parameters from settings
        model_params = {
            # Architecture from settings.py
            **model_config,  # This spreads all DSTAGNN config from settings
            
            # Override/add required parameters
            'in_channels': actual_features,        # FIXED: should be 5 (actual feature count)
            'time_strides': 1,
            'num_for_predict': num_timesteps_out,
            'len_input': len_input,                # Use full temporal window (60)
            'num_of_vertices': num_nodes,
            
            # Adjacency matrices
            'adj_mx': adj_mx,
            'adj_pa': adj_pa,
            'adj_TMD': adj_TMD,
            'device': device
        }

        # Create the DSTAGNN model directly (no wrapper)
        model = DSTAGNN(**model_params)
        
        # Store the target timesteps for reference
        self.target_timesteps = len_input
        
        return model

    def _create_stad_from_base(self, base_adj):
        """
        Create STAD matrix from base adjacency matrix using correlation patterns.
        Since data loading is handled by pipeline, use the base adjacency as foundation.
        """
        try:
            # Use base adjacency as starting point for temporal similarities
            # Apply some transformation to make it suitable for STAD
            stad_matrix = base_adj.copy()
            
            # Add some noise/variation to make it more representative of temporal patterns
            if np.sum(stad_matrix) > 0:
                # Normalize to [0,1] and apply sigmoid-like transformation
                stad_matrix = stad_matrix / (np.max(stad_matrix) + 1e-8)
                stad_matrix = 1 / (1 + np.exp(-5 * (stad_matrix - 0.5)))
                
            # Ensure no self-loops
            np.fill_diagonal(stad_matrix, 0)
            
            print(f"Created STAD matrix with {np.count_nonzero(stad_matrix)} temporal similarities")
            return stad_matrix
            
        except Exception as e:
            print(f"Warning: Error creating STAD matrix: {e}")
            return base_adj.copy()

    def _create_strg_from_stad(self, stad_matrix, num_nodes):
        """
        Create STRG matrix by sparsifying the STAD matrix.
        """
        try:
            if stad_matrix is None:
                return np.eye(num_nodes)
            
            # Sparsify: keep top k connections per node
            k = max(2, min(6, int(np.sqrt(num_nodes))))
            strg_matrix = np.zeros_like(stad_matrix)
            
            for i in range(num_nodes):
                # Get strongest connections for node i
                similarities = stad_matrix[i].copy()
                similarities[i] = -1  # Exclude self
                
                # Get top k connections
                if np.max(similarities) > 0:
                    top_k_idx = np.argsort(similarities)[-k:]
                    valid_idx = top_k_idx[similarities[top_k_idx] > 0]
                    strg_matrix[i, valid_idx] = stad_matrix[i, valid_idx]
            
            # Make symmetric
            strg_matrix = np.maximum(strg_matrix, strg_matrix.T)
            np.fill_diagonal(strg_matrix, 0)
            
            connections = np.count_nonzero(strg_matrix)
            print(f"Created STRG matrix with {connections} sparse connections")
            
            # Ensure minimum connectivity
            if connections == 0:
                print("Creating minimal ring connectivity")
                for i in range(num_nodes):
                    next_node = (i + 1) % num_nodes
                    strg_matrix[i, next_node] = 1
                    strg_matrix[next_node, i] = 1
                connections = np.count_nonzero(strg_matrix)
                print(f"Minimal graph created with {connections} connections")
            
            return strg_matrix
            
        except Exception as e:
            print(f"Warning: Error creating STRG matrix: {e}")
            # Return minimal connected graph
            strg_matrix = np.eye(num_nodes)
            for i in range(num_nodes - 1):
                strg_matrix[i, i+1] = 1
                strg_matrix[i+1, i] = 1
            return strg_matrix

    def adapt_output_for_loss(self, y_pred, y_batch):
        """
        DSTAGNN output adaptation for loss calculation.
        Ensures tensor shapes match between prediction and target.
        
        DSTAGNN outputs: (batch, nodes, output_features) 
        Target format varies by mode:
        - POINT: (batch, horizon, nodes, features) -> squeeze to (batch, nodes)
        - TREND: (batch, nodes, features) -> keep as (batch, nodes, features)
        """
        if y_batch is not None:
            # Handle different modes
            if hasattr(self, 'settings') and self.settings and self.settings.PREDICTION_MODE == "TREND":
                # TREND mode: target is (batch, nodes, 2_features), prediction should match
                # No squeezing needed for TREND mode
                pass
            else:
                # POINT mode: Handle target tensor shape: (batch, horizon=1, nodes, features=1) -> (batch, nodes)
                if y_batch.dim() == 4:
                    if y_batch.shape[1] == 1:  # Remove horizon dimension
                        y_batch = y_batch.squeeze(1)  # (B, 1, N, F) -> (B, N, F)
                    if y_batch.shape[-1] == 1:  # Remove feature dimension if needed
                        y_batch = y_batch.squeeze(-1)  # (B, N, 1) -> (B, N)
                
                # For POINT mode, also squeeze y_pred if it has singleton feature dimension
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)  # (B, N, 1) -> (B, N)
        
        return y_pred, y_batch
    
    def extract_adjacency_matrix(self, model):
        """
        Extract learned spatial attention masks from DSTAGNN.
        """
        try:
            # Access the submodule's BlockList
            if hasattr(model, 'model') and hasattr(model.model, 'BlockList'):
                blocks = model.model.BlockList
            elif hasattr(model, 'BlockList'):
                blocks = model.BlockList
            else:
                print("Warning: Could not find BlockList in DSTAGNN model.")
                return None
                
            if len(blocks) > 0:
                first_block = blocks[0]
                if hasattr(first_block, 'cheb_conv_SAt'):
                    conv_layer = first_block.cheb_conv_SAt
                    if hasattr(conv_layer, 'mask') and len(conv_layer.mask) > 0:
                        all_masks = [m.detach().cpu().numpy() for m in conv_layer.mask]
                        learned_adj = np.mean(all_masks, axis=0)
                        print("Successfully extracted learned spatial masks.")
                        return learned_adj
            
            print("Warning: Could not extract learned adjacency mask from DSTAGNN model.")
            return None
            
        except Exception as e:
            print(f"Error extracting adjacency matrix from DSTAGNN: {e}")
            return None
    
    def adapt_y_for_loss(self, y_pred, y_batch):
        """
        Additional output adaptation if needed.
        """
        return y_pred, y_batch