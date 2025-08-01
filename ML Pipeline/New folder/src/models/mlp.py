# src/models/mlp.py
#paper based configuration 
import torch.nn as nn
import torch

class MLP_Forecaster(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.0, 
                 activation="tanh", output_activation="linear", **kwargs):
        """
        MLP Forecaster matching research paper specifications:
        - 2 hidden layers with 16 and 10 neurons respectively
        - Hyperbolic tangent (tanh) activation for hidden layers
        - Linear activation for output layer
        - No dropout (as per paper)
        
        Args:
            input_size: Size of input features
            hidden_layers: List of hidden layer sizes [16, 10]
            output_size: Size of output
            dropout: Dropout rate (set to 0.0 for paper compliance)
            activation: Activation function for hidden layers ("tanh")
            output_activation: Activation function for output layer ("linear")
        """
        super(MLP_Forecaster, self).__init__()
        
        # Activation function mapping
        activation_funcs = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "linear": nn.Identity()
        }
        
        # Build the network layers
        layers = []
        last_size = input_size
        
        # Add hidden layers with specified activation and dropout
        for i, layer_size in enumerate(hidden_layers):
            layers.append(nn.Linear(last_size, layer_size))
            layers.append(activation_funcs[activation])
            
            # Only add dropout if specified and > 0
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            last_size = layer_size
        
        # Add output layer
        layers.append(nn.Linear(last_size, output_size))
        
        # Add output activation if not linear
        if output_activation != "linear":
            layers.append(activation_funcs[output_activation])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization for tanh
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights appropriately for tanh activation"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization works well with tanh
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the MLP
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Output tensor from the MLP
        """
        # Flatten input: (batch_size, sequence_length * features)
        x_flat = x.view(x.size(0), -1)
        return self.network(x_flat)
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "MLP (Multi-Layer Perceptron)",
            "hidden_layers": "2 layers [16, 10 neurons]",
            "activation": "Hyperbolic Tangent (tanh)",
            "output_activation": "Linear",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "dropout": "None (as per research paper)",
            "paper_reference": "Scaled Conjugate Gradient Back-propagation"
        }