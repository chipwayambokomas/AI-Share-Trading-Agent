# models/MLPModel.py

import torch.nn as nn

class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron model for time series prediction.
    
    Args:
        input_size (int): Number of input features (window size)
        hidden_size (int): Number of neurons in the hidden layer
        output_size (int): Number of outputs (typically 1 for point prediction)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.network(x)
