import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Temporal Block for TCN
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels  
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout rate
        """
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNModel(nn.Module):
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 num_channels: List[int],
                 kernel_size: int = 2,
                 dropout: float = 0.2):
        """
        Temporal Convolutional Network for time series prediction
        
        Args:
            input_size: Number of input features
            output_size: Number of output predictions
            num_channels: List of hidden channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TCN expects input of shape (batch_size, input_size, sequence_length)
        y1 = self.network(x)
        
        # Take the last time step output
        # Shape: (batch_size, num_channels[-1])
        o = y1[:, :, -1]
        
        # Apply final linear layer
        return self.linear(o)


class TCNStockPredictor(nn.Module):
    def __init__(self, config):
        """
        TCN-based stock price predictor
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(TCNStockPredictor, self).__init__()
        
        # Extract config parameters
        self.input_size = config.get('input_size', 5)  # Number of features (OHLCV)
        self.sequence_length = config.get('sequence_length', 60)
        self.output_size = config.get('output_size', 1)  # Predict next closing price
        
        # TCN specific parameters
        self.num_channels = config.get('num_channels', [64, 64, 64])
        self.kernel_size = config.get('kernel_size', 2)
        self.dropout = config.get('dropout', 0.2)
        
        # Initialize TCN
        self.tcn = TCNModel(
            input_size=self.input_size,
            output_size=self.output_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        
    def forward(self, x):
        """
        Forward pass for stock prediction
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predicted stock prices of shape (batch_size, output_size)
        """
        # Transpose to match TCN expected input format
        # From (batch_size, sequence_length, input_size) to (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        return self.tcn(x)

    def get_model_info(self):
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'TCN',
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'output_size': self.output_size,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }