import torch.nn as nn

class MLP_Forecaster(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.2, **kwargs):
        super(MLP_Forecaster, self).__init__()
        layers = []
        last_size = input_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(last_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_size = layer_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x_flat = x.view(x.size(0), -1) # Flatten the sequence and features
        return self.network(x_flat)