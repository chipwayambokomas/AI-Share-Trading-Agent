# dataset.py
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_size):
        self.series = series
        self.input_size = input_size
        self.X, self.y = self.create_sequences()
        

    def create_sequences(self):
        X, y = [], []
        # Ensure there's enough data to form at least one sequence
        if len(self.series) <= self.input_size:
            return np.array(X), np.array(y) # Return empty arrays

        for i in range(len(self.series) - self.input_size):
            X.append(self.series[i:i+self.input_size])
            y.append(self.series[i+self.input_size])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]