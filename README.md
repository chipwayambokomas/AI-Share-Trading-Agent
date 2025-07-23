# AI Stock Trading Agent: A Modular Forecasting Framework

This project provides a modular and extensible framework for developing, training, and evaluating various time-series forecasting models for stock market prediction. The architecture is built on object-oriented principles, allowing for the easy integration of new models with minimal changes to the core pipeline.

The system supports two primary prediction modes:

- **Point Prediction**: Forecasting the exact price of a stock for a future time step.
- **Trend Prediction**: Forecasting the characteristics (slope and duration) of the next price trend, identified via piecewise linear segmentation.

It supports both standard deep learning models (e.g., TCN, MLP) and Spatio-Temporal Graph Neural Networks (e.g., GraphWaveNet), which can model inter-stock relationships.

## Table of Contents

- [Project Architecture](#project-architecture)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [How to Add a New Model](#how-to-add-a-new-model)

## Project Architecture

The framework is designed around a decoupled, object-oriented pipeline orchestrated by the `main.py` script. Key design patterns used include:

- **Factory Pattern**: `ModelFactory` and `PreprocessorFactory` dynamically select and instantiate components based on the configuration file.
- **Strategy Pattern (via Handlers)**: Each model has a dedicated handler class (e.g., `GraphWaveNetHandler`) encapsulating model-specific logic. All handlers implement a common abstract interface (`BaseModelHandler`).
- **Single Responsibility Principle**: Each module has a clearly defined responsibility (e.g., `data_loader.py` only loads data, `model_trainer.py` only handles training).

This structure makes the framework easy to maintain, extend, and debug.

## File Structure

```
/
├── data/                     # Raw data files
├── logs/                     # Log files
├── results/                  # Evaluation outputs
├── src/
|   ├── analysis/             # graph metrics visualisation logic
│   ├── config/               # Configuration settings
│   ├── data_processing/      # Data processors and factory
│   ├── models/               # Handlers, architectures, factory
│   ├── pipeline/             # Core pipeline stages
│   └── utils.py              # Utility functions
├── main.py                   # Entry point
└── requirements.txt          # Python dependencies
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Data

Place your stock data Excel file (e.g., `JSE_Top40_OHLCV_2014_2024.xlsx`) inside the `/data` directory.  
Make sure the filename in `src/config/settings.py` matches your file.

## Running the Pipeline

### 1. Configure the Experiment

Edit `src/config/settings.py`:

- Set `PREDICTION_MODE` to `"POINT"` or `"TREND"`.
- Set `MODEL_TYPE` to the model you want to run (e.g., `"GraphWaveNet"`, `"TCN"`).
- Adjust other parameters like `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, and model-specific arguments in `MODEL_ARGS`.

### 2. Run the Pipeline

From the root directory:

```bash
python main.py
```

### 3. View the Results

- Console output shows progress.
- Logs are saved to `/logs/preprocessing.log`.
- Evaluation metrics and predictions are saved in the `/results/` directory.

## How to Add a New Model

To add a new model (e.g., `NewSTGNN`), follow these steps:

### Step 1: Add the Model Architecture

Create a new file in `src/models/architectures/new_stgnn.py`:

```python
import torch
import torch.nn as nn

class NewSTGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, some_param, **kwargs):
        super().__init__()
        # Define model layers here

    def forward(self, x):
        # Define forward pass
        return x
```

### Step 2: Create the Model Handler

Create `src/models/new_stgnn_handler.py`:

```python
import torch.nn as nn
from .base_handler import BaseModelHandler
from .architectures.new_stgnn import NewSTGNN

class NewSTGNNHandler(BaseModelHandler):
    def name(self) -> str:
        return "NewSTGNN"

    def is_graph_based(self) -> bool:
        return True

    def get_loss_function(self):
        return nn.MSELoss()

    def build_model(self, **kwargs) -> NewSTGNN:
        input_features, output_size = self.get_input_dims()
        supports = kwargs.get('supports')

        self.model_args['num_nodes'] = supports[0].shape[0]
        self.model_args['input_dim'] = input_features

        return NewSTGNN(**self.model_args)

    def adapt_output_for_loss(self, y_pred, y_batch):
        return y_pred, y_batch
    
    def extract_adjacency_matrix(self, model: GraphWaveNet):
        """
        Model specific implementation of learned adj_matrix
        """
        return final_adj_matrix
```

### Step 3: Add to Configuration

In `src/config/settings.py`:

```python
MODEL_ARGS = {
    "TCN": {...},
    "GraphWaveNet": {...},
    "NewSTGNN": {
        "some_param": 128,
        "another_param": 0.5
    }
}
```

Optionally set:

```python
MODEL_TYPE = "NewSTGNN"
```

### Step 4: Register the Handler in the Factory

Edit `src/models/model_factory.py`:

```python
from .new_stgnn_handler import NewSTGNNHandler

class ModelFactory:
    _handlers = {
        "TCN": TCNHandler,
        "MLP": MLPHandler,
        "GraphWaveNet": GraphWaveNetHandler,
        "DSTAGNN": DSTAGNNHandler,
        "NewSTGNN": NewSTGNNHandler,
    }
```

### That's it!

To run your model, set `MODEL_TYPE = "NewSTGNN"` in the configuration and run:

```bash
python main.py
```

The pipeline will automatically detect the handler and use the correct data processing and training logic.

