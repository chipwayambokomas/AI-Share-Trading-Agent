from .base_handler import BaseModelHandler
from .dnn_handler import TCNHandler, MLPHandler
from .gwn_handler import GraphWaveNetHandler
from .agcrn_handler import AGCRNHandler
from .dstagnn_handler import DSTAGNNHandler
# Import new handlers here when you add them.
# from .new_stgnn_handler import NewSTGNNHandler

class ModelFactory:
    _handlers = {
        "TCN": TCNHandler,
        "MLP": MLPHandler,
        "GraphWaveNet": GraphWaveNetHandler,
        "AGCRN": AGCRNHandler,
        "DSTAGNN": DSTAGNNHandler,
        # IMPORTANT: Register new handlers here.
        # "NewSTGNN": NewSTGNNHandler,
    }

    @staticmethod
    def create(model_type: str, settings) -> BaseModelHandler:
        # Retrieve the handler class from the registered handlers dictionary
        handler_class = ModelFactory._handlers.get(model_type)
        # If the model type is not found, raise an error
        if not handler_class:
            raise ValueError(f"Model type '{model_type}' not recognized in ModelFactory.")
        # Instantiate and return the handler with the provided settings
        return handler_class(settings)