from .dnn_processor import DNNProcessor
from .graph_processor import GraphProcessor
from ..models.base_handler import BaseModelHandler

class PreprocessorFactory:
    """
    Creates the correct data processor based on whether the model is graph-based.
    """
    @staticmethod
    def create(model_handler: BaseModelHandler, settings):
        if model_handler.is_graph_based():
            return GraphProcessor(settings)
        else:
            return DNNProcessor(settings)