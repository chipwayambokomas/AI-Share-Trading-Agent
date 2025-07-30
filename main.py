import os
from src.config import settings
from src.utils import print_header, setup_logging
from src.pipeline import (
    data_loader,
    data_partitioner,
    model_trainer,
    model_evaluator,
)
from src.models.model_factory import ModelFactory
from src.data_processing.preprocessor_factory import PreprocessorFactory
from src.analysis.graph_visualizer import GraphVisualizer


def main():
    """Main execution pipeline."""

    # `setup_logging()`: Initializes the logger to write to a file and the console.
    # `print_header()`: A utility for clean visual separation in the console output.
    # The print statement shows which experiment is currently running, based on the config.
    setup_logging()
    print_header("Project Setup")
    print(f"MODEL: {settings.MODEL_TYPE}, MODE: {settings.PREDICTION_MODE}")

    # `ModelFactory.create`: We ask the factory for a "handler" for the model specified in `settings.py`. The factory returns an object (e.g., `GraphWaveNetHandler`) 
    model_handler = ModelFactory.create(settings.MODEL_TYPE, settings)
    
    # `PreprocessorFactory.create`: We ask this factory for the correct data processor.
    # The factory uses the `model_handler` to decide. If `model_handler.is_graph_based()`
    # is True, it returns a `GraphProcessor`; otherwise, it returns a `DNNProcessor`.
    # This decouples the main pipeline from the preprocessing logic.
    preprocessor = PreprocessorFactory.create(model_handler, settings)

    # `data_loader.run`: Calls the dedicated data loading module. This stage is simple
    # and just returns the initial combined DataFrame.
    combined_df = data_loader.run(settings)

    # `preprocessor.process`: We call the `.process()` method on the processor object we got from the factory. The object itself contains all the complex logic for
    # either graph-based or DNN-based preprocessing.
    X, y, stock_ids, scalers, adj_matrix = preprocessor.process(combined_df)
    
    # `data_partitioner.run`: This module now takes `model_handler.is_graph_based()`
    # as an argument. This tells the partitioner whether to do a chronological split
    # (for graphs) or a stratified split (for DNNs) without cluttering the partitioner
    train_loader, val_loader, X_test_t, y_test_t, test_stock_ids = data_partitioner.run(
        X, y, stock_ids, model_handler.is_graph_based(), settings
    )
    
    # `model_trainer.run`: The trainer module receives the `model_handler`. The trainer
    # will then use the handler to build the model (`handler.build_model()`), get the
    # correct loss function (`handler.get_loss_function()`), and adapt data shapes
    # for training (`handler.adapt_output_for_loss()`).
    model, training_time, final_adj_matrix = model_trainer.run(
        train_loader, val_loader, model_handler, settings, supports=[adj_matrix] if adj_matrix is not None else None
    )
    
    #edit this to spit back out the learned adjacency matrix instead of us using the static one we created initially

    # `model_evaluator.run`: Similar to the trainer, the evaluator receives the handler.
    # It uses the handler to know how to process the model's output for evaluation
    # and whether to calculate graph-specific metrics.
    model_evaluator.run(
        model, X_test_t, y_test_t, test_stock_ids, scalers, model_handler, settings, final_adj_matrix
    )
    
    # `visualizer.run_all`: This takes the path of the saved graph metrics of a model, the learned adjacency matrix, the directory of where the files are to be saved and finally the adjacency matrix threshold to be used in recreating the graph ->it creates informative visuals with these metrics that can be used for analysis    
    if model_handler.is_graph_based():
        visualizer = GraphVisualizer(
            metrics_csv_path = os.path.join(settings.RESULTS_DIR,settings.MODEL_TYPE, f"evaluation_GRAPH_{settings.PREDICTION_MODE}__{settings.MODEL_TYPE}.csv"),
            adj_matrix = final_adj_matrix,
            save_dir=os.path.join(settings.RESULTS_DIR, settings.MODEL_TYPE, f"{settings.PREDICTION_MODE}_visualizations"),
            percentile_threshold=settings.EVAL_THRESHOLD
        )
        visualizer.run_all()
        

    print_header("Pipeline Finished")

# Standard Python construct to ensure the `main` function is called only when
# the script is executed directly.
if __name__ == "__main__":
    # Creates necessary directories if they don't exist. Good practice for any project.
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    main()