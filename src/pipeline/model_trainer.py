# src/pipeline/model_trainer.py

import torch
import time
from tqdm import tqdm
from ..utils import print_header
from ..models.base_handler import BaseModelHandler
import copy
import optuna

def run(train_loader, val_loader, model_handler: BaseModelHandler, settings, supports=None, verbose=True, trial = None):
    """
    Stage 4: Initializes, trains, and validates the specified model.
    """
    if verbose:
        print_header("Stage 4: Model Development")
        print(f"Training {model_handler.name()} for {settings.PREDICTION_MODE} prediction...")
    
    sample_x, _ = next(iter(train_loader))
    num_nodes = sample_x.shape[2] if model_handler.is_graph_based() else None

    model = model_handler.build_model(supports=supports, num_nodes=num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    loss_fn = model_handler.get_loss_function()
    
    best_val_loss = float('inf')
    best_model_state = None

    if verbose:
        print("\nStarting model training...")
    start_time = time.time()
    print_once = True
    for epoch in range(settings.EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Use the verbose flag to disable tqdm progress bar during HPO
        train_iterator = tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1:2d}/{settings.EPOCHS}", disable=not verbose)
        
        for X_batch, y_batch in train_iterator:
            if print_once:
                print(f"\n--- Data Shape Verification ---")
                print(f"X_batch shape entering model: {X_batch.shape}")
                print(f"y_batch shape from loader:  {y_batch.shape}")
                print_once = False
            optimizer.zero_grad()
            X_batch_adapted = model_handler.adapt_input_for_model(X_batch)
            y_pred_raw = model(X_batch_adapted)
            y_pred, y_batch_adapted = model_handler.adapt_output_for_loss(y_pred_raw, y_batch)
            y_pred, y_batch_adapted = model_handler.adapt_y_for_loss(y_pred, y_batch_adapted)
            loss = loss_fn(y_pred, y_batch_adapted)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_loss += loss.item()
            if verbose:
                train_iterator.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        avg_val_loss = float('inf')
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch_adapted = model_handler.adapt_input_for_model(X_batch)
                    y_pred_raw = model(X_batch_adapted)
                    y_pred, y_batch_adapted = model_handler.adapt_output_for_loss(y_pred_raw, y_batch)
                    val_loss += loss_fn(y_pred, y_batch_adapted).item()
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
        
        if verbose:
            print(f"Epoch {epoch+1:2d}/{settings.EPOCHS} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
            
            if trial:
                # Report the validation loss of the current epoch to Optuna
                trial.report(avg_val_loss, epoch)

                # Check if the trial should be pruned
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch+1} due to poor performance.")
                    # Raise a special exception that Optuna catches to stop the trial
                    raise optuna.exceptions.TrialPruned()
    
    training_time = time.time() - start_time
    if verbose:
        print(f"\nModel training complete in {training_time:.2f} seconds.")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    final_adj_matrix = None
    if model_handler.is_graph_based():
        final_adj_matrix = model_handler.extract_adjacency_matrix(model)
        
    # --- MODIFICATION: Return the best_val_loss ---
    return model, training_time, final_adj_matrix, best_val_loss