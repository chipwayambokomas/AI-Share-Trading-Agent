import torch
import time
from tqdm import tqdm
from ..utils import print_header
from ..models.base_handler import BaseModelHandler

def run(train_loader, val_loader, model_handler: BaseModelHandler, settings, supports=None):
    """
    Stage 4: Initializes, trains, and validates the specified model.
    """
    print_header("Stage 4: Model Development")
    print(f"Training {model_handler.name()} for {settings.PREDICTION_MODE} prediction...")

    # Build the model using the handler
    model = model_handler.build_model(supports=supports)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    
    # Get the loss function from the handler
    loss_fn = model_handler.get_loss_function()
    
    best_val_loss = float('inf')
    best_model_state = None

    print("\nStarting model training...")
    start_time = time.time()
    for epoch in range(settings.EPOCHS):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1:2d}/{settings.EPOCHS}") as tepoch:
            #we will iterate through each batch in the train_loader
            for X_batch, y_batch in tepoch:
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                y_pred_raw = model(X_batch)
                
                # Let handler adapt output and/or ground truth for loss calculation -> remeber in the model handler for example for graph wavenet trend, the output is (batch, nodes, features, time_steps) and we need to adapt it to (batch, nodes, features) -> GWN the only one that uses this right now, rest of the models use the same output shape as the ground truth and use the abstract class method and pass through
                y_pred, y_batch_adapted = model_handler.adapt_output_for_loss(y_pred_raw, y_batch)
                y_pred, y_batch_adapted = model_handler.adapt_y_for_loss(y_pred, y_batch_adapted)

                # Calculate loss and backpropagate -> adjust the model weights
                loss = loss_fn(y_pred, y_batch_adapted)
                loss.backward()
                
                #This clips the gradients so that their norm (magnitude) does not exceed 5. Sometimes gradients can explode and become too large, causing training to become unstable. Clipping them helps keep training stable and prevents “exploding gradients.”

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient clipping
                
                #apply updates to the model parameters
                optimizer.step()
                # Accumulate training loss
                train_loss += loss.item()
                
                tepoch.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        # this tells the model to not calculate gradients, which saves memory and speeds up validation
        with torch.no_grad():
            # Iterate through validation data and do the same as above
            for X_batch, y_batch in val_loader:
                y_pred_raw = model(X_batch)
                
                y_pred, y_batch_adapted = model_handler.adapt_output_for_loss(y_pred_raw, y_batch)
                y_pred, y_batch_adapted = model_handler.adapt_y_for_loss(y_pred, y_batch_adapted)
                
                val_loss += loss_fn(y_pred, y_batch_adapted).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        print(f"Epoch {epoch+1:2d}/{settings.EPOCHS} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nModel training complete in {training_time:.2f} seconds.")
    model.load_state_dict(best_model_state)
    return model, training_time