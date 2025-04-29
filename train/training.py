import torch
from .generation import generate_text_simple
from .loss import calc_loss_batch, calc_loss_loader

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def train_model_with_regularization(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, 
                       patience=5, weight_decay=0.01, dropout_rate=0.1):
    """
    Train a model with early stopping and regularization
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: The optimizer to use
        device: Device to train on
        num_epochs: Maximum number of epochs to train for
        eval_freq: How often to evaluate the model (steps)
        eval_iter: Number of batches to use for evaluation
        start_context: Starting text for generation examples
        tokenizer: Tokenizer for text generation
        patience: Number of evaluations to wait for improvement before early stopping
        weight_decay: L2 regularization strength (applied via optimizer)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        train_losses, val_losses, track_tokens_seen
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # Set up dropout for regularization
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'dropout') and module.dropout is None:
                module.dropout = torch.nn.Dropout(dropout_rate)
    
    # Early stopping variables
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            
            # Forward pass
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # L2 regularization is handled by the optimizer with weight_decay
            
            loss.backward() # Calculate loss gradients
            
            # Gradient clipping for stability (optional regularization technique)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    # Save the best model state
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping triggered after {global_step} steps")
                        # Restore best model
                        model.load_state_dict(best_model_state)
                        return train_losses, val_losses, track_tokens_seen
        
        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    
    # If we completed all epochs without early stopping, ensure we use the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()