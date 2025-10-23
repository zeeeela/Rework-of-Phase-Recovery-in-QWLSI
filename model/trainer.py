
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float("inf")  # Initialize the best validation loss
    best_model_weights = None  # To store the best model state

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)
        
        for inputs, targets in train_progress:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)[0]  # Main output (d0)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_progress.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", leave=True)
        
        with torch.no_grad():
            for val_inputs, val_targets in val_progress:
                val_outputs = model(val_inputs)[0]  # Main output (d0)
                loss = criterion(val_outputs, val_targets)
                val_loss += loss.item()
                val_progress.set_postfix({"Batch Loss": loss.item()})

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model if the validation loss improves
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            best_model_weights = model.state_dict()  # Save the model's weights

    # Save the best model weights to the specified path
    if best_model_weights:
        torch.save(best_model_weights, save_path)
        print(f"Best model saved to {save_path}")

    print("Training complete.")