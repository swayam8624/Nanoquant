import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model: nn.Module,
                train_loader,
                valid_loader=None,
                device: str = "cpu",
                epochs: int = 1,
                lr: float = 1e-5,
                scheduler=None):
    """
    Trains the given model using a generic training loop.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        valid_loader (DataLoader, optional): DataLoader for the validation data.
        device (str): Device on which to train ('cpu', 'cuda', 'mps', etc.).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        scheduler (optional): A learning rate scheduler instance.

    Returns:
        tuple: A tuple containing:
            - train_loss_history (list): List of average training losses per epoch.
            - valid_metrics_history (list): List of tuples (avg_loss, accuracy) per epoch (if valid_loader is provided).
    """
    # Move the model to the target device.
    model.to(device)
    model.train()

    # Set up the optimizer (Adam) and loss function (CrossEntropyLoss).
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    valid_metrics_history = []  # Each element: (validation_loss, validation_accuracy)

    # Training loop over epochs.
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_batches = 0

        # Iterate over training batches.
        for batch in train_loader:
            # Assume each batch is a dictionary with keys like 'input_ids' and 'label'.
            # Move inputs and labels to the device.
            inputs = {k: batch[k].to(device) for k in batch if k != "label"}
            targets = batch["label"].to(device)

            optimizer.zero_grad()  # Reset gradients.
            output = model(**inputs)  # Forward pass.
            # Handle outputs: if it's a dict (as in Hugging Face models), extract "logits".
            logits = output["logits"] if isinstance(output, dict) else output

            # Reshape outputs and targets if needed for loss computation.
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update parameters.

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

        # Evaluate on validation data, if provided.
        if valid_loader is not None:
            valid_loss, valid_accuracy = evaluate_model(model, valid_loader, device, loss_fn)
            valid_metrics_history.append((valid_loss, valid_accuracy))
            print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}")

        # Step the scheduler, if provided.
        if scheduler is not None:
            scheduler.step()

    # Plot training loss curve.
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), train_loss_history, marker="o", linestyle="--", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()

    return train_loss_history, valid_metrics_history

def evaluate_model(model: nn.Module, data_loader, device: str = "cpu", loss_fn: nn.Module = None):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device on which to evaluate.
        loss_fn (nn.Module, optional): Loss function for computing evaluation loss.

    Returns:
        tuple: (avg_loss, accuracy)
            - avg_loss: Average loss over all batches (if loss_fn is provided; otherwise, None).
            - accuracy: Accuracy of the model on the evaluation data.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: batch[k].to(device) for k in batch if k != "label"}
            targets = batch["label"].to(device)
            output = model(**inputs)
            # Extract logits if output is a dictionary.
            logits = output["logits"] if isinstance(output, dict) else output

            if loss_fn is not None:
                loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
                total_loss += loss.item()

            # Compute predictions and accuracy.
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            num_batches += 1

    avg_loss = total_loss / num_batches if loss_fn is not None else None
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy

# --- Testing the Training Module with a Dummy Model ---
if __name__ == "__main__":
    # Define a simple dummy model that mimics Hugging Face model output.
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, **kwargs):
            # Extract "input_ids" from kwargs.
            x = kwargs.get("input_ids", None)
            if x is None:
                raise ValueError("Expected keyword argument 'input_ids'")
            logits = self.fc(x)
            # Mimic a Hugging Face output object by returning a dict.
            return {"logits": logits}

    # Create dummy data: a list of dictionaries with "input_ids" and "label".
    dummy_data = [{"input_ids": torch.randn(1, 10), "label": torch.tensor([1])} for _ in range(100)]
    dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=8, shuffle=True)

    # Use the dummy model.
    dummy_model = DummyModel()
    device = "cpu"
    
    # Train the model with the dummy data.
    train_loss, valid_metrics = train_model(dummy_model, dummy_loader, valid_loader=dummy_loader, device=device, epochs=5, lr=1e-3)
    
    # Evaluate the model explicitly.
    avg_loss, accuracy = evaluate_model(dummy_model, dummy_loader, device, nn.CrossEntropyLoss())
    print(f"Final Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
