import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
import matplotlib.pyplot as plt

def prepare_qat_model(model, backend: str = "fbgemm"):
    """
    Prepares a model for Quantization-Aware Training (QAT) in FP32.

    Converts the model to float32, assigns the default QAT configuration for the specified backend,
    and inserts fake quantization modules and observers.

    Args:
        model (nn.Module): The pre-trained model to prepare for QAT.
        backend (str): The quantization backend to use (default: "fbgemm").

    Returns:
        nn.Module: The QAT-prepared model.
    """
    # Convert model to float32 to avoid half-precision issues
    model = model.float()
    
    # Use the default QAT configuration
    qconfig = quant.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    # Ensure model is in training mode
    model.train()
    # Prepare the model for QAT (inserts fake quantization modules and observers)
    model_prepared = quant.prepare_qat(model, inplace=False)
    return model_prepared

def train_qat_model(model, train_loader, device: str, epochs: int = 1, lr: float = 1e-5,
                    scheduler_step_size: int = None, scheduler_gamma: float = 0.1):
    """
    Trains the QAT-prepared model using a standard training loop.

    If the batch doesn't contain a "label" key, the function uses "input_ids" as the target labels,
    which is common in unsupervised language modeling tasks.

    Args:
        model (nn.Module): The QAT-prepared model.
        train_loader (DataLoader): Training data loader.
        device (str): The device for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        scheduler_step_size (int, optional): Scheduler step size.
        scheduler_gamma (float, optional): Scheduler decay factor.

    Returns:
        list: A list containing average loss per epoch.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = None
    if scheduler_step_size is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            # Move inputs to device
            inputs = {key: batch[key].to(device) for key in batch if key != "label"}
            # Use "label" if present; otherwise, use "input_ids" as the target.
            if "label" in batch:
                targets = batch["label"].to(device)
            else:
                targets = batch["input_ids"].to(device)
            
            optimizer.zero_grad()
            output = model(**inputs)
            # If output is a dict, extract "logits"
            logits = output["logits"] if isinstance(output, dict) else output
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f"QAT Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if scheduler is not None:
            scheduler.step()
    return loss_history

def convert_qat_model(model):
    """
    Converts a QAT-trained model into a fully quantized model.

    Finalizes the quantization process by replacing fake quantization modules with actual quantized implementations.

    Args:
        model (nn.Module): The model fine-tuned using QAT.

    Returns:
        nn.Module: The fully quantized model.
    """
    model.eval()
    quantized_model = quant.convert(model, inplace=False)
    return quantized_model

def plot_loss_curve(loss_history: list, title: str = "Training Loss"):
    """
    Plots the training loss curve.

    Args:
        loss_history (list): List of loss values recorded per epoch.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='--', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()

# --- Main Block Using Actual Model and Data Loaders ---
if __name__ == "__main__":
    from src.model_loader import load_model
    from src.data_loader import load_lyrics_dataset  # or load_sst2 if applicable
    from src.utils import select_device

    # Select device (MPS if available, otherwise CPU)
    device = select_device()
    print(f"Using device: {device}")

    # Load the base model and tokenizer (for "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model, tokenizer = load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", use_half_precision=False)
    print("Model loaded and in FP32 for QAT.")

    # Load dataset (using the lyrics dataset as an example)
    train_loader, val_loader = load_lyrics_dataset(tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_length=128, batch_size=4)
    print("Data loaded successfully.")

    # Prepare the model for QAT
    qat_model = prepare_qat_model(model, backend="fbgemm")
    print("Model prepared for QAT.")

    # Train the QAT model
    loss_history = train_qat_model(qat_model, train_loader, device, epochs=3, lr=1e-5, scheduler_step_size=1, scheduler_gamma=0.1)
    plot_loss_curve(loss_history, title="QAT Training Loss")
    
    # Convert the QAT model to a fully quantized model
    quantized_model = convert_qat_model(qat_model)
    print("QAT model converted to fully quantized version.")
