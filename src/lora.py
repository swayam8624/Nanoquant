import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import LoRA configuration and wrapper from the PEFT library.
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Please install the 'peft' package via pip (pip install peft).")

def apply_lora(model: nn.Module, 
               target_modules: list = None, 
               r: int = 8, 
               lora_alpha: int = 32, 
               lora_dropout: float = 0.1, 
               bias: str = "none") -> nn.Module:
    """
    Applies LoRA to the given model by wrapping selected target modules with low-rank adapters.
    
    Args:
        model (nn.Module): The model to adapt.
        target_modules (list, optional): List of module names to apply LoRA. 
            Default is None, which will set a standard list (e.g., projection layers).
        r (int): The low-rank dimension.
        lora_alpha (int): Scaling factor for LoRA.
        lora_dropout (float): Dropout probability applied to LoRA layers.
        bias (str): Whether to adapt biases ("none" means do not adapt biases).
        
    Returns:
        nn.Module: The model wrapped with LoRA adapters.
    """
    # For many transformer-based models, typical target modules might be ["q_proj", "v_proj", "o_proj"].
    # For a custom model or different architecture, this list can be adjusted.
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "o_proj"]
    
    # Create a LoRA configuration with the specified parameters.
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias
    )
    
    # Wrap the original model with the LoRA configuration.
    model_lora = get_peft_model(model, lora_config)
    return model_lora

def train_lora_model(model: nn.Module, train_loader, device: str, epochs: int = 1, lr: float = 1e-5):
    """
    Fine-tunes the LoRA-adapted model on a provided training dataset.
    
    This training loop iterates over batches, computes loss using cross-entropy,
    and updates only the trainable parameters (LoRA adapters) while keeping the rest fixed.
    
    Args:
        model (nn.Module): The LoRA-adapted model.
        train_loader: DataLoader that provides training batches.
        device (str): Target device (e.g., "cpu", "mps", "cuda").
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
    
    Returns:
        list: Training loss history over epochs.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            # Assume batch is a dict with keys 'input_ids', 'attention_mask', and 'label'.
            inputs = {key: batch[key].to(device) for key in batch if key != "label"}
            targets = batch["label"].to(device)
            optimizer.zero_grad()
            # Forward pass: get the model's output.
            output = model(**inputs)
            # If the output has a 'logits' attribute, use it; otherwise, use the output directly.
            logits = output.logits if hasattr(output, "logits") else output
            # Reshape outputs and targets if needed.
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f"LoRA Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Plot the training loss curve.
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LoRA Fine-Tuning Loss")
    plt.grid(True)
    plt.show()

    return loss_history


# --- Testing the LoRA Module with a Dummy Model and Data ---
if __name__ == "__main__":
    # Define a simple dummy model for demonstration.
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(768, 10)
        
        def forward(self, **kwargs):
            # Extract the 'input_ids' from kwargs.
            x = kwargs.get("input_ids", None)
            if x is None:
                raise ValueError("Expected keyword argument 'input_ids'")
            return self.fc(x)
    
    # Instantiate the dummy model.
    dummy_model = DummyModel()
    # For the dummy model, we target the "fc" layer for LoRA adaptation.
    dummy_lora_model = apply_lora(dummy_model, target_modules=["fc"])
    print("LoRA adaptation applied to dummy model.")
    
    # Create dummy training data:
    # Each example is a dict with an "input_ids" key and a "label" key.
    dummy_data = [{"input_ids": torch.randn(1, 768), "label": torch.tensor([1])} for _ in range(32)]
    dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=8, shuffle=True)
    
    # Specify device and run the training loop.
    device = "cpu"  # Change to "mps" or "cuda" as available.
    train_lora_model(dummy_lora_model, dummy_loader, device, epochs=3, lr=1e-3)
    
    print("LoRA model training complete.")
