import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
import matplotlib.pyplot as plt

def prepare_qat_model(model, backend: str = "fbgemm"):
    """
    Prepares a model for Quantization-Aware Training (QAT).

    This function sets the model’s quantization configuration (qconfig) based on the selected backend
    (e.g., "fbgemm" for x86 systems) and inserts fake quantization modules and observers into the model.
    These modules simulate the effects of quantization during training, enabling the model to adapt to
    lower precision.

    Args:
        model (nn.Module): The pre-trained model to prepare for QAT.
        backend (str): The quantization backend to use (default is "fbgemm").

    Returns:
        nn.Module: The model prepared for QAT.
    """
    # Set the model's qconfig to the default QAT configuration for the chosen backend.
    model.qconfig = quant.get_default_qat_qconfig(backend)
    # Prepare the model for QAT; this inserts fake quantization and observer modules.
    model_prepared = quant.prepare_qat(model, inplace=False)
    return model_prepared

def train_qat_model(model, train_loader, device: str, epochs: int = 1, lr: float = 1e-5,
                    scheduler_step_size: int = None, scheduler_gamma: float = 0.1):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = None
    if scheduler_step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            inputs = {key: batch[key].to(device) for key in batch if key != "label"}
            targets = batch["label"].to(device)
            optimizer.zero_grad()
            output = model(**inputs)
            # Adjusted here:
            logits = output["logits"] if isinstance(output, dict) else output
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f"QAT Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        if scheduler is not None:
            scheduler.step()
    return loss_history


def convert_qat_model(model):
    """
    Converts a QAT-trained model into a fully quantized model.

    Once training with QAT is complete, this function finalizes the quantization process. It converts
    the model by replacing the fake quantization modules with actual quantized implementations,
    reducing the model size and potentially improving inference efficiency.

    Args:
        model (nn.Module): A model that has been fine-tuned using QAT.

    Returns:
        nn.Module: The fully quantized model.
    """
    model.eval()
    quantized_model = quant.convert(model, inplace=False)
    return quantized_model

# --- Testing the QAT Module with a Dummy Model ---
if __name__ == "__main__":
    # Define a simple dummy model for demonstration purposes.
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, **kwargs):
            # Extract "input_ids" from kwargs. Raise an error if not provided.
            x = kwargs.get("input_ids", None)
            if x is None:
                raise ValueError("Expected keyword argument 'input_ids'")
            # Compute logits using the linear layer.
            logits = self.fc(x)
            # Return output in dictionary form to mimic Hugging Face models.
            return {"logits": logits}


    
    # Instantiate the dummy model.
    dummy_model = DummyModel()
    # Prepare the dummy model for QAT.
    dummy_model_prepared = prepare_qat_model(dummy_model)
    
    # Create a dummy dataset consisting of 32 random samples.
    dummy_data = [{"input_ids": torch.randn(1, 10), "label": torch.tensor([1])} for _ in range(32)]
    dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=8, shuffle=True)
    
    # Specify the device and run the training loop for a few epochs.
    device = "cpu"
    train_qat_model(dummy_model_prepared, dummy_loader, device, epochs=3, lr=1e-3)
    
    # Convert the trained QAT model to a fully quantized model.
    quantized_dummy = convert_qat_model(dummy_model_prepared)
    print("QAT dummy model converted successfully!")
