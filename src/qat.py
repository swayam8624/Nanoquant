import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
import matplotlib.pyplot as plt
from torch.ao.quantization import QConfig, default_embedding_qat_qconfig
from torch.ao.quantization.observer import PerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.fake_quantize import default_weight_fake_quant
from src.utils import plot_loss_curve


def prepare_qat_model(model, backend: str = "fbgemm"):
    """
    Prepares a model for Quantization-Aware Training (QAT) with correct dtypes.
    """
    # Convert model to float32 (but keep quantization in int8)
    model = model.float()
    
    # Use appropriate quantization dtypes for observers
    qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,  # Activation quantization type
            reduce_range=False
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,   # Weight quantization type
            qscheme=torch.per_channel_symmetric
        )
    )
    
    # Assign QConfig to model
    model.qconfig = qconfig
    
    # In prepare_qat_model, after setting model.qconfig
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            module.qconfig = quant.qconfig.default_embedding_qat_qconfig
        elif isinstance(module, nn.LayerNorm):
            module.qconfig = None  # Typically don't quantize LayerNorm
    
    # Prepare for QAT
    model.train()
    model_prepared = quant.prepare_qat(model, inplace=False)
    return model_prepared


def train_qat_model(model, train_loader, device: str, epochs: int = 1, lr: float = 1e-5):
    """
    Trains the QAT-prepared model with correct logits handling.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = {
                k: v.to(device) 
                for k, v in batch.items() 
                if k != "label"
            }
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            
            # Extract logits and handle sequence classification
            logits = outputs.logits if isinstance(outputs, dict) else outputs
            if logits.dim() == 3:
                logits = logits[:, 0, :]  # Use [CLS] token for classification
            
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    return loss_history


def convert_qat_model(model):
    """Converts to quantized model."""
    model.eval()
    quantized_model = quant.convert(model, inplace=False)
    return quantized_model


# --- Main Block (Fixed) ---
if __name__ == "__main__":
    from src.model_loader import load_model
    from src.data_loader import load_sst2

    # Force CPU for FBGEMM backend
    device = "cpu"  # Override to CPU (FBGEMM requires CPU)

    # Load model in FP32
    model, tokenizer = load_model(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_half_precision=False  # Ensure FP32
    )

    # Load dataset
    train_loader, _, _ = load_sst2(
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_length=128,
        batch_size=16
    )

    # Prepare QAT
    qat_model = prepare_qat_model(model)

    # Train
    loss_history = train_qat_model(
        qat_model, 
        train_loader, 
        device, 
        epochs=3, 
        lr=1e-5
    )
    plot_loss_curve(loss_history)

    # Convert to quantized model
    quantized_model = convert_qat_model(qat_model)