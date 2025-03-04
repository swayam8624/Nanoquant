import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torch.ao.quantization import QConfig, default_embedding_qat_qconfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, PerChannelMinMaxObserver
import matplotlib.pyplot as plt
import gc

def prepare_qat_model(model):
    """MPS-compatible QAT preparation with memory optimizations"""
    model = model.float().eval()
    model.config.use_cache = False
    
    model.train()
    
    # Configure quantization for MPS compatibility
    qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,
            quant_min=0,
            quant_max=255
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_channel_symmetric
        )
    )
    
    # Apply quantization config to appropriate modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.qconfig = qconfig
        elif isinstance(module, nn.Embedding):
            module.qconfig = default_embedding_qat_qconfig
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    return quant.prepare_qat(model, inplace=False)

def train_qat_model(model, train_loader, device, epochs=3, lr=1e-5):
    """MPS-optimized training loop with memory safeguards"""
    model.to(device)
    
    # Optimizer configuration for MPS
    is_mps = str(device) == 'mps'
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=False,
        foreach=is_mps  # Use foreach implementation for MPS
    )
    
    loss_fn = nn.CrossEntropyLoss()
    loss_history = []
    accumulation_steps = 4  # Gradient accumulation
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Memory management
            if batch_idx % 5 == 0:
                gc.collect()
                if is_mps:
                    torch.mps.empty_cache()
            
            # MPS-compatible data loading
            inputs = {
                k: v.to(device, non_blocking=is_mps)
                for k, v in batch.items() if k != "label"
            }
            labels = batch["label"].to(device, non_blocking=is_mps)
            
            # Forward pass with reduced precision
            with torch.autocast(device_type='mps' if is_mps else 'cpu', enabled=is_mps):
                outputs = model(**inputs)
                logits = outputs.logits[:, 0, :]  # CLS token
                loss = loss_fn(logits, labels) / accumulation_steps
            
            # Gradient accumulation
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Memory offloading
                if is_mps:
                    for param in model.parameters():
                        param.data = param.data.cpu()
                        param.grad = None
                    model = model.to(device)
                
                total_loss += loss.item() * accumulation_steps
                print(f"Batch {batch_idx+1} Loss: {loss.item() * accumulation_steps:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
    
    return loss_history

def convert_qat_model(model):
    """Safe conversion for MPS devices"""
    model.eval()
    if str(next(model.parameters()).device) == 'mps':
        model = model.cpu()
    return quant.convert(model, inplace=False)

# Usage Example
if __name__ == "__main__":
    from src.model_loader import load_model
    from src.data_loader import load_sst2
    
    # Configure MPS memory
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.7)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model with 8-bit quantization
    model, tokenizer = load_model(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    
    # Load dataset with small batches
    train_loader = load_sst2(
        tokenizer_name=tokenizer,
        max_length=32,
        batch_size=2
    )
    
    # QAT workflow
    try:
        qat_model = prepare_qat_model(model)
        loss_history = train_qat_model(qat_model, train_loader, device)
        quantized_model = convert_qat_model(qat_model)
        print("Quantization successful!")
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        print("Recommended actions:")
        print("1. Reduce max_length to 16")
        print("2. Set batch_size=1")
        print("3. Use 4-bit quantization instead")