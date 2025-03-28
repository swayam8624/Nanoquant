import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple

# Relative imports for package structure
from .model_loader import load_model
from .data_loader import load_lyrics_dataset
from .utils import select_device

def custom_quantize_layer(
    layer: nn.Module,
    calibration_input: torch.Tensor,
    num_levels: int = 256,
    lr: float = 1e-3,
    epochs: int = 5
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Optimized version with proper weight
    handling and gradient flow
    """
    # Store original weights and setup scale parameter
    original_weight = layer.weight.data.clone()
    device = original_weight.device
    
    # Initialize scale parameter using max weight magnitude
    s = torch.tensor(
        original_weight.abs().max() / ((num_levels / 2) - 1),
        device=device,
        requires_grad=True
    )
    
    optimizer = optim.Adam([s], lr=lr)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    kl_losses = []
    
    # Quantization bounds
    qmin, qmax = -(num_levels // 2), num_levels // 2 - 1

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Quantize weights without modifying layer
        with torch.no_grad():
            quantized_weights = torch.clamp(
                torch.round(original_weight / s),
                qmin, qmax
            ) * s

        # Compute outputs
        full_output = nn.functional.linear(
            calibration_input, original_weight, layer.bias
        )
        quantized_output = nn.functional.linear(
            calibration_input, quantized_weights, layer.bias
        )

        # KL Divergence with numerical stability
        full_dist = torch.log_softmax(full_output, dim=-1)
        quantized_dist = torch.softmax(quantized_output, dim=-1) + 1e-8
        
        loss = kl_loss_fn(full_dist, quantized_dist)
        loss.backward()
        optimizer.step()
        
        kl_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Scale: {s.item():.4f}")

    # Apply final quantization
    with torch.no_grad():
        final_weights = torch.clamp(
            torch.round(original_weight / s),
            qmin, qmax
        ) * s
        layer.weight.data.copy_(final_weights)

    # Visualization
    plt.figure(figsize=(8, 4))
    plt.plot(kl_losses, marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("Quantization Calibration Progress")
    plt.grid(True)
    plt.show()

    return layer, s

if __name__ == "__main__":
    # Initialize device and model
    device = select_device()
    model, tokenizer = load_model(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_half_precision=False
    )
    
    # Load calibration data
    train_loader, _ = load_lyrics_dataset(
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_length=128,
        batch_size=4
    )
    calibration_input = next(iter(train_loader))["input_ids"].to(device)

    # Target first linear layer
    target_layer = next(
        module for module in model.modules()
        if isinstance(module, nn.Linear)
    )

    # Quantize and validate
    quant_layer, scale = custom_quantize_layer(
        target_layer, calibration_input
    )
    print(f"Quantization complete. Final scale: {scale.item():.4f}")