import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def custom_quantize_layer(layer: nn.Module, calibration_input: torch.Tensor, num_levels: int = 256, lr: float = 1e-3, epochs: int = 5):
    """
    Applies custom quantization to a given linear layer by learning a scale factor.
    
    The approach is to minimize the KL divergence between the softmax distributions
    of the full-precision output and the quantized output on a calibration input.
    
    Args:
        layer (nn.Module): The linear layer to be quantized.
        calibration_input (torch.Tensor): Input tensor used for calibration.
        num_levels (int): Number of quantization levels (default: 256).
        lr (float): Learning rate for optimizing the scale factor (default: 1e-3).
        epochs (int): Number of calibration epochs (default: 5).
    
    Returns:
        tuple: The quantized layer and the learned scale factor.
    """
    # Clone the full-precision weights of the layer.
    weight = layer.weight.data.clone()
    
    # Initialize a learnable scale parameter 's'. We start with the maximum absolute value of the weights
    # divided by (num_levels/2 - 1), which is a common initialization for symmetric quantization.
    s = torch.tensor(weight.abs().max() / ((num_levels / 2) - 1), device=weight.device, requires_grad=True)
    
    optimizer = optim.Adam([s], lr=lr)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    kl_losses = []
    
    # Define the quantization range for symmetric quantization.
    qmin = -(num_levels // 2)
    qmax = num_levels // 2 - 1

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Quantize the weights using the current scale 's'.
        quantized_weights = torch.clamp(torch.round(weight / s), qmin, qmax) * s
        
        # Temporarily replace the layer's weights with the quantized weights.
        original_weight = layer.weight.data.clone()
        layer.weight.data.copy_(quantized_weights)
        
        # Compute the full-precision output using the original weight (using a detached computation).
        with torch.no_grad():
            full_output = nn.functional.linear(calibration_input, weight, layer.bias)
        # Compute the output with quantized weights.
        quantized_output = layer(calibration_input)
        
        # Compute the softmax distributions; add a small epsilon for numerical stability.
        eps = 1e-8
        full_dist = torch.log_softmax(full_output, dim=-1)
        quantized_dist = torch.softmax(quantized_output, dim=-1) + eps
        
        # Compute the KL divergence loss between the full-precision and quantized output distributions.
        loss = kl_loss_fn(full_dist, quantized_dist)
        loss.backward()
        optimizer.step()
        
        # Restore the original weights for the next iteration.
        layer.weight.data.copy_(original_weight)
        
        kl_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} - KL Loss: {loss.item():.4f}, Learned Scale: {s.item():.4f}")
    
    # After calibration, apply the final quantization permanently.
    final_quantized_weights = torch.clamp(torch.round(weight / s), qmin, qmax) * s
    layer.weight.data.copy_(final_quantized_weights)
    
    # Plot the calibration loss curve.
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), kl_losses, marker='x', linestyle='-', color='r')
    plt.xlabel("Calibration Epoch")
    plt.ylabel("KL Loss")
    plt.title("Custom Quantization Calibration Loss")
    plt.grid(True)
    plt.show()
    
    return layer, s

if __name__ == "__main__":
    # For demonstration purposes, define a dummy linear layer.
    linear_layer = nn.Linear(768, 10)
    
    # Create a dummy calibration input with appropriate shape.
    calibration_input = torch.randn(1, 768)
    
    # Apply the custom quantization routine to the dummy linear layer.
    quantized_layer, learned_scale = custom_quantize_layer(linear_layer, calibration_input, num_levels=256, lr=1e-3, epochs=5)
    
    print(f"Custom quantization completed. Final learned scale factor: {learned_scale.item():.4f}")
