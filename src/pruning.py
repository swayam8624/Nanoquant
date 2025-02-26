import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

def apply_sparse_pruning(module: nn.Module, amount: float = 0.3, method: str = 'l1_unstructured') -> nn.Module:
    """
    Applies sparse pruning to a given module using the specified pruning method.
    
    Args:
        module (nn.Module): The module (e.g., nn.Linear) to be pruned.
        amount (float): The fraction of parameters to prune (default: 0.3).
        method (str): The pruning method to use ('l1_unstructured' or 'random_unstructured').
        
    Returns:
        nn.Module: The module with pruning applied.
    """
    # Check if the module has a 'weight' attribute.
    if not hasattr(module, 'weight'):
        raise ValueError("The module does not have a 'weight' attribute and cannot be pruned.")
    
    if method == 'l1_unstructured':
        prune.l1_unstructured(module, name="weight", amount=amount)
    elif method == 'random_unstructured':
        prune.random_unstructured(module, name="weight", amount=amount)
    else:
        raise ValueError(f"Pruning method {method} not supported.")
    
    return module

def remove_pruning(module: nn.Module) -> nn.Module:
    """
    Removes the pruning reparameterization from a module, finalizing the pruned weights.
    
    Args:
        module (nn.Module): The pruned module.
        
    Returns:
        nn.Module: The module with pruning removed.
    """
    # Attempt to remove the pruning reparameterization
    try:
        prune.remove(module, 'weight')
    except Exception as e:
        print("Pruning removal error:", e)
    return module

def prune_model(model: nn.Module, target_module_names: list, amount: float = 0.3, method: str = 'l1_unstructured') -> nn.Module:
    """
    Applies sparse pruning to all modules in a model whose names match the target_module_names.
    
    Args:
        model (nn.Module): The model to prune.
        target_module_names (list): List of strings. Modules with names containing one of these strings will be pruned.
        amount (float): Fraction of weights to prune in each targeted module.
        method (str): The pruning method to use.
        
    Returns:
        nn.Module: The pruned model.
    """
    for name, module in model.named_modules():
        # Check if the module's name contains any of the target substrings and if it has a 'weight' attribute.
        if any(target in name for target in target_module_names) and hasattr(module, 'weight'):
            print(f"Applying pruning to module: {name}")
            apply_sparse_pruning(module, amount=amount, method=method)
            # Optionally remove the reparameterization to finalize the pruned weights.
            remove_pruning(module)
    return model

if __name__ == "__main__":
    # --- Testing the Pruning Module with a Dummy Model ---
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
    
    # Instantiate the dummy model.
    dummy_model = DummyModel()
    
    # Create a dummy input tensor.
    input_tensor = torch.randn(16, 100)
    
    # Get output from the model before pruning.
    output_before = dummy_model(input_tensor)
    
    # Apply pruning to all modules with names containing "fc".
    target_names = ['fc']
    pruned_model = prune_model(dummy_model, target_module_names=target_names, amount=0.3)
    
    # Get output from the pruned model.
    output_after = pruned_model(input_tensor)
    
    # Print outputs for comparison.
    print("Output before pruning:", output_before)
    print("Output after pruning:", output_after)
    
    # Optional: Plot a histogram of the weights of fc1 to visualize pruning effects.
    fc1_weights = dummy_model.fc1.weight.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(fc1_weights.flatten(), bins=50, alpha=0.7, color='green')
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.title("Histogram of fc1 Weights After Pruning")
    plt.grid(True)
    plt.show()
