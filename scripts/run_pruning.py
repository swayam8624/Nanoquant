#!/usr/bin/env python
import sys
import os

# Add the project root (one directory up) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_loader import load_model
from src.data_loader import load_sst2
from src.pruning import prune_model
from src.utils import get_device, ensure_dir

def main():
    # Determine the best available device.
    device = get_device()
    print(f"Using device: {device}")

    # Load the DeepSeek R1 model and its tokenizer in full precision.
    model, tokenizer = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=False)
    print("Model loaded successfully.")

    # Define target modules for pruning.
    # Adjust target module substrings based on your model's architecture.
    target_modules = ["dense", "proj", "fc"]  # Example target names; modify as needed.
    
    # Apply sparse pruning to the model.
    pruned_model = prune_model(model, target_module_names=target_modules, amount=0.3, method='l1_unstructured')
    print("Pruning applied to the model.")

    # Optionally, load test data to evaluate the pruned model (for further evaluation, not shown here).
    # load_sst2 returns (train_loader, val_loader, test_loader)
    _, _, test_loader = load_sst2(tokenizer_name="Qwen/Qwen2.5-Math-7B", max_length=128, batch_size=16)

    # Save the pruned model and tokenizer.
    output_dir = "models/pruned_model"
    ensure_dir(output_dir)
    pruned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Pruned model saved successfully to {output_dir}")

if __name__ == "__main__":
    main()
