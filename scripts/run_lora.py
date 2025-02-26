#!/usr/bin/env python
import sys
import os

# Ensure the project root is in the module search path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_loader import load_model
from src.data_loader import load_sst2
from src.lora import apply_lora, train_lora_model
from src.utils import get_device, ensure_dir

def main():
    # Determine the best available device.
    device = get_device()
    print(f"Using device: {device}")

    # Load the DeepSeek R1 model and its tokenizer.
    # For LoRA adaptation, you can opt for full precision.
    model, tokenizer = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=False)
    print("Model loaded successfully.")

    # Apply LoRA adaptation to the model.
    # Specify target modules appropriate for your model architecture.
    lora_model = apply_lora(model, target_modules=["q_proj", "v_proj", "o_proj"])
    print("LoRA adaptation applied to the model.")

    # Load the SST-2 dataset (or another benchmark) via the data loader.
    # load_sst2 returns (train_loader, val_loader, test_loader).
    train_loader, val_loader, test_loader = load_sst2(tokenizer_name="Qwen/Qwen2.5-Math-7B", max_length=128, batch_size=16)
    print("Data loaded successfully.")

    # Fine-tune the LoRA-adapted model using the training loop.
    print("Starting LoRA fine-tuning...")
    train_loss_history = train_lora_model(lora_model, train_loader, device, epochs=3, lr=1e-5)
    
    # Save the fine-tuned LoRA model and its tokenizer.
    output_dir = "models/lora_model"
    ensure_dir(output_dir)
    lora_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA-adapted model saved successfully to {output_dir}")

if __name__ == "__main__":
    main()
