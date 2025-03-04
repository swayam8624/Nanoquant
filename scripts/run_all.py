#!/usr/bin/env python
import sys
import os

# Ensure the project root (one directory up) is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_loader import load_model
from src.data_loader import load_sst2
from src.qat import prepare_qat_model, train_qat_model, convert_qat_model
from src.lora import apply_lora, train_lora_model
from src.pruning import prune_model
from src.utils import get_device, ensure_dir, plot_loss_curve

def main():
    # 1. Device Setup & Model Loading
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the base DeepSeek R1 model and its tokenizer (using half precision for QAT)
    model, tokenizer = load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", use_half_precision=False)
    print("Loaded base model.")
    
    # Load dataset (SST-2) for training/fine-tuning
    train_loader, val_loader, test_loader = load_sst2(tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_length=128, batch_size=16)
    print("Data loaded successfully.")
    
    # 2. QAT Pipeline
    print("\n--- Starting QAT Pipeline ---")
    # Prepare the model for QAT
    model.train()
    qat_model = prepare_qat_model(model, backend="fbgemm")
    print("Model prepared for QAT.")
    
    # Fine-tune the QAT-prepared model
    qat_loss_history = train_qat_model(qat_model, train_loader, device, epochs=3, lr=1e-5, scheduler_step_size=1, scheduler_gamma=0.1)
    plot_loss_curve(qat_loss_history, title="QAT Training Loss")
    
    # Convert the QAT model to a fully quantized model
    quantized_model = convert_qat_model(qat_model)
    print("Model converted to fully quantized version.")
    
    # 3. LoRA Pipeline
    print("\n--- Starting LoRA Pipeline ---")
    # Apply LoRA adaptation on the quantized model.
    # (Here we assume the quantized model still supports fine-tuning of the LoRA adapters.)
    lora_model = apply_lora(quantized_model, target_modules=["q_proj", "v_proj", "o_proj"])
    print("LoRA adaptation applied to the quantized model.")
    
    # Fine-tune the LoRA-adapted model
    lora_loss_history = train_lora_model(lora_model, train_loader, device, epochs=3, lr=1e-5)
    plot_loss_curve(lora_loss_history, title="LoRA Fine-Tuning Loss")
    
    # 4. Pruning Pipeline
    print("\n--- Starting Pruning Pipeline ---")
    # Define target module substrings that should be pruned (adjust as needed for your model's architecture)
    target_modules = ["dense", "proj", "fc"]
    pruned_model = prune_model(lora_model, target_module_names=target_modules, amount=0.3, method='l1_unstructured')
    print("Sparse pruning applied to the model.")
    
    # 5. Save the Final Model
    output_dir = "models/final_model"
    ensure_dir(output_dir)
    pruned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model (quantized, LoRA-adapted, and pruned) saved at {output_dir}")

if __name__ == "__main__":
    main()
