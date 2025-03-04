#!/usr/bin/env python
import os
import torch
from src.model_loader import load_model
from src.data_loader import load_sst2
from src.qat import prepare_qat_model, train_qat_model, convert_qat_model
from src.utils import get_device, ensure_dir

def main():
    # Determine the best available device.
    device = get_device()
    print(f"Using device: {device}")

    # Load the DeepSeek R1 model and its tokenizer.
    # Adjust use_half_precision as desired.
    model, tokenizer = load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", use_half_precision=False)
    
    # Prepare the model for Quantization-Aware Training (QAT).
    qat_model = prepare_qat_model(model, backend="fbgemm")
    print("Model prepared for QAT.")

    # Load the SST-2 dataset.
    # load_sst2 returns (train_loader, val_loader, test_loader)
    train_loader, val_loader, test_loader = load_sst2(tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_length=128, batch_size=16)
    print("Data loaded successfully.")

    # Fine-tune the QAT-prepared model.
    print("Starting QAT fine-tuning...")
    train_loss_history = train_qat_model(qat_model, train_loader, device, epochs=3, lr=1e-5, scheduler_step_size=1, scheduler_gamma=0.1)
    
    # Convert the QAT model to a fully quantized model.
    quantized_model = convert_qat_model(qat_model)
    print("Model converted to fully quantized version.")

    # Save the quantized model and tokenizer.
    output_dir = "models/quantized_model"
    ensure_dir(output_dir)
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved successfully to {output_dir}")

if __name__ == "__main__":
    main()
