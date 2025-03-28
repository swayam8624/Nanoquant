#!/usr/bin/env python
import sys
import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules from our project
from src.data_loader import load_sst2
from src.qat import prepare_qat_model, train_qat_model, convert_qat_model
from src.lora import apply_lora, train_lora_model
from src.pruning import prune_model
from src.utils import get_device, ensure_dir, setup_logger, plot_loss_curve

# Import Qiskit modules using the new API
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.primitives import BackendSampler

# Set up logging
logger = setup_logger(log_file="logs/run_quantum.log")

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", use_half_precision=True):
    torch.set_default_dtype(torch.float16)   

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        attn_implementation = "sdpa"
        logger.info("Using MPS with SDPA")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU, expect slower performance")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    ).to(device)

    mem_params = sum([p.nelement() * p.element_size() for p in model.parameters()])
    mem_buffers = sum([b.nelement() * b.element_size() for b in model.buffers()])
    model_mem_gb = (mem_params + mem_buffers) / (1024**3)

    logger.info(f"Model loaded on device: {device}, Memory usage: {round(model_mem_gb, 2)} GB")

    return model, tokenizer



def design_quantum_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """
    Designs a quantum circuit to serve as a quantum feature extractor.
    This circuit puts each qubit in superposition, applies an RX rotation,
    entangles qubits (if more than one) and measures all qubits.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    # Put all qubits in superposition
    for q in range(num_qubits):
        qc.h(q)
    # Apply fixed RX rotation
    for q in range(num_qubits):
        qc.rx(np.pi/4, q)
    # Entangle qubits if more than one
    if num_qubits > 1:
        qc.cx(0, 1)
    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def run_quantum_simulation(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Runs a quantum simulation using Qiskit's new API with BackendSampler.
    Transpiles the circuit for the Aer simulator, then runs it using BackendSampler.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to simulate.
        shots (int): Number of measurement shots.
    
    Returns:
        dict: The measurement counts from the simulation.
    """
    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(circuit, backend)
    sampler = BackendSampler(backend)  # No shots argument here
    job = sampler.run([transpiled_circuit])  # Run as a list
    result = job.result()
    counts = result.get_counts()
    return counts


def extract_quantum_feature(counts: dict, desired_outcome: str = "00") -> float:
    """
    Extracts a quantum feature from measurement counts by computing the probability
    of a desired outcome.
    
    Args:
        counts (dict): Measurement counts from the quantum simulation.
        desired_outcome (str): Outcome for which to compute probability.
    
    Returns:
        float: The probability of the desired outcome.
    """
    total_shots = sum(counts.values())
    feature = counts.get(desired_outcome, 0) / total_shots
    return feature

def integrate_quantum_features(classical_output: int, quantum_feature: float) -> dict:
    """
    Integrates classical prediction with the quantum-derived feature.
    
    Args:
        classical_output (int): The predicted class from the classical model.
        quantum_feature (float): The extracted quantum feature.
    
    Returns:
        dict: A dictionary containing the integrated results.
    """
    integrated_result = {
        "classical_prediction": classical_output,
        "quantum_feature": quantum_feature,
        "combined_score": 0.7 * classical_output + 0.3 * quantum_feature  # Weighted fusion
  # Simple illustrative fusion
    }
    return integrated_result

def main():
    logger.info("Starting Quantum AI integration pipeline.")
    
    # 1. Device Setup & Model Loading
    # Select MPS if available; otherwise use CPU.
    device = get_device()
    print(f"Using device: {device}")

    logger.info(f"Using device: {device}")
    model, tokenizer = load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", use_half_precision=False)
    logger.info("Loaded base model and tokenizer.")
    
    # 2. Data Loading
    train_loader, val_loader, test_loader = load_sst2(tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_length=128, batch_size=16)
    logger.info("Data loaded successfully.")
    model.train()
    
    # 3. QAT Pipeline
    logger.info("Starting QAT pipeline.")
    qat_model = prepare_qat_model(model, backend="fbgemm")
    logger.info("Model prepared for QAT.")
    qat_loss_history = train_qat_model(qat_model, train_loader, device, epochs=3, lr=1e-5, scheduler_step_size=1, scheduler_gamma=0.1)
    plot_loss_curve(qat_loss_history, title="QAT Training Loss")
    quantized_model = convert_qat_model(qat_model)
    logger.info("Model converted to fully quantized version.")
    
    # 4. LoRA Pipeline
    logger.info("Starting LoRA pipeline.")
    lora_model = apply_lora(quantized_model, target_modules=["q_proj", "v_proj", "o_proj"])
    logger.info("LoRA adaptation applied to the quantized model.")
    lora_loss_history = train_lora_model(lora_model, train_loader, device, epochs=3, lr=1e-5)
    plot_loss_curve(lora_loss_history, title="LoRA Fine-Tuning Loss")
    
    # 5. Pruning Pipeline
    logger.info("Starting Pruning pipeline.")
    target_modules = ["dense", "proj", "fc"]
    pruned_model = prune_model(lora_model, target_module_names=target_modules, amount=0.3, method='l1_unstructured')
    logger.info("Sparse pruning applied to the model.")
    
    # Save the final classical model
    final_model_dir = "models/final_quantum_ai_model"
    ensure_dir(final_model_dir)
    ensure_dir(final_model_dir)
    pruned_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final classical model saved at {final_model_dir}")
    
    # 6. Inference on a Test Sample
    sample_batch = next(iter(test_loader))
    inputs = {k: sample_batch[k].to(device) for k in sample_batch if k != "label"}
    with torch.no_grad():
        output = pruned_model(**inputs)
    logits = output["logits"] if isinstance(output, dict) else output
    if logits.ndim > 1:
        classical_prediction = torch.argmax(logits, dim=1).item()
    else:
        classical_prediction = torch.argmax(logits).item()

    logger.info(f"Classical model prediction: {classical_prediction}")
    
    # 7. Quantum Simulation Integration
    qc = design_quantum_circuit(num_qubits=2)
    logger.info("Quantum circuit designed:\n" + qc.draw(output='text'))
    counts = run_quantum_simulation(qc, shots=1024)
    logger.info(f"Quantum simulation counts: {counts}")
    quantum_feature = extract_quantum_feature(counts, desired_outcome="00")
    logger.info(f"Extracted quantum feature (probability of '00'): {quantum_feature}")
    
    # 8. Integrate Classical and Quantum Outputs
    integrated_output = integrate_quantum_features(classical_prediction, quantum_feature)
    logger.info("Final integrated output: " + str(integrated_output))
    
    # Optionally, save the integrated output to a file.
    output_file = os.path.join(final_model_dir, "integrated_output.txt")
    with open(output_file, "w") as f:
        f.write(str(integrated_output))
    logger.info(f"Integrated output saved to {output_file}")
    
    logger.info("Quantum AI integration pipeline completed successfully.")

if __name__ == "__main__":
    main()
