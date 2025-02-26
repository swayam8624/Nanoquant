#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute

# Add the project root to sys.path so that modules in 'src' can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules from src/
from src.model_loader import load_model
from src.data_loader import load_sst2
from src.qat import prepare_qat_model, train_qat_model, convert_qat_model
from src.lora import apply_lora, train_lora_model
from src.pruning import prune_model
from src.utils import get_device, ensure_dir, setup_logger, plot_loss_curve

# Set up logging (logs will be saved to logs/run_quantum_ai.log)
logger = setup_logger(log_file="logs/run_quantum_ai.log")

def run_quantum_simulation(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Executes a given quantum circuit using Qiskit's QASM simulator and returns the measurement counts.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to simulate.
        shots (int): Number of measurement shots.
    
    Returns:
        dict: The measurement counts from the simulation.
    """
    backend = Aer.get_backend("qasm_simulator")
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    return counts

def design_quantum_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """
    Designs a quantum circuit to serve as a quantum feature extractor.
    
    This circuit applies a Hadamard gate to each qubit to create superposition, applies a fixed RX rotation,
    entangles qubits (if more than one), and then measures all qubits.
    
    Args:
        num_qubits (int): Number of qubits to include in the circuit.
    
    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    # Put all qubits into superposition
    for q in range(num_qubits):
        qc.h(q)
    # Apply parameterized rotations (fixed angle here; can be adapted to be trainable or data-dependent)
    for q in range(num_qubits):
        qc.rx(np.pi/4, q)
    # Entangle qubits if more than one qubit is used
    if num_qubits > 1:
        qc.cx(0, 1)
    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

def extract_quantum_feature(counts: dict, desired_outcome: str = "00") -> float:
    """
    Extracts a quantum feature from the measurement counts, defined as the probability of a desired outcome.
    
    Args:
        counts (dict): The measurement counts returned from the quantum simulation.
        desired_outcome (str): The outcome (e.g., "00") for which to compute the probability.
    
    Returns:
        float: The probability of the desired outcome.
    """
    total_shots = sum(counts.values())
    feature = counts.get(desired_outcome, 0) / total_shots
    return feature

def integrate_quantum_features(classical_output: int, quantum_feature: float) -> dict:
    """
    Integrates the classical prediction and the quantum-derived feature into a final result.
    
    In this example, the integration is a simple combination, but this function can be extended to perform
    more sophisticated fusion (e.g., weighted combination or feature concatenation).
    
    Args:
        classical_output (int): The predicted class from the classical model.
        quantum_feature (float): The quantum feature extracted from simulation.
    
    Returns:
        dict: A dictionary with the integrated results.
    """
    # Example: Combine the two values into a dictionary. The combined score is an illustrative metric.
    integrated_result = {
        "classical_prediction": classical_output,
        "quantum_feature": quantum_feature,
        "combined_score": classical_output + quantum_feature
    }
    return integrated_result

def main():
    logger.info("Starting Quantum AI integration pipeline.")
    
    # 1. Device Setup & Model Loading
    device = get_device()
    logger.info(f"Using device: {device}")
    model, tokenizer = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=True)
    logger.info("Loaded base model and tokenizer.")
    
    # 2. Data Loading
    train_loader, val_loader, test_loader = load_sst2(tokenizer_name="Qwen/Qwen2.5-Math-7B", max_length=128, batch_size=16)
    logger.info("Data loaded successfully.")
    
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
    
    # Save the final model before quantum integration
    final_model_dir = "models/final_quantum_ai_model"
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
    classical_prediction = torch.argmax(logits, dim=1).item()
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
    
    # Optionally, save the integrated output to a text file for further analysis.
    output_file = os.path.join(final_model_dir, "integrated_output.txt")
    with open(output_file, "w") as f:
        f.write(str(integrated_output))
    logger.info(f"Integrated output saved to {output_file}")
    
    logger.info("Quantum AI integration pipeline completed successfully.")

if __name__ == "__main__":
    main()
