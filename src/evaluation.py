import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, data_loader, device, model_name="Model"):
    """
    Evaluates the given model on the test dataset.
    Computes accuracy, latency, and generates a confusion matrix.
    
    Args:
        model: The trained model.
        data_loader: DataLoader for the evaluation/test data.
        device: Device to run evaluation on.
        model_name (str): Name of the model (for display purposes).
    
    Returns:
        Tuple (accuracy, avg_latency, conf_matrix).
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Assuming each batch is a dictionary with keys like "input_ids" and "label"
            inputs = {k: batch[k].to(device) for k in batch if k != "label"}
            labels = batch["label"].to(device)
            
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # If the model returns a dict, extract "logits"; otherwise, use the output directly.
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_latency = np.mean(inference_times) * 1000  # in milliseconds
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)
    
    print(f"\nEvaluation Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_latency:.2f} ms per sample")
    print("\nClassification Report:\n", class_report)
    
    # Plot the confusion matrix.
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    return accuracy, avg_latency, conf_matrix

if __name__ == "__main__":
    # Import the model loader and data loader functions from our project.
    from model_loader import load_model
    from data_loader import load_sst2

    # For testing purposes, we load the SST-2 test split.
    tokenizer_name = "Qwen/Qwen2.5-Math-7B"
    # load_sst2 returns (train_loader, val_loader, test_loader)
    _, _, test_loader = load_sst2(tokenizer_name, max_length=128, batch_size=16)
    
    # Set up the device.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load models.
    # In a real project these would be different saved checkpoints.
    full_precision_model, _ = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=False)
    quantized_model, _ = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=True)
    pruned_model, _ = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=False)
    lora_model, _ = load_model(model_name="Qwen/Qwen2.5-Math-7B", use_half_precision=False)
    
    # Evaluate each model.
    evaluate_model(full_precision_model, test_loader, device, "Full Precision Model")
    evaluate_model(quantized_model, test_loader, device, "Quantized Model")
    evaluate_model(pruned_model, test_loader, device, "Pruned Model")
    evaluate_model(lora_model, test_loader, device, "LoRA Model")
