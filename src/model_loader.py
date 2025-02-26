import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str = "Qwen/Qwen2.5-Math-7B", use_half_precision: bool = True, device: str = None):
    """
    Loads the DeepSeek R1 model and its tokenizer.

    Args:
        model_name (str): The name or path of the model to load.
        use_half_precision (bool): If True, loads the model with float16 precision.
        device (str): The target device to load the model onto. If None, it defaults to 'mps' on Macs if available, otherwise 'cpu'.

    Returns:
        model: The loaded AutoModelForCausalLM model.
        tokenizer: The corresponding AutoTokenizer.
    """
    # Automatically determine device if not provided
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Select the appropriate data type
    dtype = torch.float16 if use_half_precision else torch.float32

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    # Move the model to the specified device
    model.to(device)
    print(f"Model loaded on device: {next(model.parameters()).device}")

    return model, tokenizer

if __name__ == "__main__":
    # When run directly, load the model and tokenizer and print confirmation.
    model, tokenizer = load_model()
    print("DeepSeek R1 model and tokenizer loaded successfully!")
