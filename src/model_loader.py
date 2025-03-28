import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
               use_half_precision: bool = True, 
               device: str = None):
    """
    Loads a model and its tokenizer from Hugging Face.

    Args:
        model_name (str): The Hugging Face model identifier.
        use_half_precision (bool): If True, the model is loaded in FP16 (half precision); 
                                   otherwise, in FP32.
        device (str): The target device as a string ("mps", "cuda", "cpu"). If None, auto-selects the best available device.
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Auto-detect device if not provided
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    device = torch.device(device)
    
    # Set the appropriate torch dtype
    torch_dtype = torch.float16 if use_half_precision else torch.float32

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True  # helps reduce memory usage during loading
    )
    
    # Move model to the selected device
    model.to(device)
    print(f"Model '{model_name}' loaded on {device} using dtype {torch_dtype}.")
    return model, tokenizer

if __name__ == "__main__":
    # For testing: load the model and tokenize a sample text
    model, tokenizer = load_model()
    sample_text = "Once upon a time, in a world of endless verses,"
    inputs = tokenizer(sample_text, return_tensors="pt")
    print("Sample tokenized input:", inputs)
