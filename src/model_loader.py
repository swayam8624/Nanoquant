import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", use_half_precision: bool = False, device: str = "mps"):
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
    
      
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        attn_implementation = "sdpa"  # Best available for MPS
        print("[INFO] Using MPS with SDPA")
    else:
        device = torch.device("cpu")
        attn_implementation = None  # No special attention on CPU
        print("[WARNING] Using CPU, expect slower performance")
    
    # Select the appropriate data type
    dtype = torch.bfloat16 if use_half_precision else torch.float32

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        attn_implementation=attn_implementation
    )
    
    # Move the model to the specified device
    model.to(device)

    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    print(round(model_mem_gb, 2))
    
    for param in model.parameters():
    # Check if parameter dtype is  Half (float16)
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)

    return model, tokenizer

if __name__ == "__main__":
    # When run directly, load the model and tokenizer and print confirmation.
    model, tokenizer = load_model()
    print("DeepSeek R1 model and tokenizer loaded successfully!")
