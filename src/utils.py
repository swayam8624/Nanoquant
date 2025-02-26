import os
import torch
import logging
import matplotlib.pyplot as plt

def setup_logger(log_file: str = None, level=logging.INFO):
    """
    Set up a logger that outputs to the console and optionally to a file.
    
    Args:
        log_file (str): Optional path for the log file.
        level: Logging level (default: logging.INFO).
        
    Returns:
        logger: A configured logger instance.
    """
    logger = logging.getLogger("NanoQuant")
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler, if a log_file path is provided
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def save_checkpoint(model, optimizer, epoch: int, checkpoint_path: str):
    """
    Save model and optimizer states as a checkpoint.
    
    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch (int): Current epoch number.
        checkpoint_path (str): File path for saving the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path: str, device):
    """
    Load a checkpoint into the model and optimizer.
    
    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        checkpoint_path (str): Path to the checkpoint file.
        device: Device on which to map the checkpoint.
        
    Returns:
        epoch: The epoch number from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return epoch

def ensure_dir(directory: str):
    """
    Ensure that the specified directory exists. If not, create it.
    
    Args:
        directory (str): The directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

def plot_loss_curve(loss_history: list, title: str = "Training Loss", xlabel: str = "Epoch", ylabel: str = "Loss"):
    """
    Plot the training loss curve.
    
    Args:
        loss_history (list): A list of loss values recorded over epochs.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linestyle="--", color="blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def get_device():
    """
    Determines and returns the appropriate device.
    
    Returns:
        torch.device: 'mps' if available (for Mac), else 'cuda' if available, else 'cpu'.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
