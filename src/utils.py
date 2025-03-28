import os
import logging
import torch
import matplotlib.pyplot as plt

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs to both console and an optional file.

    Args:
        log_file (str): Optional path for a log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: A configured logger instance.
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

    # File handler if log_file is provided
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def select_device() -> torch.device:
    """
    Selects the best available device: MPS if available, then CUDA, otherwise CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def ensure_dir(directory: str) -> None:
    """
    Ensures that the specified directory exists; creates it if it doesn't.

    Args:
        directory (str): The directory path to ensure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_loss_curve(loss_history: list, title: str = "Training Loss", xlabel: str = "Epoch", ylabel: str = "Loss") -> None:
    """
    Plots the training loss curve.

    Args:
        loss_history (list): A list of loss values recorded over epochs.
        title (str): The title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='--', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
