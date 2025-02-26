import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_sst2(tokenizer_name: str, max_length: int = 128, batch_size: int = 16):
    """
    Load the SST-2 dataset from the GLUE benchmark, tokenize it, and return DataLoaders.
    
    SST-2 is widely used for sentiment analysis and offers a balanced dataset to judge 
    the impact of model optimization techniques on performance.
    
    Args:
        tokenizer_name (str): The name or path of the tokenizer.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for creating DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader (DataLoader): PyTorch DataLoaders for training,
        validation, and testing.
    """
    # Load the SST-2 dataset from GLUE
    dataset = load_dataset("glue", "sst2")
    
    # Initialize the tokenizer using the provided tokenizer name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Define a tokenization function that processes each example's sentence
    def tokenize_function(example):
        return tokenizer(
            example["sentence"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    # Tokenize the training, validation, and test splits
    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_val = dataset["validation"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)
    
    # Set the format to PyTorch tensors
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create DataLoaders for each split
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loader functionality
    tokenizer_name = "Qwen/Qwen2.5-Math-7B"  # Use the same tokenizer as your model
    train_loader, val_loader, test_loader = load_sst2(tokenizer_name)
    
    # Print a sample batch from the training DataLoader
    for batch in train_loader:
        print("Sample batch keys:", batch.keys())
        print("Input IDs shape:", batch["input_ids"].shape)
        break
