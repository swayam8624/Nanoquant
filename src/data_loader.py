import os
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_lyrics_dataset(tokenizer_name: str, max_length: int = 128, batch_size: int = 16):
    """
    Loads the Genius Lyrics dataset (with keys 'id', 'artist', 'song') from Hugging Face,
    performs a train/validation split, tokenizes the 'song' field using the specified tokenizer,
    and returns DataLoaders for training and validation.
    
    Args:
        tokenizer_name (str): The name or path of the tokenizer to use.
        max_length (int): Maximum token length for tokenization.
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load the dataset
    dataset = load_dataset("tommybrenson/genius-lyrics")
    
    # Use the available keys to verify that the "song" field is present.
    ds = dataset["train"]
    print("Available keys:", ds.column_names)
    
    # Since the keys available are "id", "artist", "song", we'll use "song" as the text field.
    text_field = "song"
    
    # Perform a train/validation split (90% train, 10% validation)
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        # Tokenize the "song" field
        return tokenizer(examples[text_field], truncation=True, padding="max_length", max_length=max_length)
    
    # Apply tokenization on train and validation splits
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    
    # Set format for PyTorch tensors
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # For testing: load the dataset using a chosen tokenizer.
    tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Change if needed
    train_loader, val_loader = load_lyrics_dataset(tokenizer_name, max_length=128, batch_size=4)
    
    # Print a sample batch to verify output.
    for batch in train_loader:
        print("Sample batch keys:", batch.keys())
        print("Input IDs shape:", batch["input_ids"].shape)
        break
