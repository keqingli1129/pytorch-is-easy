from gutenbergGPT2Dataset import GutenbergGPT2Dataset
from transformers import GPT2Tokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader


def generate_trainingset(tokenizer, block_size=128):
    # Load your dataset (adjust path as needed)
    dataset = load_from_disk("./gutenberg_english_local")

    # Initialize tokenizer (using GPT2 tokenizer as example)
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set block size (should match your GPT2 config)
    block_size = 128

    # Create PyTorch datasets for each split
    train_dataset = GutenbergGPT2Dataset(dataset["train"][0:100], tokenizer, block_size)
    val_dataset = GutenbergGPT2Dataset(dataset["validation"][0:100], tokenizer, block_size)
    test_dataset = GutenbergGPT2Dataset(dataset["test"][0:100], tokenizer, block_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = generate_trainingset()
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()