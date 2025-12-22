import torch
from torch.utils.data import Dataset

class GutenbergGPT2Dataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer

        all_text = " ".join(hf_dataset["TEXT"])
        tokens = tokenizer.encode(all_text)

        # Ensure we can form x of length block_size and y of length block_size (shifted by 1)
        # That requires at least block_size + 1 tokens.
        n_full = (max(0, len(tokens) - 1) // block_size) * block_size
        # Keep n_full + 1 tokens so y can shift by 1
        tokens = tokens[: n_full + 1]

        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.num_samples = n_full // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]                # shape: [block_size]
        y = self.tokens[start + 1: end + 1]       # shape: [block_size]
        return x, y


# Usage example (not part of the file):
# dataset = GutenbergGPT2Dataset(hf_dataset, tokenizer, block_size=128)