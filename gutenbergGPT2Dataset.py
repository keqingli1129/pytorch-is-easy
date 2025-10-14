import torch
from torch.utils.data import Dataset

class GutenbergGPT2Dataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        # Concatenate all texts and tokenize
        all_text = " ".join(hf_dataset["TEXT"])
        tokens = tokenizer.encode(all_text)
        # Truncate to a multiple of block_size
        n = (len(tokens) // block_size) * block_size
        self.tokens = torch.tensor(tokens[:n], dtype=torch.long)
        self.num_samples = len(self.tokens) // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]
        return x, y

# Usage example (not part of the file):
# dataset = GutenbergGPT2Dataset(hf_dataset, tokenizer, block_size=128)