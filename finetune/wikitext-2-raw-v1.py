import torch
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Optional: Filter out empty lines to avoid wasting compute
dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
print(f"Number of training examples: {len(dataset)}")