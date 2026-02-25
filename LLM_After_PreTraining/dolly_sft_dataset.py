"""
Dolly-15k SFT Dataset & DataLoader
====================================
Dataset: databricks/databricks-dolly-15k
  - ~15,000 high-quality instruction/response pairs
  - Categories: brainstorming, QA, summarization, classification, etc.
  - Open-source (cc-by-sa-3.0), safe for commercial use
  - Clean schema: instruction, context, response, category

This module creates a PyTorch Dataset that tokenizes each example into
(input_ids, labels) pairs with proper prompt masking for SFT, plus a
collate function for batched DataLoader usage.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class DollySFTDataset(Dataset):
    """
    A PyTorch Dataset for SFT on databricks-dolly-15k.

    Each sample is tokenized into input_ids and labels, where the prompt
    portion of the labels is masked with -100 so the model only learns
    to predict the assistant's response.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the dataset from Hugging Face Hub
        self.raw_dataset = load_dataset(
            "databricks/databricks-dolly-15k", split=split
        )
        print(f"Loaded {len(self.raw_dataset)} examples from dolly-15k ({split})")

    def __len__(self):
        return len(self.raw_dataset)

    def _format_prompt(self, instruction: str, context: str) -> str:
        """Format the user prompt, optionally including context."""
        if context and context.strip():
            return (
                f"<|user|>\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Context:\n{context}\n"
                f"<|end|>\n<|assistant|>\n"
            )
        else:
            return (
                f"<|user|>\n"
                f"### Instruction:\n{instruction}\n"
                f"<|end|>\n<|assistant|>\n"
            )

    def __getitem__(self, idx):
        example = self.raw_dataset[idx]

        instruction = example["instruction"]
        context = example["context"]
        response = example["response"]
        category = example["category"]

        # --- Build the prompt and full text ---
        prompt_text = self._format_prompt(instruction, context)
        full_text = f"{prompt_text}{response}<|end|>"

        # --- Tokenize to find the masking boundary ---
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        mask_len = len(prompt_ids)

        # --- Tokenize the full conversation ---
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = full_encoding["input_ids"]

        # --- Create labels with prompt masking ---
        labels = input_ids.copy()
        # Mask the prompt tokens (and any special tokens before them)
        # so we only compute loss on the response portion.
        mask_end = min(mask_len, len(labels))
        for i in range(mask_end):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "category": category,
        }


def sft_collate_fn(batch, pad_token_id: int = 0):
    """
    Collate function that pads input_ids and labels to the longest
    sequence in the batch. Labels are padded with -100 (ignored by loss).
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Determine the max length in this batch
    max_len = max(ids.size(0) for ids in input_ids)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids, lbl in zip(input_ids, labels):
        pad_len = max_len - ids.size(0)
        # Pad on the right
        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat([torch.ones_like(ids), torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def create_dolly_dataloader(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_length: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """
    Convenience function: builds the tokenizer, dataset, and dataloader
    in one call. Returns (dataloader, tokenizer, dataset) for inspection.
    """
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Create dataset
    dataset = DollySFTDataset(tokenizer=tokenizer, max_length=max_length)

    # 3. Create dataloader with custom collate function
    pad_token_id = tokenizer.pad_token_id
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: sft_collate_fn(batch, pad_token_id=pad_token_id),
    )

    return dataloader, tokenizer, dataset


# ── Quick demo ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Dolly-15k SFT Dataset Demo")
    print("=" * 60)

    dataloader, tokenizer, dataset = create_dolly_dataloader(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_length=512,
        batch_size=4,
    )

    # --- Inspect a single example ---
    print("\n--- Single Example ---")
    sample = dataset[0]
    print(f"Category   : {sample['category']}")
    print(f"Input IDs  : shape={sample['input_ids'].shape}")
    print(f"Labels     : shape={sample['labels'].shape}")
    print(f"Masked toks: {(sample['labels'] == -100).sum().item()} / {sample['labels'].size(0)}")
    print(f"\nDecoded input:\n{tokenizer.decode(sample['input_ids'], skip_special_tokens=False)}")

    # Show which tokens the model actually learns from
    response_ids = sample["labels"][sample["labels"] != -100]
    print(f"\nResponse the model learns:\n{tokenizer.decode(response_ids)}")

    # --- Inspect a batch ---
    print("\n--- First Batch ---")
    batch = next(iter(dataloader))
    print(f"input_ids      : {batch['input_ids'].shape}")
    print(f"labels         : {batch['labels'].shape}")
    print(f"attention_mask : {batch['attention_mask'].shape}")
    print(f"Pad token ID   : {tokenizer.pad_token_id}")
    print(f"\nTotal batches  : {len(dataloader)}")
    print(f"Total examples : {len(dataset)}")
    print("=" * 60)
