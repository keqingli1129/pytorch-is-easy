import torch

# A minimal tokenizer for our example
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            '<pad>': 0, 'The': 1, 'quick': 2, 'brown': 3, 'fox': 4,
            'jumps': 5, 'over': 6, 'lazy': 7, 'dog': 8,
            '<|user|>': 9, '<|assistant|>': 10, '<|end|>': 11,
            'What': 12, 'is': 13, 'a': 14, '?': 15
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        # A simple split-based tokenizer, robust to punctuation
        tokens = text.replace('?', ' ?').split()
        return [self.vocab[t] for t in tokens]

    def decode(self, tensor):
        return " ".join([self.inv_vocab[i] for i in tensor.tolist()])

# Instantiate our tokenizer
tokenizer = SimpleTokenizer()

# A sample batch of data (list of dictionaries)
# Note: For simplicity, our responses are the same length.
sft_batch = [
    {"prompt": "The quick brown fox", "response": "jumps over The lazy dog"},
    {"prompt": "What is a dog ?", "response": "a lazy brown fox"},
]

def sft_data_collator(batch, tokenizer):
    all_input_ids = []
    all_labels = []

    for example in batch:
        # 1. Format the text with the chat template.
        prompt_part = f"<|user|> {example['prompt']} <|end|> <|assistant|>"
        full_text = f"{prompt_part} {example['response']} <|end|>"

        # 2. Tokenize the prompt part to find the masking boundary.
        # This tells us how many tokens to ignore in the loss calculation.
        prompt_ids = tokenizer.encode(prompt_part)
        mask_until_idx = len(prompt_ids)

        # 3. Tokenize the full text for the model's input.
        input_ids = tokenizer.encode(full_text)

        # 4. Create labels by cloning the input_ids.
        labels = torch.tensor(input_ids).clone()

        # 5. Apply the mask. This is the core SFT trick.
        # We set the label for all prompt tokens to -100.
        labels[:mask_until_idx] = -100

        all_input_ids.append(torch.tensor(input_ids))
        all_labels.append(labels)

    # In a real implementation, you'd pad all sequences to the same length here.
    return {
        "input_ids": torch.stack(all_input_ids),
        "labels": torch.stack(all_labels)
    }

# Let's process our batch
prepared_batch = sft_data_collator(sft_batch, tokenizer)

# Assume 'policy_model' is our LLM and 'optimizer' is an AdamW optimizer.
# The model's forward pass is expected to return (logits, loss).
def sft_training_step(policy_model, optimizer, batch):
    policy_model.train()
    optimizer.zero_grad()

    # The model's forward pass automatically calculates the masked loss
    # because PyTorch's cross_entropy ignores labels with value -100.
    outputs = policy_model(
        input_ids=batch["input_ids"],
        labels=batch["labels"]
    )
    loss = outputs.loss # Assuming a Hugging Face-style model output

    loss.backward()
    optimizer.step()
    return loss.item()

# Conceptual usage:
# loss_value = sft_training_step(my_gpt_model, my_optimizer, prepared_batch)


def main():
    # Print the prepared batch for inspection
    print("Prepared batch:")
    print("Input IDs:\n", prepared_batch["input_ids"])
    print("Labels:\n", prepared_batch["labels"])

if __name__ == "__main__":
    main()

    