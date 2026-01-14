import torch
import torch.nn.functional as F
from gpt2_min import GPT2, GPTConfig

# 1. Setup a tiny model and dummy vocabulary
# Vocab: 0=<pad>, 1=A, 2=B, 3=C, 4=D
vocab_map = {0: "<pad>", 1: "A", 2: "B", 3: "C", 4: "D"}
config = GPTConfig(vocab_size=5, block_size=10, n_layer=1, n_head=1, n_embd=32)
model = GPT2(config)
model.eval()

# 2. Start with a prompt: "A B" (Indices: [1, 2])
# We will generate 3 new tokens.
current_ids = torch.tensor([[1, 2]]) # Shape: [1, 2]

print(f"Start: {current_ids.tolist()} -> 'A B'")

# 3. Generation Loop (One by One)
for step in range(3):
    print(f"\n--- Step {step + 1} ---")
    
    # Forward pass (Get logits for all existing tokens)
    # Shape of logits: [Batch=1, SeqLen, VocabSize=5]
    logits, _ = model(current_ids)
    
    # We only care about the prediction for the LAST token
    last_token_logits = logits[:, -1, :]
    
    # Greedy decoding: Pick the token with the highest score
    # (In real life we sample, but let's be deterministic here)
    next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)
    
    print(f"Logits for next token: {last_token_logits.detach().numpy()}")
    print(f"Selected Token ID: {next_token.item()} ('{vocab_map.get(next_token.item(), '?')}')")
    
    # Append to the sequence
    current_ids = torch.cat((current_ids, next_token), dim=1)
    
    print(f"Updated Sequence: {current_ids.tolist()}")

# 4. Demonstrate the "Last Token Index" Logic
print("\n\n--- Masking & Indexing Demo ---")

# Let's say we have a batch of 2 sequences with padding
# Seq 1: "A B C <pad>" (Length 3)
# Seq 2: "A B C D"     (Length 4)
batch_ids = torch.tensor([
    [1, 2, 3, 0], 
    [1, 2, 3, 4]
])
attention_mask = torch.tensor([
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])

print(f"Batch IDs:\n{batch_ids}")
print(f"Mask:\n{attention_mask}")

# Calculate Lengths
lengths = attention_mask.sum(dim=1)
print(f"Lengths (sum of mask): {lengths.tolist()}")

# Calculate Last Indices
last_indices = lengths - 1
print(f"Last Indices (Length - 1): {last_indices.tolist()}")

# Retrieve the last real token using these indices
# We use 'gather' or advanced indexing for this
batch_indices = torch.arange(batch_ids.size(0)) # [0, 1]
last_tokens = batch_ids[batch_indices, last_indices]

print(f"Last Tokens Retrieved: {last_tokens.tolist()}")
print(f"Seq 1 Last Token: {vocab_map[last_tokens[0].item()]}")
print(f"Seq 2 Last Token: {vocab_map[last_tokens[1].item()]}")