import torch
import torch.nn as nn
from gpt2_min import GPT2, GPTConfig

class GPT2RewardModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 1. Initialize the backbone model
        self.backbone = GPT2(config)
        
        # 2. Add the Value Head (Reward Head)
        # This projects the hidden state (n_embd) to a single scalar reward
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, idx, attention_mask=None):
        """
        Args:
            idx: (Batch, SeqLen) tensor of token indices
            attention_mask: (Batch, SeqLen) tensor (optional, for compatibility)
        Returns:
            rewards: (Batch, SeqLen) tensor of scalar rewards
        """
        B, T = idx.size()
        
        # --- 1. Get Hidden States from Backbone ---
        # We need to access the hidden states before the lm_head.
        # Looking at your gpt2_min.py, the forward() method returns logits directly.
        # We need to replicate the forward pass logic here to get 'x' (the hidden state).
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Forward pass through embeddings and blocks (copied from GPT2.forward)
        x = self.backbone.wte(idx) + self.backbone.wpe(pos)
        x = self.backbone.drop(x)
        
        for block in self.backbone.h:
            x = block(x)
            
        x = self.backbone.ln_f(x)
        # x is now shape (Batch, SeqLen, n_embd)
        
        # --- 2. Project to Reward ---
        rewards = self.value_head(x) # Shape: (Batch, SeqLen, 1)
        rewards = rewards.squeeze(-1) # Shape: (Batch, SeqLen)
        
        return rewards

# Example Usage within your training loop context:
if __name__ == "__main__":
    # Initialize config and model
    config = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    reward_model = GPT2RewardModel(config)
    
    # Example input
    idx = torch.randint(0, 50257, (2, 32)) # Batch=2, SeqLen=32
    
    # Get rewards
    output = reward_model(idx)
    print(f"Output Shape: {output.shape}") # Should be [2, 32]
