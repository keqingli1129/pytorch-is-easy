# gpt_for_generation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt.gptconfig import GPTConfig
from gpt.gpt2_min import CausalSelfAttention, MLP, Block

# --- Boilerplate Transformer Blocks (Self-Attention, MLP, etc.) ---
# (Full implementation code as provided in the prompt)
# class CausalSelfAttention(nn.Module): ...
# class MLP(nn.Module): ...
# class Block(nn.Module): ...

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # The main body of the transformer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # The 'language model head' for generation
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight # Weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Standard transformer forward pass
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # --- The key part for generation ---
        # Project the final hidden states to vocabulary size
        logits = self.lm_head(x) # Shape: (Batch, SeqLen, VocabSize)
        return logits