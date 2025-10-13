import torch
import torch.nn as nn
from gptconfig import GPTConfig

class Embedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.embedding(x) + self.positional_encoding(pos)
    

if __name__ == "__main__":
    config = GPTConfig(vocab_size=1000, block_size=16, n_embd=768)  # Example values
    model = Embedding(config)
    x = torch.randint(0, config.vocab_size, (1, config.block_size))
    print(x)
    print(x.shape)
    out = model(x)
    print(out.shape)  # should be (1, block_size, n_embd)