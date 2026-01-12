from gpt2rewardmodel  import GPT2RewardModel
import torch
from gpt2_min import GPTConfig
# Import the new wrapper
# from reward_model_wrapper import GPT2RewardModel 

# Initialize
config = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768) # Set your config
reward_model = GPT2RewardModel(config)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)

def compute_rm_loss_custom(model, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # 1. Get full sequence rewards
    chosen_seq_rewards = model(chosen_ids)     # (B, L)
    rejected_seq_rewards = model(rejected_ids) # (B, L)
    
    # 2. Select the reward at the last valid token position
    # Assumes Right Padding
    last_chosen_idx = chosen_mask.sum(dim=1) - 1
    last_rejected_idx = rejected_mask.sum(dim=1) - 1
    
    batch_idx = torch.arange(chosen_ids.size(0), device=chosen_ids.device)
    
    chosen_score = chosen_seq_rewards[batch_idx, last_chosen_idx]
    rejected_score = rejected_seq_rewards[batch_idx, last_rejected_idx]
    
    # 3. Compute Loss
    loss = -torch.log(torch.sigmoid(chosen_score - rejected_score)).mean()
    return loss
