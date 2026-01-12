import torch
from torch.utils.data import Dataset, DataLoader
from reward_model import RewardModel

# Use the same model ID as in your reward_model.py
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

class PreferenceDataset(Dataset):
    """
    Dataset of preference pairs for reward model training.
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize both chosen and rejected sequences
        # We process them separately to ensure correct padding and truncation
        chosen_enc = self.tokenizer(
            item['chosen'], 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            item['rejected'], 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            'chosen_input_ids': chosen_enc['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_enc['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(0)
        }

# Example preference data
preference_data = [
    {"chosen": "The quick brown fox jumps over the lazy dog", "rejected": "The quick brown fox jumps over dog"},
    {"chosen": "What is a dog? A lazy brown fox", "rejected": "What is a dog? A fox"}
]

# Initialize the RewardModel wrapper
# This instantiates the Qwen model and tokenizer internally
reward_model_wrapper = RewardModel(model_id=MODEL_ID)
tokenizer = reward_model_wrapper.tokenizer

# Create dataset and dataloader
preference_dataset = PreferenceDataset(preference_data, tokenizer)
preference_dataloader = DataLoader(preference_dataset, batch_size=2, shuffle=True)

def compute_rm_loss(reward_model_wrapper, batch):
    """
    Computes the loss for a Reward Model on a batch of preference pairs.
    """
    # Decoding the batch for the wrapper's interface
    # Note: efficient implementations would pass tensors directly, but 
    # we are adapting to the wrapper's text-based interface here.
    # Ideally, refactor RewardModel to accept tensors directly for training efficiency.
    
    chosen_ids = batch['chosen_input_ids']
    rejected_ids = batch['rejected_input_ids']
    
    # We need to decode back to text because the provided RewardModel.get_reward() 
    # takes raw text inputs.
    # To optimize this, you should modify RewardModel to accept input_ids/attention_mask.
    chosen_texts = tokenizer.batch_decode(chosen_ids, skip_special_tokens=True)
    rejected_texts = tokenizer.batch_decode(rejected_ids, skip_special_tokens=True)
    
    # Get the scores for the chosen and rejected sequences
    chosen_rewards = reward_model_wrapper.get_reward(chosen_texts)
    rejected_rewards = reward_model_wrapper.get_reward(rejected_texts)
    
    # The core of the loss function: -log(sigmoid(chosen - rejected))
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    
    return loss

# Optimizer setup
# Access the underlying torch.nn.Module via .model
optimizer = torch.optim.Adam(reward_model_wrapper.model.parameters(), lr=1e-5)

# Training Loop
reward_model_wrapper.model.train() # Set to train mode

for i, batch in enumerate(preference_dataloader):
    optimizer.zero_grad()
    
    loss = compute_rm_loss(reward_model_wrapper, batch)
    
    loss.backward()
    optimizer.step()
    
    print(f"Batch {i}, Loss: {loss.item()}")
