import torch
from reward_model import RewardModel
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from preference_dataset import PreferenceDataset

# Pre-processing function to apply Chat Template
# Qwen2.5 expects specific special tokens (<|im_start|>, etc.)
def format_chat(example, tokenizer):
    # apply_chat_template converts list of msgs -> string with special tokens
    # We do NOT tokenize yet (tokenize=False) because PreferenceDataset handles that
    chosen_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
    rejected_text = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
    
    return {
        "chosen": chosen_text, 
        "rejected": rejected_text
    }

def compute_rm_loss(reward_model_wrapper, batch, tokenizer):
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

def main():
        # This instantiates the Qwen model and tokenizer internally
        # Use the same model ID as in your reward_model.py
        MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
        reward_model_wrapper = RewardModel(model_id=MODEL_ID)
        tokenizer = reward_model_wrapper.tokenizer

        # 1. Load the dataset from Hugging Face
        # We use the 'train_prefs' split which is standard in this dataset
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        # Save the dataset locally if not already saved
        local_dataset_path = "./ultrafeedback_binarized_train_prefs"
        try:
            # Try to load from disk
            dataset = Dataset.load_from_disk(local_dataset_path)
            print("Loaded dataset from local disk.")
        except Exception:
            # If not found, download and save
            dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
            dataset.save_to_disk(local_dataset_path)
            print("Downloaded and saved dataset locally.")
        # 2. Apply formatting to the entire dataset
        # limit to 1000 samples for quick testing if needed
        processed_dataset = dataset.select(range(1000)).map(lambda x: format_chat(x, tokenizer))
        preference_dataset = PreferenceDataset(processed_dataset, tokenizer)
        preference_dataloader = DataLoader(preference_dataset, batch_size=2, shuffle=True)
        # ... Proceed with your existing training loop ...
        print("Dataset loaded. First example chosen text:")
        print(processed_dataset[0]['chosen'][:200]) # Preview

        # Optimizer setup
        # Access the underlying torch.nn.Module via .model
        optimizer = torch.optim.Adam(reward_model_wrapper.model.parameters(), lr=1e-5)
        reward_model_wrapper.model.train()
        for i, batch in enumerate(preference_dataloader):
            optimizer.zero_grad()
            loss = compute_rm_loss(reward_model_wrapper, batch)
            loss.backward()
            optimizer.step()
            print(f"Batch {i}, Loss: {loss.item()}")

if __name__ == "__main__":
        main()