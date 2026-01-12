import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

class RewardModel:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Qwen and Llama models often lack a default pad token. 
        # We set it to eos_token to avoid errors during padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)

    def get_hidden_state(self, texts):
        # Tokenize input texts
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get last hidden state of the last token for each sequence
        last_hidden = outputs.hidden_states[-1]
        
        # Calculate indices of the last token in each sequence
        last_token_indices = (inputs['attention_mask'].sum(dim=1) - 1)
        batch_indices = torch.arange(last_hidden.size(0))
        
        last_token_hidden = last_hidden[batch_indices, last_token_indices]
        return last_token_hidden

    def get_reward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # The value head output is in outputs.value
        # Removing the last dimension to get a 1D tensor of rewards
        reward = outputs.value.squeeze(-1)
        
        # If we are using the last token's value as the sequence reward:
        # (This logic ensures we grab the value corresponding to the end of the sentence)
        last_token_indices = (inputs['attention_mask'].sum(dim=1) - 1)
        batch_indices = torch.arange(reward.size(0))
        sequence_reward = reward[batch_indices, last_token_indices]
        
        return sequence_reward
