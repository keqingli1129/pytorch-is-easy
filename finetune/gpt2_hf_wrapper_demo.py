from gpt2_hf_wrapper import GPT2HF, GPT2ConfigHF
from trl import AutoModelForCausalLMWithValueHead

# 1. Initialize your wrapped model
config = GPT2ConfigHF(n_layer=4, n_head=4, n_embd=256) # Example small config
my_hf_model = GPT2HF(config)

# 2. Wrap it for Reward Modeling (adds the value head)
# We pass the model instance directly
model_with_value_head = AutoModelForCausalLMWithValueHead(my_hf_model)

# Now 'model_with_value_head' works with the RewardModel logic!