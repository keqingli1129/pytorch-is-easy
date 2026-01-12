import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. Load the Tokenizer
# Qwen2.5 uses a specific BPE tokenizer. We set padding_side='left' for generation/inference.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Qwen often requires setting a pad token explicitly

# 2. Define the Base Policy Model (Actor)
# Loading in bfloat16 is recommended for modern GPUs (Ampere+) to save memory/compute.
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 3. Wrap with Value Head (for PPO)
# This adds a linear layer to estimate the value function (V(s)) while keeping the base model as the policy (pi(a|s)).
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

# Optional: Enable gradient checkpointing for memory efficiency
policy_model.gradient_checkpointing_enable()

print(f"Policy model defined: {policy_model}")
