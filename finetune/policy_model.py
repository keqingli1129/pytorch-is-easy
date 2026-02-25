import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead


class PolicyModel:
    """Wraps a causal LM with a value head for PPO-based RLHF training."""

    def __init__(self, model_id: str):
        self.model_id = model_id

        # 1. Load the Tokenizer
        # Qwen2.5 uses a specific BPE tokenizer. We set padding_side='left' for generation/inference.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Define the Base Policy Model (Actor)
        # Loading in bfloat16 is recommended for modern GPUs (Ampere+) to save memory/compute.
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 3. Wrap with Value Head (for PPO)
        # This adds a linear layer to estimate the value function V(s)
        # while keeping the base model as the policy pi(a|s).
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

        # Optional: Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

    def __repr__(self) -> str:
        return f"PolicyModel(model_id='{self.model_id}')"


if __name__ == "__main__":
    policy = PolicyModel("Qwen/Qwen2.5-0.5B-Instruct")
    print(policy)
    print(policy.model)
