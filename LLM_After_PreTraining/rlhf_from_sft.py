"""
RLHF training using a previously SFT-trained Qwen2.5-0.5B model.

Pipeline:  Pre-trained → SFT (sft_training_step.py) → RLHF (this file)

The SFT checkpoint serves two roles:
  1. **Policy model (actor)** – the model we continue to update with PPO.
  2. **Reference model (ref)** – a frozen copy used for the KL penalty
     so the policy doesn't drift too far from the SFT behaviour.
"""

import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead


# ---------------------------------------------------------------------------
# 1. Load the SFT-trained checkpoint (saved by sft_training_step.py)
# ---------------------------------------------------------------------------
SFT_MODEL_DIR = "./qwen2.5-0.5B-SFT"          # <-- output of SFT training
REWARD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # placeholder; swap with your trained RM

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Policy model (actor) – wrapped with a value head for PPO
base_model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
actor = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
actor.gradient_checkpointing_enable()

# Reference model – frozen copy of the same SFT checkpoint
ref_model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False


# ---------------------------------------------------------------------------
# 2. Minimal PPO helpers (same logic as finetune/ppo_code.py)
# ---------------------------------------------------------------------------
CLIP_EPS = 0.2
KL_COEF = 0.1
PPO_EPOCHS = 4


def compute_rewards(prompt_ids, response_ids, actor_logprobs, ref_logprobs,
                    reward_scores):
    """Augmented reward = R(x, y) - beta * KL(pi || pi_ref)."""
    kl = actor_logprobs - ref_logprobs            # per-token KL estimate
    rewards = reward_scores - KL_COEF * kl.sum(-1)
    return rewards


def ppo_loss(old_logprobs, new_logprobs, advantages, new_values, returns):
    ratio = (new_logprobs - old_logprobs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(new_values, returns)
    return policy_loss + 0.5 * value_loss


# ---------------------------------------------------------------------------
# 3. Training loop skeleton
# ---------------------------------------------------------------------------
def main():
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-5)

    # Replace this with your actual prompt dataset
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about autumn leaves.",
        "What are the benefits of exercise?",
    ]

    device = next(actor.parameters()).device

    for step, prompt in enumerate(prompts):
        # --- Rollout ---
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_ids = actor.generate(
                **inputs, max_new_tokens=128, do_sample=True, temperature=0.7,
            )

        prompt_len = inputs.input_ids.shape[1]
        response_ids = gen_ids[:, prompt_len:]

        # Old policy log-probs
        with torch.no_grad():
            actor_out = actor(gen_ids)
            logits = actor_out.logits[:, prompt_len - 1:-1, :]
            old_logprobs = torch.gather(
                F.log_softmax(logits, dim=-1), 2,
                response_ids.unsqueeze(-1),
            ).squeeze(-1)

            old_values = actor_out.value[:, prompt_len:].squeeze(-1)

            # Reference log-probs
            ref_out = ref_model(gen_ids)
            ref_logprobs = torch.gather(
                F.log_softmax(ref_out.logits[:, prompt_len - 1:-1, :], dim=-1), 2,
                response_ids.unsqueeze(-1),
            ).squeeze(-1)

        # Placeholder reward – replace with your reward model
        reward_scores = torch.zeros(gen_ids.shape[0], device=device)

        rewards = compute_rewards(
            inputs.input_ids, response_ids,
            old_logprobs, ref_logprobs, reward_scores,
        )
        advantages = rewards.unsqueeze(-1).expand_as(old_logprobs)  # simplified
        returns = advantages + old_values

        # --- PPO Update ---
        for _ in range(PPO_EPOCHS):
            cur_out = actor(gen_ids)
            cur_logits = cur_out.logits[:, prompt_len - 1:-1, :]
            new_logprobs = torch.gather(
                F.log_softmax(cur_logits, dim=-1), 2,
                response_ids.unsqueeze(-1),
            ).squeeze(-1)
            new_values = cur_out.value[:, prompt_len:].squeeze(-1)

            loss = ppo_loss(
                old_logprobs.detach(), new_logprobs,
                advantages.detach(), new_values, returns.detach(),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Step {step} | PPO loss: {loss.item():.4f}")

    # Save the RLHF-trained model
    rlhf_output = "./qwen2.5-0.5B-RLHF"
    actor.save_pretrained(rlhf_output)
    tokenizer.save_pretrained(rlhf_output)
    print(f"RLHF model saved to {rlhf_output}")


if __name__ == "__main__":
    main()
