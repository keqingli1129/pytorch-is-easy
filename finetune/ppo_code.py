import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self, actor, critic, ref_model, reward_model, tokenizer, lr=1e-5):
        self.actor = actor              # The LLM to train
        self.critic = critic            # Value function (Scalar output)
        self.ref_model = ref_model      # Frozen copy for KL penalty
        self.reward_model = reward_model # Judge

        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # PPO Hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lam = 0.95        # GAE lambda
        self.kl_coef = 0.1     # Beta for KL penalty
        self.ppo_epochs = 4    # Update iterations per batch

    def compute_gae(self, rewards, values, next_value, masks):
        """
        Generalized Advantage Estimation (GAE)
        Calculates A_t for the PPO objective.
        """
        advantages = []
        gae = 0
        
        # Iterate backwards
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            advantages.insert(0, gae)
            next_value = values[step]
            
        return torch.tensor(advantages)

    def ppo_loss(self, old_log_probs, new_log_probs, advantages, returns, new_values):
        """
        The Core PPO Equation:
        L = min( ratio*A, clip(ratio, 1-e, 1+e)*A )
        """
        # 1. Calculate Ratio r_t(theta)
        # exp(new - old) is equivalent to new / old
        ratio = (new_log_probs - old_log_probs).exp()

        # 2. The Surrogate Objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # 3. Policy Loss (Maximize objective -> Minimize negative)
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. Value Loss (MSE between predicted value and actual return)
        value_loss = F.mse_loss(new_values, returns)

        return policy_loss + 0.5 * value_loss

    def train_step(self, prompts):
        # -------------------------------------------------------
        # Phase 1: Rollout (Experience Collection)
        # -------------------------------------------------------
        with torch.no_grad():
            # Generate response
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            outputs = self.actor.generate(**inputs, max_new_tokens=50)
            
            # Separate prompt from response for masking
            prompt_len = inputs.input_ids.shape[1]
            response_ids = outputs[:, prompt_len:]

            # 1. Get Log Probs from OLD Policy (pi_old)
            # We run a forward pass to get the logits of the generated sequence
            actor_out = self.actor(outputs)
            logits = actor_out.logits[:, prompt_len-1:-1, :]
            old_log_probs = F.log_softmax(logits, dim=-1)
            # Gather log prob of the specific tokens that were generated
            old_log_probs = torch.gather(old_log_probs, 2, response_ids.unsqueeze(-1)).squeeze(-1)

            # 2. Get Values (Critic)
            old_values = self.critic(outputs).squeeze(-1)[:, prompt_len:]

            # 3. Calculate Reward (Reward Model + KL Penalty)
            # Raw reward from the judge
            raw_rewards = self.reward_model(outputs) 
            
            # KL Penalty: log(pi_new) - log(pi_ref)
            ref_logits = self.ref_model(outputs).logits[:, prompt_len-1:-1, :]
            ref_log_probs = torch.gather(F.log_softmax(ref_logits, dim=-1), 2, response_ids.unsqueeze(-1)).squeeze(-1)
            kl_div = old_log_probs - ref_log_probs
            
            # Final Reward = Score - Beta * KL
            rewards = raw_rewards - self.kl_coef * kl_div

        # 4. Compute Advantages (GAE)
        # (Simplified: assuming batch is one long sequence for demonstration)
        advantages = self.compute_gae(rewards, old_values, 0, torch.ones_like(rewards))
        returns = advantages + old_values # Target for value function

        # -------------------------------------------------------
        # Phase 2: Optimization (PPO Update)
        # -------------------------------------------------------
        for _ in range(self.ppo_epochs):
            # Forward pass with gradients enabled
            # We need new log_probs because the model weights change every update
            current_out = self.actor(outputs)
            current_logits = current_out.logits[:, prompt_len-1:-1, :]
            
            new_log_probs = F.log_softmax(current_logits, dim=-1)
            new_log_probs = torch.gather(new_log_probs, 2, response_ids.unsqueeze(-1)).squeeze(-1)
            
            new_values = self.critic(outputs).squeeze(-1)[:, prompt_len:]

            # Calculate Loss
            loss = self.ppo_loss(old_log_probs.detach(), new_log_probs, advantages, returns, new_values)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

# Example Usage
# trainer = PPOTrainer(actor_model, critic_model, ref_model, reward_model, tokenizer)
# loss = trainer.train_step(["The capital of France is"])
