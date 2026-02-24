import torch
import copy

# --- PHASE 1: ROLLOUT ---
# Snapshot the current actor to create the 'old' policy for clipping
pi_phi_old = copy.deepcopy(actor_policy).freeze()

# Generate a batch of experience
# This involves getting responses and log-probs from the models
static_dataset = []
for prompt in prompts:
    # 1. Generate response using the frozen snapshot
    # actions are token IDs, states are the context at each step
    actions, states = pi_phi_old.generate(prompt)

    # 2. Get necessary log-probabilities and values
    log_probs_old = pi_phi_old.get_log_probs(states, actions)
    log_probs_SFT = reference_policy.get_log_probs(states, actions)
    values = critic.get_values(states) # From the live critic

    # 3. Get the final reward score
    reward_score = reward_model.score(prompt, actions)

    # 4. Compute augmented rewards and advantages
    rewards_aug = calculate_augmented_rewards(reward_score, log_probs_old, log_probs_SFT)
    advantages = calculate_advantages(rewards_aug, values)
    
    # 5. Store everything in a static dataset
    static_dataset.append((states, actions, log_probs_old, advantages))

    # --- PHASE 2: LEARNING ---
for epoch in range(NUM_EPOCHS):
    for (states, actions, log_probs_old, advantages) in static_dataset:
        
        # --- Policy Loss Calculation ---
        # L_Policy = -min(ratio * A, clip(ratio, 1-e, 1+e) * A)
        
        # Get log-probs from the LIVE actor
        log_probs_new = actor_policy.get_log_probs(states, actions)
        
        # Calculate the ratio
        ratio = torch.exp(log_probs_new - log_probs_old)
        
        # Calculate the two terms of the min function
        unclipped_term = ratio * advantages
        clipped_term = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages
        
        # Final policy loss
        policy_loss = -torch.min(unclipped_term, clipped_term).mean()

        # --- Value Loss Calculation ---
        # L_Value = (V_psi(s) - R_target)^2
        
        # Get predictions from the LIVE critic
        predicted_values = critic.get_values(states)
        
        # Calculate the target reward (rewards-to-go)
        # This can be pre-calculated during the rollout
        target_rewards = advantages + values_from_rollout 
        
        value_loss = value_loss_fn(predicted_values, target_rewards)

        # --- Entropy Bonus Calculation ---
        # S = -sum(p * log(p))
        # We get the policy's probability distribution to calculate this
        entropy = actor_policy.get_entropy(states)
        
        # --- Combine and Update ---
        # L_total = L_Policy - c1*L_Value + c2*S
        total_loss = policy_loss - VALUE_COEFF * value_loss + ENTROPY_COEFF * entropy

        optimizer.zero_grad()
        total_loss.backward() # This calculates gradients for BOTH actor and critic
        optimizer.step()