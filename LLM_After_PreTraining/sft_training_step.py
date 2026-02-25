import torch
import finetune.policy_model as policy_model
# Assume 'policy_model' is our LLM and 'optimizer' is an AdamW optimizer.
# The model's forward pass is expected to return (logits, loss).
policy_model = policy_model.policy_model
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5)
def sft_training_step(policy_model, optimizer, batch):
    policy_model.train()
    optimizer.zero_grad()

    # The model's forward pass automatically calculates the masked loss
    # because PyTorch's cross_entropy ignores labels with value -100.
    outputs = policy_model(
        input_ids=batch["input_ids"],
        labels=batch["labels"]
    )
    loss = outputs.loss # Assuming a Hugging Face-style model output

    loss.backward()
    optimizer.step()
    return loss.item()