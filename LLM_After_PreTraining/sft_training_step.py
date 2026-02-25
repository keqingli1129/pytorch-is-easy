import torch
from finetune.policy_model import PolicyModel

def sft_training_step(policy_model, optimizer, batch):
    policy_model.model.train()
    optimizer.zero_grad()

    # The model's forward pass automatically calculates the masked loss
    # because PyTorch's cross_entropy ignores labels with value -100.
    outputs = policy_model.model(
        input_ids=batch["input_ids"],
        labels=batch["labels"]
    )
    loss = outputs.loss # Assuming a Hugging Face-style model output

    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    # Example batch data
    batch = {
        "input_ids": torch.randint(0, 50000, (2, 128)),
        "labels": torch.randint(0, 50000, (2, 128))
    }
    # Assume 'policy_model' is our LLM and 'optimizer' is an AdamW optimizer.
    # The model's forward pass is expected to return (logits, loss).
    policy_model = PolicyModel("Qwen/Qwen2.5-0.5B-Instruct")  # Initialize your model here
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=5e-5)
    # Training loop
    num_steps = 10
    for step in range(num_steps):
        loss = sft_training_step(policy_model, optimizer, batch)
        print(f"Step {step + 1}/{num_steps}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()