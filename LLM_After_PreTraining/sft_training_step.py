import torch
from finetune.policy_model import PolicyModel
from LLM_After_PreTraining.dolly_sft_dataset import create_dolly_dataloader


def sft_training_step(policy_model, optimizer, batch):
    policy_model.model.train()
    optimizer.zero_grad()

    # Move tensors to the same device as the model
    device = next(policy_model.model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # The model's forward pass automatically calculates the masked loss
    # because PyTorch's cross_entropy ignores labels with value -100.
    outputs = policy_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss  # Assuming a Hugging Face-style model output

    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. Build the Dolly-15k dataloader
    dataloader, tokenizer, dataset = create_dolly_dataloader(
        model_id=model_id,
        max_length=512,
        batch_size=4,
    )

    # 2. Initialise the policy model and optimizer
    policy_model = PolicyModel(model_id)
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=5e-5)

    # 3. Training loop over real data
    num_epochs = 1
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            loss = sft_training_step(policy_model, optimizer, batch)
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {step}/{len(dataloader)} | Loss: {loss:.4f}")
        print(f"Epoch {epoch+1} complete.")

    # 4. Save the SFT-trained model so it can be loaded for RLHF
    sft_output_dir = "./qwen2.5-0.5B-SFT"
    policy_model.model.save_pretrained(sft_output_dir)
    tokenizer.save_pretrained(sft_output_dir)
    print(f"SFT model saved to {sft_output_dir}")


if __name__ == "__main__":
    main()