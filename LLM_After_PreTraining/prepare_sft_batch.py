import torch

def prepare_sft_batch(prompt: str, response: str, tokenizer):
    """
    This is the core engineering of SFT. It takes a prompt/response pair
    and creates the input_ids and the strategically masked labels.
    """
    # 1. Format the text with special tokens for conversation structure.
    prompt_part = f"<|user|> {prompt} <|end|> <|assistant|>"
    full_text = f"{prompt_part} {response} <|end|>"

    # 2. Tokenize to find the boundary for loss masking.
    # We only want to train the model on the assistant's response.
    prompt_ids = tokenizer.encode(prompt_part)
    mask_until_idx = len(prompt_ids)

    # 3. Tokenize the full conversation for model input.
    input_ids = tokenizer.encode(full_text)

    # 4. Create the labels tensor by cloning the input_ids.
    labels = torch.tensor(input_ids).clone()

    # 5. Apply the mask. This is the critical step.
    # We replace the prompt tokens in the labels with -100.
    labels[:mask_until_idx] = -100

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": labels
    }
