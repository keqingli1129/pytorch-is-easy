from transformers import AutoTokenizer

def check_tokenizer(model_name):
    print(f"Checking {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        return

    example = {
        "instruction": "Explain quantum mechanics.",
        "input": "",
        "output": "It is effectively magic."
    }

    # User snippet logic
    instruction_text = "\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: "
    response_text = example["output"] + tokenizer.eos_token
    
    print(f"Instruction text: {repr(instruction_text)}")
    print(f"Response text: {repr(response_text)}")

    instruction = tokenizer(instruction_text)
    response = tokenizer(response_text)

    print(f"Instruction IDs: {instruction['input_ids']}")
    print(f"Response IDs: {response['input_ids']}")
    
    combined = instruction["input_ids"] + response["input_ids"]
    print(f"Combined IDs: {combined}")
    
    decoded = tokenizer.decode(combined)
    print(f"Decoded: {repr(decoded)}")

    # Check for double BOS or similar issues
    # Note: different tokenizers treat special tokens differently.
    # We want to see if `response` starts with a special token that shouldn't be there (like another BOS).
    
    if tokenizer.bos_token_id is not None:
        print(f"BOS ID: {tokenizer.bos_token_id}")
        if response['input_ids'][0] == tokenizer.bos_token_id:
            print("WARNING: Response starts with BOS token!")
    
    if tokenizer.eos_token_id is not None:
        print(f"EOS ID: {tokenizer.eos_token_id}")

if __name__ == "__main__":
    check_tokenizer("Langboat/bloom-1b4-zh")
    # check_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
