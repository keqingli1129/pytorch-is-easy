from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def format_chat_prompt(user_input, system_message="You are a helpful assistant."):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user.

    Returns:
        str: Formatted prompt for the model.
    """
    
    # Format user message
    user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    
    # Start assistant's turn
    assistant_prompt = "<|im_start|>assistant\n"
    
    # Combine prompts
    formatted_prompt = user_prompt + assistant_prompt
    
    return formatted_prompt


def main():
    dataset = load_dataset("shawhin/youtube-titles-dpo")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # set pad token

    # Set up text generation pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    # Example prompt (before fine-tuning)
    prompt = format_chat_prompt(dataset["valid"]["prompt"][0][0]["content"])
    outputs = generator(
        prompt,
        max_length=100,
        truncation=True,
        num_return_sequences=1,
        temperature=0.7,
    )
    print(outputs[0]["generated_text"])

    ft_model_name = model_name.split("/")[1].replace("Instruct", "DPO")

    training_args = DPOConfig(
        output_dir=ft_model_name,
        logging_steps=25,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        eval_strategy="epoch",
        eval_steps=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
    )
    trainer.train()

    # Load the fine-tuned model
    ft_model = trainer.model

    # Set up text generation pipeline (after fine-tuning)
    generator = pipeline("text-generation", model=ft_model, tokenizer=tokenizer, device=device)
    prompt = format_chat_prompt(dataset["valid"]["prompt"][0][0]["content"])
    outputs = generator(
        prompt,
        max_length=100,
        truncation=True,
        num_return_sequences=1,
        temperature=0.7,
    )
    print(outputs[0]["generated_text"])

    model_id = f"keqingli1129/{ft_model_name}"
    trainer.push_to_hub(model_id)


if __name__ == "__main__":
    main()

