from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from peft import LoraConfig, get_peft_model

dataset = load_dataset("shawhin/youtube-titles-dpo")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name,
    dtype=torch.float16).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # set pad token

# def format_chat_prompt(user_input, system_message="You are a helpful assistant."):
#     """
#     Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

#     Args:
#         user_input (str): The input text from the user.

#     Returns:
#         str: Formatted prompt for the model.
#     """
    
#     # Format user message
#     user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    
#     # Start assistant's turn
#     assistant_prompt = "<|im_start|>assistant\n"
    
#     # Combine prompts
#     formatted_prompt = user_prompt + assistant_prompt
    
#     return formatted_prompt

# # Set up text generation pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device='cpu')

# # Example prompt
# prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])

# # Generate output
# outputs = generator(prompt, max_length=100, truncation=True, num_return_sequences=1, temperature=0.7)

# print(outputs[0]['generated_text'])


ft_model_name = model_name.split('/')[1].replace("Instruct", "DPO")

training_args = DPOConfig(
    output_dir="./qwen-dpo-legacy",
    
    # Batch size of 1 is mandatory for 4GB VRAM
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Simulate batch size 8
    
    # --- Precision ---
    fp16=True,             # ENABLE this. Maxwell handles FP16 well.
    bf16=False,            # DISABLE. Hardware crash if enabled.
    
    # --- Optimizer ---
    optim="adamw_torch",   # Native PyTorch optimizer (No bitsandbytes dependency)
    
    # --- Checkpointing ---
    gradient_checkpointing=True, # Critical for 0.5B model on 4GB
    
    # --- Context Length ---
    max_length=512,        # Strict limit. 1024 might OOM.
    max_prompt_length=256,
    
    # --- Speed/Logging ---
    dataloader_num_workers=0, # Windows/Old GPU safety
    logging_steps=10,
    save_strategy="epoch",    # Save less frequently
    report_to="none"
)

# device = torch.device('cuda' if# --- 2. LoRA Config (The "Lite" Version) ---
peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], # Minimal targets to save VRAM
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, peft_config)

trainer = DPOTrainer(
    model=model, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
)

trainer.train()