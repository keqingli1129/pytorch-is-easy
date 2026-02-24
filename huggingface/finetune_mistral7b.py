import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ============================================
# 1. 4-bit Quantization Config
# ============================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",                  # Normal Float 4-bit (better for weights)
    bnb_4bit_compute_dtype=torch.bfloat16,      # Compute in bf16 (A100 optimized)
    bnb_4bit_use_double_quant=True,             # Nested quantization for memory savings
)

# ============================================
# 2. Load Model and Tokenizer
# ============================================
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                          # Auto-distribute layers across GPU
    trust_remote_code=True,
)
model.config.use_cache = False                  # Disable for training

# ============================================
# 3. Prepare Model for Training
# ============================================
model = prepare_model_for_kbit_training(model)

# ============================================
# 4. LoRA Configuration
# ============================================
lora_config = LoraConfig(
    r=16,                                       # LoRA rank (8-64 typical)
    lora_alpha=32,                              # Scaling factor (usually 2*r)
    target_modules=[                            # Mistral attention layers
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,                          # Regularization
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()              # Show ~0.1% trainable

# ============================================
# 5. Load and Format Dataset
# ============================================
# Example: Using a chat-formatted dataset
dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train[:5000]")

def format_chat_prompt(example):
    """Format to Mistral Instruct template"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]}
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
    }

dataset = dataset.map(format_chat_prompt)

# ============================================
# 6. Training Arguments
# ============================================
training_args = TrainingArguments(
    output_dir="./mistral-7b-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,              # Adjust based on sequence length
    gradient_accumulation_steps=4,              # Effective batch = 16
    optim="paged_adamw_8bit",                   # 8-bit optimizer for memory
    learning_rate=2e-4,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    group_by_length=True,                       # More efficient packing
    lr_scheduler_type="cosine",
    logging_steps=25,
    save_strategy="epoch",
    bf16=True,                                  # A100 supports bf16 natively
    tf32=True,                                  # TensorFloat-32 on A100
    push_to_hub=False,
)

# ============================================
# 7. Initialize Trainer and Train
# ============================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=2048,                        # Adjust: 2048-4096 on A100 40GB
    packing=True,                               # Efficient sequence packing
    dataset_text_field="text",
)

trainer.train()

# ============================================
# 8. Save Model
# ============================================
trainer.model.save_pretrained("./mistral-7b-qlora-adapter")
tokenizer.save_pretrained("./mistral-7b-qlora-adapter")
