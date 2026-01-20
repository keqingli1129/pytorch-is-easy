import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# 1. Prepare the Dataset
# "Standard language modeling" format just needs a text column
data = [
    {"text": "The sky is blue because of Rayleigh scattering."},
    {"text": "Photosynthesis is the process by which plants make food."},
    {"text": "Deep learning uses neural networks with many layers."},
    # ... add thousands more examples for real training
]
dataset = Dataset.from_list(data)

# 2. Model & Tokenizer Configuration
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Or "Qwen/Qwen2.5-0.5B", "Llama-3.2-1B"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Small model for demonstration

# Quantization config for memory efficiency (4-bit loading)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Fix for models missing pad token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
# TaskType options include:
# CAUSAL_LM (Causal Language Modeling)
# SEQ_2_SEQ_LM (Sequence-to-Sequence Language Modeling)
# TOKEN_CLS (Token Classification)
# SEQ_CLS (Sequence Classification)
# QUESTION_ANS (Question Answering)
# FEATURE_EXTRACTION (Feature Extraction)
# IMAGE_CLASSIFICATION (Image Classification)
# SPEECH_RECOGNITION (Speech Recognition)
# MULTIPLE_CHOICE (Multiple Choice)
# ... (others may exist depending on library version)

# Identify target modules for LoRA
# "q_proj" (query projection)
# "k_proj" (key projection)
# "v_proj" (value projection)
# "o_proj" (output projection)
# "gate_proj"
# "up_proj"
# "down_proj"
# "fc1", "fc2" (for some architectures)
# "mlp"
# "Wq", "Wk", "Wv", "Wo" (for some models)
# "attention" (sometimes used as a catch-all)
# ... (others depending on the model, e.g., "dense", "intermediate", etc.)

# 3. LoRA Configuration (Parameter-Efficient Fine-Tuning)
peft_config = LoraConfig(
    r=16,          # Rank
    lora_alpha=32, # Alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"] # Target attention layers
)

# 4. SFT Configuration
sft_config = SFTConfig(
    output_dir="./sft_output",
    dataset_text_field="text",  # <--- Key arg: specifies the column with your text
    max_seq_length=512,
    packing=True,               # Highly recommended: Packs short examples into full context length
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    fp16=True,                  # Use fp16 or bf16 depending on GPU
)

# Optional: Custom formatting function (if needed)
# def formatting_func(example):
#     return f"Question: {example['question']}\nAnswer: {example['answer']}"

# trainer = SFTTrainer(..., formatting_func=formatting_func)
# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=sft_config,
)

# 6. Start Training
print("Starting training...")
trainer.train()

# 7. Save Model
trainer.save_model("./sft_final_model")
print("Training complete and model saved.")
