import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_id = "HuggingFaceH4/ultrachat_200k" 
batch_size = 2  # Adjust based on GPU VRAM (Qwen-0.5B is small, but context length matters)
grad_accum_steps = 4
learning_rate = 2e-5
num_epochs = 1
max_seq_length = 1024 # Truncate for memory efficiency

# --- 1. Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)

# --- 2. Load and Prep Dataset ---
# We load a small subset for demonstration. Remove 'split' slicing for full training.
dataset = load_dataset(dataset_id, split="train_sft[:2000]") 

def collate_fn(batch):
    # This replaces your manual sft_data_collator with one compatible with Qwen's Chat Templates
    # Batch is a list of dicts from the dataset. Ultrachat has a 'messages' column.
    
    formatted_texts = []
    for example in batch:
        # Check dataset structure; Ultrachat uses 'messages' list of dicts
        msgs = example["messages"] 
        # Apply Qwen's chat template: <|im_start|>user...<|im_end|>...
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted_texts.append(text)
    
    # Tokenize with padding and truncation
    encodings = tokenizer(
        formatted_texts, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length, 
        return_tensors="pt"
    )
    
    # Create labels: same as input_ids, but mask padding tokens
    labels = encodings.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Optional: Mask user prompts here if you want to train only on assistant responses (recommended)
    # For brevity, this standard version trains on the whole sequence (standard CLM)
    
    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "labels": labels
    }

train_dataloader = DataLoader(
    dataset, 
    shuffle=True, 
    batch_size=batch_size, 
    collate_fn=collate_fn
)

# --- 3. Optimizer & Scheduler ---
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)

# --- 4. Training Loop ---
model.train()
print(f"Starting training on {len(dataset)} examples...")

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # Move batch to GPU
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass with gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
        if step % 50 == 0 and (step + 1) % grad_accum_steps == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item() * grad_accum_steps:.4f}")

# --- 5. Save Model ---
model.save_pretrained("./qwen-finetuned")
tokenizer.save_pretrained("./qwen-finetuned")
print("Training complete. Model saved.")
