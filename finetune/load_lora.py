import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Define paths
base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./qwen-dpo-legacy/checkpoint-387"  # Path to your saved LoRA

# 2. Load Base Model (FP16 for GTX 950M)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="cuda:0"
)

# 3. Load & Attach LoRA Adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    dtype=torch.float16
)

# 4. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 5. Run Inference
inputs = tokenizer("User: Hello!\nAssistant:", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

merged_model = model.merge_and_unload()

# Save the new standalone model
merged_model.save_pretrained("./qwen-0.5b-merged")
tokenizer.save_pretrained("./qwen-0.5b-merged")
