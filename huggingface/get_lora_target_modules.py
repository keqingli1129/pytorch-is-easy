import torch
from transformers import Conv1D, AutoModelForCausalLM

def get_lora_target_modules(model):
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding,
                               torch.nn.Conv2d, Conv1D)):
            layer_names.append(name.split(".")[-1])
    return list(set(layer_names))
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name)
print(get_lora_target_modules(model))
