import transformers
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Hello, world!", return_tensors="pt")
# print(transformers.__version__)