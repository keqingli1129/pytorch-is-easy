import tiktoken

# Load the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

text = "Hello, world!"
tokens = enc.encode(text)
print("Token IDs:", tokens)

decoded = enc.decode(tokens)
print("Decoded text:", decoded)