import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from gpt2_min import GPT2
from gptconfig import GPTConfig
from generate_trainingset import generate_trainingset
from gutenberg_dataset import generate_gutenberg_dataset
from gutenbergGPT2Dataset import GutenbergGPT2Dataset
from transformers import GPT2Tokenizer
import tiktoken

def evaluate(model, val_loader):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            # x = x.to(device)
            # y = y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    # Model config (should match tokenizer and block_size)
    config = GPTConfig(
        vocab_size=tokenizer.n_vocab,
        block_size=128,
        batch_size = 32,
        learning_rate = 3e-4,
        epochs = 3,
        n_layer=4,      # You can adjust these for your experiment
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    hf_dataset = generate_gutenberg_dataset()
    # Create PyTorch datasets for each split
    train_dataset = GutenbergGPT2Dataset(hf_dataset["train"][0:100], tokenizer, config.block_size)
    val_dataset = GutenbergGPT2Dataset(hf_dataset["validation"][0:100], tokenizer, config.block_size)
    test_dataset = GutenbergGPT2Dataset(hf_dataset["test"][0:100], tokenizer, config.block_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, config.batch_size)
    test_loader = DataLoader(test_dataset, config.batch_size)
    # train_loader, val_loader, test_loader = generate_trainingset(tokenizer, config.block_size)
    # model = GPT2(config).to(device)
    model = GPT2(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.epochs):
        # pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        # for x, y in pbar:
        #     x = x.to(device)
        #     y = y.to(device)
        #     optimizer.zero_grad()
        #     logits, loss = model(x, y)
        #     loss.backward()
        #     optimizer.step()
        #     pbar.set_postfix(loss=loss.item())
        # val_loss = evaluate(model, val_loader)
        # print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
        total_loss = 0
        batch_count = 0
        for x, y in train_loader:
            # x = x.to(device)``
            # y = y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            print(f"Batch {batch_count}: Train loss = {loss.item():.4f}")
        avg_loss = total_loss / batch_count
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Train loss = {avg_loss:.4f}, Validation loss = {val_loss:.4f}")
    # Save model
    torch.save(model.state_dict(), "gpt2_mini_gutenberg.pt")

if __name__ == "__main__":
    train()