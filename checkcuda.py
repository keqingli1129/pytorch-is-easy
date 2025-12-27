import torch
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}") # Should be (5, 0)
if __name__ == "__main__":
    main()