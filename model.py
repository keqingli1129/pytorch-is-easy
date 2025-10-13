import torch

def main():
    # torch.manual_seed(42)  # Set random seed for reproducibility
    # D = 10
    # D_in = 1
    # D_out = 1
    # X = torch.randn(D, D_in)
    # true_w = torch.tensor([[2.0]])
    # true_b = torch.tensor([1.0])
    # true_y = X @ true_w + true_b
    
    # print(f"Input Tensor X:\n{X} : {X.shape}")
    # w = torch.randn(D_in, D_out, requires_grad=True)
    # b = torch.randn(D_out, requires_grad=True )
    # print(f"True weights:\n{w} : {w.shape}")
    # print(f"True bias:\n{b} : {b.shape}")
    # y_hat = X @ w+ b
    # print(f"Output Tensor y_hat:\n{y_hat} : {y_hat.shape}")
    # print(f"Output Tensor y_true:\n{true_y} : {true_y.shape}")

    # error = y_hat - true_y
    # loss = (error**2).mean()
    # print(f"Loss: {loss.item()}")
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")

if __name__ == "__main__":
    main()