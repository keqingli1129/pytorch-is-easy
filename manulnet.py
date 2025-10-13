import torch

def main():
    N = 10
    D_in = 1
    D_out = 1
    x = torch.randn(N, D_in)
    # print(x.shape)
    # W = torch.randn(D_in, D_out)
    # print(W.shape)
    # w_true = torch.tensor([[2.0]])
    # print(w_true.shape)
    # b = torch.tensor(1.0)
    # y_true = x @ w_true + b + 0.1*torch.randn(N, D_out)
    # print(y_true.shape)
    W = torch.randn(D_in, D_out, requires_grad=True)
    b = torch.randn(D_out, requires_grad=True)
    y = x @ W + b
    y_true = 2 * x + 1 + 0.1 * torch.randn(N, D_out)
    loss = (y - y_true).pow(2).sum()
    print(loss)
if __name__ == "__main__":
    main()