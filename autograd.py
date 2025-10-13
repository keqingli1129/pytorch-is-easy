import torch

def main():
    # x_date = torch.tensor([[[0.5, -0.5]]])
    # w = torch.tensor([[1.0], [2.0]], requires_grad=True)
    # print("x_date shape:", x_date.shape)
    # print("w shape:", w.shape)
    # y = x_date @ w
    # print("y:", y)
    # print("y shape:", y.shape)
    # y.backward(torch.ones_like(y))
    # print("w.grad:", w.grad)
    # a = torch.tensor(2.0, requires_grad=True)
    # b = torch.tensor(3.0, requires_grad=True)
    # x = torch.tensor(4.0, requires_grad=True)
    # z = x*y
    # y = a + b
    a = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, requires_grad=True )
    b = torch.tensor([[1,2], [3,4], [5,6]], dtype=torch.float32, requires_grad=True)
    y = a @ b
    print("y:", y)

if __name__ == "__main__":
    main()