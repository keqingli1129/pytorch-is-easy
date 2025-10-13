import torch
import torch.nn as nn


def main():
    # C_in = 2
    # C_out = 4

    # linear_layer = nn.Linear(C_in, C_out)
    # print(f"Weight shape: {linear_layer.weight.shape}")
    # print(f"Bias shape: {linear_layer.bias.shape}")
    # linear_layer.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    # linear_layer.bias.data = torch.tensor([1.0, 2.0, 3.0, 4.0])

    x = torch.tensor([[[0.5, -0.5]]])
    # print(f"Input shape: {x.shape}")
    # y = linear_layer(x)
    # print(f"Output shape: {y.shape}")    
    # print(f"Output tensor:\n{y}")

    fc = nn.Linear(2, 8)
    proj = nn.Linear(8, 2)
    torch.manual_seed(42)  # Set random seed for reproducibility
    fc.weight.data = torch.randn(8, 2)
    fc.bias.data = torch.randn(8)
    proj.weight.data = torch.randn(2, 8)
    proj.bias.data = torch.randn(2)
    x_expand = fc(x)
    print(x_expand)
    x_activated = torch.relu(x_expand)
    print(x_activated)
    x_projected = proj(x_activated)
    print(x_projected)
    drop = nn.Dropout(p=0.5)
    x_dropped = drop(x_projected)
    print(x_dropped)

if __name__ == "__main__":
    main()