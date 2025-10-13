import torch.nn as nn
import torch

class LinearRegModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
    

if __name__ == "__main__":

    model = LinearRegModel(in_features = 1, out_features = 1)
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    epochs = 1000
    X = torch.randn(10, 1)

    for epoch in range(epochs):
        y_pred = model(X)
        y_true = 2 * X + 1 + 0.1 * torch.randn(10, 1)
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")