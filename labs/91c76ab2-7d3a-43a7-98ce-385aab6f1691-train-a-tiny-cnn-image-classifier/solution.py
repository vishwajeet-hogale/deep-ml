import torch
import torch.nn as nn
import torch.optim as optim


class TinyCNN(nn.Module):
    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * (img_size // 2) * (img_size // 2), n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))  # (N, 8, H/2, W/2)
        x = x.flatten(1)                        # (N, 8*H/2*W/2)
        return self.fc(x)                       # (N, n_classes)


def build_model(img_size=8, n_classes=2):
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    N = train_x.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = train_x[idx], train_y[idx]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model