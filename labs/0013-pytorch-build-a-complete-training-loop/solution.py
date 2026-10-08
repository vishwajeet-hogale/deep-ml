import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    N = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]

        train_loss = total_loss / N

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_accuracy = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

    return history