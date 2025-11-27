import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import train_loader, val_loader, dimension
from model import MotionMLP
import os

os.makedirs("checkpoints", exist_ok=True)

model = MotionMLP(input_dim=dimension).cuda()
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_one_epoch(model, loader, optimizer, device="cuda"):
    model.train()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)  # (B, T, D)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)  # (B, T, D)

        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        count += x.size(0)

    return total_loss / count

@torch.no_grad()
def eval_epoch(model, loader, device="cuda"):
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        total_loss += loss.item() * x.size(0)
        count += x.size(0)
    return total_loss / count

def train():
    num_epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
    torch.save(model.state_dict(), "checkpoints/mlp_autoreg.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, "checkpoints/mlp_autoreg_full.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()