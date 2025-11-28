import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import get_loaders
from model import MotionMLP
import os
import config as cfg


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
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader, _, _ = get_loaders(batch_size=cfg.BATCH_SIZE)
    model = MotionMLP().cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Starting training...")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )
    torch.save(
        model.state_dict(), f"checkpoints/motionmlp_ep{epoch:03d}_val{val_loss:.4f}.pth"
    )
    print("Training complete. Model saved to checkpoints/")


if __name__ == "__main__":
    train()
