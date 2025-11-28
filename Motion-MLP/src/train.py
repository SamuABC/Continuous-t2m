import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

import config as cfg
from dataset import get_loaders
from model import MotionMLP


def save_loss_plot(train_losses, val_losses, save_path, best_epoch, best_val_loss):
    """
    Plots losses and marks the best epoch with a vertical line.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    # Mark the best epoch visually
    plt.axvline(
        x=best_epoch, color="g", linestyle="--", label=f"Best Model (Ep {best_epoch})"
    )
    plt.scatter(best_epoch, best_val_loss, color="green", zorder=5)

    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    # Add info to title
    plt.title(f"Training Loss (Best: {best_val_loss:.6f} at Ep {best_epoch})")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


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

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            )
            print(f"--> New best model saved (Val Loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{cfg.PATIENCE} epochs.")

            # Early stopping
            if epochs_no_improve >= cfg.PATIENCE:
                print("Early stopping triggered. Training stopped.")
                break

    print("Training complete. Best model saved to checkpoints/")
    save_loss_plot(
        train_losses,
        val_losses,
        os.path.join(cfg.CHECKPOINT_DIR, "lossplot.png"),
        best_epoch,
        best_val_loss,
    )


if __name__ == "__main__":
    train()
