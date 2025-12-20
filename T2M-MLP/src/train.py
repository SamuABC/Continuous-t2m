import torch
import os
import torch.optim as optim
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import MotionQwen
from dataset import HumanML3DDataset, collate_fn

# --- Configuration ---
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./HumanML3D"
SPLIT_FILE_TRAIN = "./HumanML3D/train.txt"
SPLIT_FILE_VAL = "./HumanML3D/val.txt"
MODEL_ID = "Qwen/Qwen1.5-0.5B"
MOTION_DIM = 263
TF_START = 1.0
TF_END = 0.7
CHECKPOINT_DIR = "checkpoints"

# --- Setup ---
model = MotionQwen(base_model_id=MODEL_ID, motion_dim=MOTION_DIM).to(DEVICE)
tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = HumanML3DDataset(DATA_ROOT, SPLIT_FILE_TRAIN, tokenizer)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

val_dataset = HumanML3DDataset(DATA_ROOT, SPLIT_FILE_VAL, tokenizer)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

optimizer = optim.AdamW(model.parameters(), lr=LR)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# --- Tracking Structures ---
# Store epoch averages
train_losses_epoch = []
val_losses_epoch = []
# Store every single step (batch) loss for fine-grained analysis
train_losses_step = []


def save_history(epoch, train_ep, val_ep, train_step):
    """Saves raw data to JSON for later plotting."""
    data = {
        "epoch": epoch,
        "train_loss_epoch": train_ep,
        "val_loss_epoch": val_ep,
        "train_loss_step": train_step,
    }
    with open(os.path.join(CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(data, f)


def plot_losses(train_step, train_ep, val_ep):
    """Generates a detailed dual-plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Detailed Step Loss
    ax1.plot(train_step, label="Step Loss", alpha=0.3, color="gray")
    # Add a moving average for readability if we have enough steps
    if len(train_step) > 50:
        window = 50
        # Simple moving average
        avg = np.convolve(train_step, np.ones(window) / window, mode="valid")
        ax1.plot(
            np.arange(window - 1, len(train_step)),
            avg,
            label="Moving Avg (50 steps)",
            color="blue",
        )

    ax1.set_title("Training Loss per Step")
    ax1.set_xlabel("Steps (Batches)")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Epoch Averages (Train vs Val)
    epochs_range = range(1, len(train_ep) + 1)
    ax2.plot(epochs_range, train_ep, label="Train Avg", marker="o", color="blue")
    ax2.plot(epochs_range, val_ep, label="Val Avg", marker="x", color="red")

    ax2.set_title("Average Loss per Epoch")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_plot_detailed.png"))
    plt.close()


# --- Training Loop ---
print(f"Starting training on {DEVICE}...")

for epoch in range(EPOCHS):
    tf_ratio = TF_START - (epoch / max(1, EPOCHS - 1)) * (TF_START - TF_END)
    tf_ratio = max(TF_END, tf_ratio)

    print(f"Epoch {epoch+1} | Teacher Forcing: {tf_ratio:.2f}")

    # --- Training ---
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        motion = batch["motion"].to(DEVICE)
        motion_mask = batch["motion_mask"].to(DEVICE)

        optimizer.zero_grad()
        loss, predictions = model(
            input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio
        )
        loss.backward()
        optimizer.step()

        # Log step loss
        current_loss = loss.item()
        train_losses_step.append(current_loss)
        total_train_loss += current_loss

        progress_bar.set_postfix({"loss": current_loss})

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses_epoch.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            motion = batch["motion"].to(DEVICE)
            motion_mask = batch["motion_mask"].to(DEVICE)
            loss, _ = model(input_ids, motion, motion_mask)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses_epoch.append(avg_val_loss)

    print(f"Epoch Done. Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # --- Save & Plot ---
    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, f"scheduled_sampling_ckpt_ep{epoch+1}.pt"),
    )

    # Save raw data first (safety)
    save_history(epoch + 1, train_losses_epoch, val_losses_epoch, train_losses_step)

    # Update plots
    plot_losses(train_losses_step, train_losses_epoch, val_losses_epoch)

print("Training complete.")
