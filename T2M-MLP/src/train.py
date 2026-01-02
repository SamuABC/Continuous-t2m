import json
import os

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from dataset import HumanML3DDataset
from model import MotionQwen
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_history(epoch, train_ep, val_ep, train_step):
    """Saves raw data to JSON for later plotting."""
    data = {
        "epoch": epoch,
        "train_loss_epoch": train_ep,
        "val_loss_epoch": val_ep,
        "train_loss_step": train_step,
    }
    with open(os.path.join(cfg.CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(data, f)


def plot_losses(train_step, train_ep, val_ep):
    """Generates a detailed dual-plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # subplot 1: step loss
    ax1.plot(train_step, label="Step Loss", alpha=0.3, color="gray")
    # add a moving average for readability if there are enough steps
    if len(train_step) > 50:
        window = 50
        # moving average
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

    # subplot 2: epoch averages (train vs. val)
    epochs_range = range(1, len(train_ep) + 1)
    ax2.plot(epochs_range, train_ep, label="Train Avg", marker="o", color="blue")
    ax2.plot(epochs_range, val_ep, label="Val Avg", marker="x", color="red")

    ax2.set_title("Average Loss per Epoch")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "loss_plot_detailed.png"))
    plt.close()


# --- Setup ---
model = MotionQwen(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM).to(
    cfg.DEVICE
)

train_dataset = HumanML3DDataset(
    cfg.DATA_ROOT, cfg.DATA_ROOT + "/train.txt", model.tokenizer
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
)

val_dataset = HumanML3DDataset(
    cfg.DATA_ROOT, cfg.DATA_ROOT + "/val.txt", model.tokenizer
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    collate_fn=val_dataset.collate_fn,
)

optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)

if not os.path.exists(cfg.CHECKPOINT_DIR):
    os.makedirs(cfg.CHECKPOINT_DIR)

# store epoch averages
train_losses_epoch = []
val_losses_epoch = []
# store every single step (batch) loss
train_losses_step = []


# --- Training Loop ---
print(f"Starting training on {cfg.DEVICE}...")

for epoch in range(cfg.EPOCHS):
    tf_ratio = cfg.TF_START - (epoch / max(1, cfg.EPOCHS - 1)) * (
        cfg.TF_START - cfg.TF_END
    )
    tf_ratio = max(cfg.TF_END, tf_ratio)

    print(f"Epoch {epoch+1} | Teacher Forcing: {tf_ratio:.2f}")

    # --- Training ---
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(cfg.DEVICE)
        motion = batch["motion"].to(cfg.DEVICE)
        motion_mask = batch["motion_mask"].to(cfg.DEVICE)

        optimizer.zero_grad()
        loss, _ = model(input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio)
        loss.backward()
        optimizer.step()

        # log step loss
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
            input_ids = batch["input_ids"].to(cfg.DEVICE)
            motion = batch["motion"].to(cfg.DEVICE)
            motion_mask = batch["motion_mask"].to(cfg.DEVICE)
            loss, _ = model(input_ids, motion, motion_mask)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses_epoch.append(avg_val_loss)

    print(f"Epoch Done. Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # --- Save & Plot ---
    torch.save(
        model.state_dict(),
        os.path.join(cfg.CHECKPOINT_DIR, f"scheduled_sampling_ckpt_ep{epoch+1}.pt"),
    )
    save_history(epoch + 1, train_losses_epoch, val_losses_epoch, train_losses_step)
    plot_losses(train_losses_step, train_losses_epoch, val_losses_epoch)

print("Training complete.")
