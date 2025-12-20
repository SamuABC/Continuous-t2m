import torch
import os
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # Required for plotting

from model import MotionQwen
from dataset import HumanML3DDataset, collate_fn

# --- Configuration ---
BATCH_SIZE = 16  # Adjust based on GPU VRAM
LR = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./HumanML3D"
SPLIT_FILE_TRAIN = "./HumanML3D/train.txt"
SPLIT_FILE_VAL = "./HumanML3D/val.txt"
MODEL_ID = "Qwen/Qwen1.5-0.5B"
MOTION_DIM = 263  # HumanML3D standard dimension
TF_START = 1.0  # Start with full teacher forcing
TF_END = 0.7  # End with 30% model predictions (don't go too low for Transformers)

# --- Setup ---
# 1. Initialize Model
model = MotionQwen(base_model_id=MODEL_ID, motion_dim=MOTION_DIM).to(DEVICE)

# 2. Tokenizer Setup
# The tokenizer is inside the model, but we need it for the dataset
tokenizer = model.tokenizer
# Ensure padding token is set (Qwen sometimes lacks a default pad token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Datasets & Loaders
# Training Data
train_dataset = HumanML3DDataset(DATA_ROOT, SPLIT_FILE_TRAIN, tokenizer)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# Validation Data
val_dataset = HumanML3DDataset(DATA_ROOT, SPLIT_FILE_VAL, tokenizer)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# 4. Optimization
optimizer = optim.AdamW(model.parameters(), lr=LR)

# 5. Checkpoints directory
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# Lists to store loss history for plotting
train_losses = []
val_losses = []

# --- Training Loop ---
print(f"Starting training on {DEVICE}...")
print(
    f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}"
)

for epoch in range(EPOCHS):
    # Calculate Teacher Forcing Ratio for this epoch (Linear Decay)
    # Formula: ratio = start - (progress * (start - end))
    tf_ratio = TF_START - (epoch / max(1, EPOCHS - 1)) * (TF_START - TF_END)
    tf_ratio = max(TF_END, tf_ratio)  # Clamp

    print(f"Epoch {epoch+1} | Teacher Forcing Ratio: {tf_ratio:.2f}")

    # ==========================
    #       Training Phase
    # ==========================
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        motion = batch["motion"].to(DEVICE)
        motion_mask = batch["motion_mask"].to(DEVICE)

        optimizer.zero_grad()

        # Pass the calculated ratio
        loss, predictions = model(
            input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio
        )

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix({"train_loss": loss.item()})

    # Calculate average training loss for this epoch
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # ==========================
    #      Validation Phase
    # ==========================
    model.eval()  # Set model to evaluation mode (disables Dropout)
    total_val_loss = 0

    # No gradient calculation needed for validation (saves memory)
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            motion = batch["motion"].to(DEVICE)
            motion_mask = batch["motion_mask"].to(DEVICE)

            # Forward pass
            loss, _ = model(input_ids, motion, motion_mask)
            total_val_loss += loss.item()

    # Calculate average validation loss for this epoch
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    # Print Summary
    print(
        f"Epoch {epoch+1} Completed. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    # Save Checkpoint
    torch.save(model.state_dict(), f"checkpoints/scheduled_sampling{epoch+1}.pt")

    # --- Plotting Results ---
    # Create a new figure for this epoch update
    plt.figure(figsize=(10, 6))

    # X-axis is based on current number of epochs completed
    epochs_range = range(1, len(train_losses) + 1)

    plt.plot(epochs_range, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="x")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save plot to file
    LOSSPLOT_PATH = "checkpoints/loss_plot.png"
    plt.savefig(LOSSPLOT_PATH)
    print(f"Loss plot saved to '{LOSSPLOT_PATH}'.")
    plt.close()

print("Training complete.")
