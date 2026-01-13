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
from visualization.visualization import visualize_transformer_motion


def save_history(epoch, train_ep, val_ep, tf_ratios):
    """Saves raw data to JSON including Teacher Forcing history."""
    data = {
        "number_of_epochs": epoch,
        "train_loss_epoch": train_ep,
        "val_loss_epoch": val_ep,
        "tf_ratios": tf_ratios,
    }
    with open(os.path.join(cfg.CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(data, f)


def plot_losses(train_ep, val_ep, tf_ratios):
    """Generates a plot: Epoch Loss (Top) and Teacher Forcing (Bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    epochs_range = range(1, len(train_ep) + 1)

    # Subplot 1: Loss per Epoch
    ax1.plot(epochs_range, train_ep, label="Train Avg", marker="o", color="blue")
    ax1.plot(epochs_range, val_ep, label="Val Avg", marker="x", color="red")
    ax1.set_title("Average Loss per Epoch")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Teacher Forcing Schedule
    ax2.plot(
        epochs_range,
        tf_ratios,
        label="Teacher Forcing Ratio",
        marker=".",
        color="green",
    )
    ax2.set_title("Teacher Forcing Schedule")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "loss_plot.png"))
    plt.close()


def validate_visual(model, epoch, save_dir):
    """
    Generates and saves GIFs for fixed prompts to visually track progress.
    """
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_prompts = [
        "a person is walking forward",
        "a person is dancing specifically a waltz",
        "a person raises their right hand and waves",
        "a person walks slowly hunched over to the right.",
    ]

    print(f"--- Generating Visual Validation for Epoch {epoch} ---")

    mean = np.load(cfg.DATA_ROOT + "/Mean.npy")
    std = np.load(cfg.DATA_ROOT + "/Std.npy")

    for i, prompt in enumerate(test_prompts):
        try:
            with torch.no_grad():
                # generate full sequence (autoregressive)
                generated_motion = model.generate(prompt)

            # post-processing
            motion_data = generated_motion[0].cpu().numpy()
            motion_data = motion_data * std + mean  # denormalize

            output_path = os.path.join(save_dir, f"ep{epoch}_sample{i}.gif")

            visualize_transformer_motion(motion_data, prompt, output_path=output_path)

        except Exception as e:
            print(f"Failed to generate visualization for '{prompt}': {e}")

    model.train()


if __name__ == "__main__":
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN
    )

    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    # store plotting data
    train_losses_epoch = []
    val_losses_epoch = []
    tf_ratios = []

    # --- Training Loop ---
    print(f"Starting training on {cfg.DEVICE}...")
    for epoch in range(cfg.EPOCHS):
        if epoch < 0.2 * cfg.EPOCHS:
            # warm-up phase (0-20% epochs)
            tf_ratio = 1.0
        elif epoch < 0.8 * cfg.EPOCHS:
            # linear decay phase (20-80% epochs)
            progress = (epoch - 0.2 * cfg.EPOCHS) / (0.6 * cfg.EPOCHS)
            tf_ratio = 1.0 - progress
        else:
            # final phase without teacher forcing (80-100% epochs)
            tf_ratio = 0.0

        print(
            f"Epoch {epoch+1} | Teacher Forcing: {tf_ratio:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        tf_ratios.append(tf_ratio)

        # --- Training ---
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(cfg.DEVICE)
            motion = batch["motion"].to(cfg.DEVICE)
            motion_mask = batch["motion_mask"].to(cfg.DEVICE)

            optimizer.zero_grad()
            loss, _ = model(
                input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # gradient clipping
            optimizer.step()

            # log loss
            current_loss = loss.item()
            total_train_loss += current_loss
            progress_bar.set_postfix({"loss": current_loss})

        scheduler.step()

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

        # --- Visual Validation ---
        if (epoch + 1) % 10 == 0 or epoch == cfg.EPOCHS - 1:
            validate_visual(model, epoch + 1, cfg.CHECKPOINT_DIR + "/visualizations")

        print(f"Epoch Done. Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # --- Save & Plot ---
        trainable_state_dict = {
            k: v for k, v in model.named_parameters() if v.requires_grad
        }
        torch.save(
            trainable_state_dict,
            os.path.join(cfg.CHECKPOINT_DIR, f"trained_params_ep{epoch+1}.pt"),
        )
        save_history(epoch + 1, train_losses_epoch, val_losses_epoch, tf_ratios)
        plot_losses(train_losses_epoch, val_losses_epoch, tf_ratios)

    print("Training complete.")
