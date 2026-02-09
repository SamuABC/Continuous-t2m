import os

import config as cfg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from motion_autoencoder.motion_autoencoder import MotionAutoEncoder, NTXentLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from dataset import HumanML3DDataset


def plot_ae_losses(recon_losses, aux_losses, save_dir, aux_label):
    epochs = range(1, len(recon_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Reconstruction Loss (Left Axis)
    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Reconstruction Loss (MSE)", color=color)
    ax1.plot(epochs, recon_losses, color=color, label="Reconstruction")
    ax1.tick_params(axis="y", labelcolor=color)

    # Plot Auxiliary Loss (Right Axis)
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel(aux_label, color=color)
    ax2.plot(epochs, aux_losses, color=color, linestyle="--", label="Contrastive")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Autoencoder Pretraining: {cfg.BASE_MODEL_ID}")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "ae_loss_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def train_contrastive_ae():
    """
    This training loop trains a motion autoencoder with a contrastive learning objective.
    Positive pairs are sampled from nearby frames in the motion sequence, and an NTXent
    loss encourages their latent representations to be close while pushing apart
    representations of randomly sampled negatives.
    """
    # 1. Setup Directories
    safe_model_name = cfg.BASE_MODEL_ID.replace("/", "_")
    checkpoint_dir = os.path.join(cfg.AUTOENCODER_CHECKPOINT_DIR, safe_model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = cfg.DEVICE

    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HumanML3DDataset(cfg.DATA_ROOT, cfg.DATA_ROOT + "/train.txt", tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.AE_BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=2,
    )

    # 2. Model & Loss
    config = AutoConfig.from_pretrained(cfg.BASE_MODEL_ID)
    model = MotionAutoEncoder(cfg.MOTION_DIM, config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn_recon = nn.MSELoss()
    loss_fn_cont = NTXentLoss(temperature=0.1)

    # Training Params
    hist_recon, hist_cont = [], []

    model.train()

    print("Starting Contrastive Training with Temporal Window...")

    for epoch in range(cfg.AE_TRAIN_EPOCHS):
        epoch_recon = 0.0
        epoch_cont = 0.0

        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{cfg.AE_TRAIN_EPOCHS}", unit="batch"
        )

        for batch in pbar:
            x = batch["motion"].to(device).float()  # (B, T, D)
            motion_mask = batch["motion_mask"].to(device)  # (B, T)

            B, T, D = x.shape

            # Get actual sequence lengths
            lengths = motion_mask.sum(dim=1).long()

            # --- Temporal Sampling Logic ---
            # 1. Select random Anchor Frame 't'
            # Range: [0, length-1]
            max_idx = (lengths - 1).clamp(min=0)
            t_anchor = (torch.rand(B, device=device) * max_idx).long()

            # 2. Select Offset Magnitude
            # Strictly range [1, window] to avoid identity mapping (0 offset)
            offset_mag = torch.randint(1, cfg.POSITIVE_WINDOW + 1, (B,), device=device)

            # 3. Select Random Direction (-1 or 1)
            sign = torch.randint(0, 2, (B,), device=device) * 2 - 1

            # 4. Calculate Candidate Position
            t_pos = t_anchor + (sign * offset_mag)

            # 5. Boundary Logic (Bounce back if out of bounds)
            # If t_pos < 0 (went too far back), flip to look forward (add magnitude)
            t_pos = torch.where(t_pos < 0, t_anchor + offset_mag, t_pos)

            # If t_pos >= length (went too far forward), flip to look backward (subtract magnitude)
            t_pos = torch.where(t_pos >= lengths, t_anchor - offset_mag, t_pos)

            # 6. Final Safety Clamp (handles edge case where seq_len < window_size)
            t_pos = t_pos.clamp(min=0).min(lengths - 1)

            # --- Extract Frames ---
            # Use batch indexing
            batch_indices = torch.arange(B, device=device)

            x_anchor = x[batch_indices, t_anchor]
            x_positive = x[batch_indices, t_pos]

            # --- Forward Pass ---
            recon, _, z_anchor_norm = model(x_anchor)
            _, _, z_pos_norm = model(
                x_positive
            )  # z is only needed for the positive sample

            # --- Loss Calculation ---
            loss_mse = loss_fn_recon(recon, x_anchor)
            loss_ntx = loss_fn_cont(z_anchor_norm, z_pos_norm)

            loss = loss_mse + (cfg.LAMBDA_CONTRASTIVE * loss_ntx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += loss_mse.item()
            epoch_cont += loss_ntx.item() * cfg.LAMBDA_CONTRASTIVE

            pbar.set_postfix(
                {"Recon": f"{loss_mse.item():.4f}", "Cont": f"{loss_ntx.item():.4f}"}
            )

        # Logging
        avg_recon = epoch_recon / len(dataloader)
        avg_cont = epoch_cont / len(dataloader)
        hist_recon.append(avg_recon)
        hist_cont.append(avg_cont)

    # Save
    model_save_path = os.path.join(checkpoint_dir, "motion_ae_contrastive.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_ae_losses(
        hist_recon, hist_cont, save_dir=checkpoint_dir, aux_label="Contrastive Loss"
    )

    return model


def train_smooth_ae():
    """
    This training loop trains a motion autoencoder with a smoothness constraint in the latent space.
    This is done by sampling positive pairs from nearby frames in the motion sequence
    and adding a cosine similarity loss to encourage their latent representations to be close.
    """
    # 1. Setup Directories
    safe_model_name = cfg.BASE_MODEL_ID.replace("/", "_")
    checkpoint_dir = os.path.join(cfg.AUTOENCODER_CHECKPOINT_DIR, safe_model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = cfg.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HumanML3DDataset(cfg.DATA_ROOT, cfg.DATA_ROOT + "/train.txt", tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.AE_BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=2,
    )

    # 2. Model & Loss
    config = AutoConfig.from_pretrained(cfg.BASE_MODEL_ID)
    model = MotionAutoEncoder(cfg.MOTION_DIM, config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn_recon = nn.MSELoss()

    # Training Params
    hist_recon, hist_smooth = [], []
    model.train()

    print("Starting Training with Denoising + Latent Smoothness...")

    for epoch in range(cfg.AE_TRAIN_EPOCHS):
        epoch_recon = 0.0
        epoch_smooth = 0.0
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{cfg.AE_TRAIN_EPOCHS}", unit="batch"
        )

        for batch in pbar:
            x = batch["motion"].to(device).float()  # (B, T, D)
            motion_mask = batch["motion_mask"].to(device)  # (B, T)

            B, T, D = x.shape

            lengths = motion_mask.sum(dim=1).long()

            # --- Temporal Sampling Logic ---
            # 1. Select random Anchor Frame 't'
            # Range: [0, length-1]
            max_idx = (lengths - 1).clamp(min=0)
            t_anchor = (torch.rand(B, device=device) * max_idx).long()

            # 2. Select Offset Magnitude
            # Strictly range [1, window] to avoid identity mapping (0 offset)
            offset_mag = torch.randint(1, cfg.POSITIVE_WINDOW + 1, (B,), device=device)

            # 3. Select Random Direction (-1 or 1)
            sign = torch.randint(0, 2, (B,), device=device) * 2 - 1

            # 4. Calculate Candidate Position
            t_pos = t_anchor + (sign * offset_mag)

            # 5. Boundary Logic (Bounce back if out of bounds)
            # If t_pos < 0 (went too far back), flip to look forward (add magnitude)
            t_pos = torch.where(t_pos < 0, t_anchor + offset_mag, t_pos)

            # If t_pos >= length (went too far forward), flip to look backward (subtract magnitude)
            t_pos = torch.where(t_pos >= lengths, t_anchor - offset_mag, t_pos)

            # 6. Final Safety Clamp (handles rare edge case where seq_len < window_size)
            t_pos = t_pos.clamp(min=0).min(lengths - 1)

            # --- Extract Frames ---
            batch_indices = torch.arange(B, device=device)
            x_anchor = x[batch_indices, t_anchor]
            x_positive = x[batch_indices, t_pos]

            # --- Denoising Step ---
            # Add noise to anchor input to prevent identity mapping
            noise = torch.randn_like(x_anchor) * cfg.AE_NOISE_LEVEL
            x_anchor_noisy = x_anchor + noise

            # --- Forward Pass ---
            # Reconstruct the noisy anchor, but get latent for both
            recon, z_anchor, _ = model(
                x_anchor_noisy
            )  # use raw z for cosine_similarity

            _, z_pos, _ = model(x_positive)  # Positive sample (clean)

            # --- Loss Calculation ---
            loss_mse = loss_fn_recon(recon, x_anchor)

            # Smoothness (Cosine Distance)
            cos_sim = F.cosine_similarity(z_anchor, z_pos, dim=1).mean()
            loss_smooth = 1.0 - cos_sim

            total_loss = loss_mse + (cfg.LAMBDA_SMOOTHNESS * loss_smooth)

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            epoch_recon += loss_mse.item()
            epoch_smooth += loss_smooth.item() * cfg.LAMBDA_SMOOTHNESS

            pbar.set_postfix(
                {
                    "Recon": f"{loss_mse.item():.4f}",
                    "Smooth": f"{loss_smooth.item():.4f}",
                }
            )

        # Logging
        avg_recon = epoch_recon / len(dataloader)
        avg_smooth = epoch_smooth / len(dataloader)
        hist_recon.append(avg_recon)
        hist_smooth.append(avg_smooth)

    # Save
    model_save_path = os.path.join(checkpoint_dir, "motion_ae_smooth.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    plot_ae_losses(
        hist_recon, hist_smooth, save_dir=checkpoint_dir, aux_label="Smoothness Loss"
    )

    return model


if __name__ == "__main__":
    # comment/uncomment the desired training

    # train_contrastive_ae()
    train_smooth_ae()
