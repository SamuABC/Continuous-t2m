import json
import os

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from evaluation import evaluate_diversity, evaluate_fid, evaluate_matching_score
from guoevaluation.dataset_motion_loader import get_dataset_motion_loader
from guoevaluation.evaluator_wrapper import EvaluatorModelWrapper
from guoevaluation.get_opt import get_opt
from matplotlib.ticker import MaxNLocator
from model import MotionQwen
from model_motion_loader import get_qwen_model_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualization.visualization import visualize_transformer_motion

from dataset import HumanML3DDataset


def save_history(epoch, train_ep, val_metrics):
    """Saves raw data to JSON including new validation metrics."""
    data = {
        "number_of_epochs": epoch,
        "train_loss_epoch": train_ep,
        "validation_metrics": val_metrics,  # Dict with lists for fid, div, matching, epochs
    }
    try:
        with open(os.path.join(cfg.CHECKPOINT_DIR, "training_history.json"), "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save training history: {e}")


def plot_metrics(train_losses, val_metrics):
    """
    Generates a 2x2 Grid:
    Top-Left: Train Loss
    Top-Right: FID
    Bottom-Left: Diversity
    Bottom-Right: Matching
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # X-Axis definitions
    train_epochs = range(1, len(train_losses) + 1)
    val_epochs = val_metrics["epochs"]  # Epochs where validation happened

    # 1. Train Loss
    axs[0, 0].plot(train_epochs, train_losses, label="Train Loss", color="blue")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].grid(True)

    # 2. FID (Lower is better)
    if val_epochs:
        axs[0, 1].plot(
            val_epochs, val_metrics["fid"], label="FID", marker="o", color="red"
        )
    axs[0, 1].set_title("FID")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("FID")
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].grid(True)

    # 3. Diversity (Higher is usually better)
    if val_epochs:
        axs[1, 0].plot(
            val_epochs,
            val_metrics["diversity"],
            label="Diversity",
            marker="o",
            color="green",
        )
    axs[1, 0].set_title("Diversity")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Score")
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].grid(True)

    # 4. Matching (Higher is better)
    if val_epochs:
        axs[1, 1].plot(
            val_epochs,
            val_metrics["matching"],
            label="Matching",
            marker="o",
            color="purple",
        )
    axs[1, 1].set_title("Matching Score")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("Score")
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].grid(True)

    # force integer ticks on x-axis for all subplots
    for ax in axs.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "metrics_plot.png"))
    plt.close()


def validate_visual(model, epoch, save_dir):
    """
    Generates and saves GIFs using the first 4 prompts from the validation set.
    """
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"--- Generating Visual Validation for Epoch {epoch} ---")

    test_prompts = [
        "a person in a sitting position with his hands forward adjusts a steering wheel left and right.",  # val: 008646
        "a person gets down on their hands and knees and crawls around, then gets back up.",  # val: 008859
        "a person is walking forward, stops and waves someone with the right hand.",  # custom
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

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN
    )

    dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD01/opt.txt"
    eval_split_file = "val.txt"
    wrapper_opt = get_opt(dataset_opt_path, cfg.DEVICE)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    gt_loader, gt_dataset = get_dataset_motion_loader(
        dataset_opt_path, cfg.BATCH_SIZE, cfg.DEVICE, _split_file=eval_split_file
    )
    eval_log_path = os.path.join(cfg.CHECKPOINT_DIR, "eval_during_training.log")

    PARAMS_DIRECTORY = os.path.join(cfg.CHECKPOINT_DIR, "trained_params")

    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    if not os.path.exists(PARAMS_DIRECTORY):
        os.makedirs(PARAMS_DIRECTORY)

    # store plotting data
    train_losses_epoch = []
    val_metrics = {"epochs": [], "fid": [], "diversity": [], "matching": []}

    # clear eval log file
    with open(eval_log_path, "w") as f:
        pass

    # --- Training Loop ---
    print(f"Starting training on {cfg.DEVICE}...")
    for epoch in range(cfg.EPOCHS):
        if epoch < 0.2 * cfg.EPOCHS:
            # warm-up phase (0-20% epochs): full teacher forcing
            tf_ratio = 1.0
        elif epoch < 0.8 * cfg.EPOCHS:
            # linear decay phase (20-80% epochs)
            progress = (epoch - 0.2 * cfg.EPOCHS) / (0.6 * cfg.EPOCHS)
            # decay from 1.0 down to cfg.LOWEST_TF_RATIO
            tf_ratio = 1.0 - (progress * (1.0 - cfg.LOWEST_TF_RATIO))
        else:
            # final phase (80-100% epochs): hold at lowest ratio
            tf_ratio = cfg.LOWEST_TF_RATIO

        print(
            f"Epoch {epoch+1} | Teacher Forcing: {tf_ratio:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

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
        if (epoch + 1) % 10 == 0 or epoch == cfg.EPOCHS - 1 or epoch == 0:
            print(f"--- Running Evaluation at Epoch {epoch + 1} ---")
            model.eval()

            gen_loader = get_qwen_model_loader(model, gt_loader, cfg.DEVICE)

            motion_loaders = {"MotionQwen": gen_loader}

            with open(eval_log_path, "a") as f:
                print(f"Epoch {epoch+1}", file=f, flush=True)

                # Matching Score & R-Precision
                mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                    eval_wrapper, motion_loaders, f
                )

                # FID
                fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

                # Diversity
                div_score_dict = evaluate_diversity(acti_dict, f)

            current_fid = fid_score_dict["MotionQwen"]
            current_div = div_score_dict["MotionQwen"]
            current_match = mat_score_dict["MotionQwen"]

            # Update metrics dictionary
            val_metrics["epochs"].append(epoch + 1)
            val_metrics["fid"].append(float(current_fid))
            val_metrics["diversity"].append(float(current_div))
            val_metrics["matching"].append(float(current_match))

            # Visual validation
            validate_visual(model, epoch + 1, cfg.CHECKPOINT_DIR + "/visualizations")

            # plot
            plot_metrics(train_losses_epoch, val_metrics)

        print(f"Epoch {epoch + 1} Done. Train: {avg_train_loss:.4f}")

        # --- Save---
        trainable_state_dict = {
            k: v for k, v in model.named_parameters() if v.requires_grad
        }
        torch.save(
            trainable_state_dict,
            os.path.join(PARAMS_DIRECTORY, f"trained_params_ep{epoch+1}.pt"),
        )
        save_history(epoch + 1, train_losses_epoch, val_metrics)

    print("Training complete.")
