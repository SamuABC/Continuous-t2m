import codecs
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs
from evaluation import evaluate_diversity, evaluate_fid, evaluate_matching_score
from guoevaluation.dataset_motion_loader import get_dataset_motion_loader
from guoevaluation.evaluator_wrapper import EvaluatorModelWrapper
from guoevaluation.get_opt import get_opt
from model import MotionQwen
from model_motion_loader import get_qwen_model_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualization.visualization import visualize_transformer_motion

from dataset import HumanML3DDataset


def save_history(
    epoch, train_ep, train_pos_ep, train_vel_ep, train_lang_ep, val_metrics
):
    """Saves raw data to JSON."""
    data = {
        "number_of_epochs": epoch,
        "train_loss_epoch": train_ep,
        "train_loss_pos_epoch": train_pos_ep,
        "train_loss_vel_epoch": train_vel_ep,
        "train_loss_lang_epoch": train_lang_ep,
        "validation_metrics": val_metrics,
    }
    try:
        with open(os.path.join(cfg.CHECKPOINT_DIR, "training_history.json"), "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save training history: {e}")


def plot_metrics(
    train_losses, train_losses_pos, train_losses_vel, train_losses_lang, val_metrics
):
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
    axs[0, 0].plot(
        train_epochs, train_losses, label="Total Loss", color="blue", linewidth=2
    )
    # position and velocity losses
    axs[0, 0].plot(
        train_epochs, train_losses_pos, label="Pos Loss", color="orange", linestyle="--"
    )
    axs[0, 0].plot(
        train_epochs,
        train_losses_vel,
        label="Vel Loss",
        color="magenta",
        linestyle="--",
    )

    if cfg.LAMBDA_LANG > 0.0:
        # only plot language loss if language loss weight > 0
        axs[0, 0].plot(
            train_epochs,
            train_losses_lang,
            label="Language Loss",
            color="brown",
            linestyle="--",
            alpha=0.8,
        )
    axs[0, 0].set_title(f"Training Loss (Lang_lambda={cfg.LAMBDA_LANG})")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].grid(True)

    # 2. FID
    if val_epochs:
        axs[0, 1].plot(
            val_epochs, val_metrics["fid"], label="FID", marker="o", color="red"
        )
    axs[0, 1].axhline(
        y=0.0016, color="red", linestyle="--", label="Ground Truth (0.002)"
    )
    axs[0, 1].set_title("FID")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("FID")
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].grid(True)

    # 3. Diversity
    if val_epochs:
        axs[1, 0].plot(
            val_epochs,
            val_metrics["diversity"],
            label="Diversity",
            marker="o",
            color="green",
        )
    axs[1, 0].axhline(
        y=9.5225, color="green", linestyle="--", label="Ground Truth (9.52)"
    )
    axs[1, 0].set_title("Diversity")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Score")
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].grid(True)

    # 4. Matching
    if val_epochs:
        axs[1, 1].plot(
            val_epochs,
            val_metrics["matching"],
            label="Matching",
            marker="o",
            color="purple",
        )
    axs[1, 1].axhline(
        y=2.9554, color="purple", linestyle="--", label="Ground Truth (2.96)"
    )
    axs[1, 1].set_title("Matching Score")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("Score")
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].grid(True)

    # show x-ticks 5, 10, 15, ...
    # for ax in axs.flat:
    #     xmin, xmax = ax.get_xlim()
    #     ax.set_xticks(range(0, int(xmax) + 1, 5))

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "metrics_plot.png"))
    plt.close()


def validate_visual(model, epoch, save_dir):
    """
    Loads the first prompts directly from val.txt and test.txt files,
    generates motion, and saves GIFs.
    """
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\n--- Generating Visual Validation for Epoch {epoch} ---")

    mean = np.load(os.path.join(cfg.DATA_ROOT, "Mean.npy"))
    std = np.load(os.path.join(cfg.DATA_ROOT, "Std.npy"))

    # Define splits and their corresponding file names
    splits = {"train": "train.txt", "val": "val.txt"}

    FIRST_N = 3  # number of samples to generate per split

    for split_name, filename in splits.items():
        split_path = os.path.join(cfg.DATA_ROOT, filename)

        try:
            # Read all IDs and take the first 2
            with open(split_path, "r") as f:
                ids = [line.strip() for line in f.readlines()]
                target_ids = ids[:FIRST_N]

            for i, name in enumerate(target_ids):
                # Construct path to text file
                text_path = os.path.join(cfg.DATA_ROOT, "texts", name + ".txt")

                # Read text (logic from HumanML3DDataset)
                with codecs.open(text_path, "r", encoding="utf-8") as f:
                    texts = [line.strip().split("#")[0] for line in f.readlines()]
                    prompt = texts[0]

                # Generate
                with torch.no_grad():
                    generated_motion = model.generate(prompt)

                # Post-processing
                motion_data = generated_motion[0].cpu().numpy()
                motion_data = motion_data * std + mean

                # Output filename: e.g., ep1_val_0_000123.gif
                out_name = f"ep{epoch}_{split_name}_{i}_{name}.gif"
                output_path = os.path.join(save_dir, out_name)

                visualize_transformer_motion(
                    motion_data, prompt, output_path=output_path
                )

        except Exception as e:
            print(f"Error processing split {split_name}: {e}")

    print()

    model.train()


if __name__ == "__main__":
    # --- Setup ---
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )  # ignore unused params

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    model = MotionQwen(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM)

    # load checkpoint if continuing training
    if cfg.CONTINUE_WITH_CHECKPOINT:
        print(f"Loading checkpoint from {cfg.CHECKPOINT_TO_CONTINUE_PATH}...")
        model.load_state_dict(
            torch.load(cfg.CHECKPOINT_TO_CONTINUE_PATH, map_location=device),
            strict=False,
        )
        print("Checkpoint loaded successfully. Continuing previous training session.")

    # data loader
    train_dataset = HumanML3DDataset(
        cfg.DATA_ROOT, cfg.DATA_ROOT + "/train.txt", model.tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    # optimizer
    # start with the lowest lr if continuing training
    if cfg.CONTINUE_WITH_CHECKPOINT:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LR_MIN, weight_decay=cfg.WEIGHT_DECAY
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
        )

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN
    )

    # prepare with accelerator for parallel training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # eval setup
    dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD01/opt.txt"
    eval_split_file = "val.txt"
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    gt_loader, gt_dataset = get_dataset_motion_loader(
        dataset_opt_path, cfg.EVAL_BATCH_SIZE, device, _split_file=eval_split_file
    )
    eval_log_path = os.path.join(cfg.CHECKPOINT_DIR, "eval_during_training.log")

    PARAMS_DIRECTORY = os.path.join(cfg.CHECKPOINT_DIR, "trained_params")

    if accelerator.is_main_process:
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(PARAMS_DIRECTORY, exist_ok=True)

        # clear eval log file
        with open(eval_log_path, "w") as f:
            pass

    accelerator.wait_for_everyone()

    # store plotting data
    train_losses_epoch = []
    train_losses_pos_epoch = []
    train_losses_vel_epoch = []
    train_losses_lang_epoch = []
    val_metrics = {"epochs": [], "fid": [], "diversity": [], "matching": []}

    # --- Training Loop ---
    print(f"Starting training on {device}...")

    if accelerator.is_main_process:
        if cfg.LAMBDA_LANG > 0.0:
            print("Language loss enabled.")
        if cfg.USE_CFG:
            print("Classifier Free Guidance enabled.")

    for epoch in range(cfg.EPOCHS):
        if cfg.CONTINUE_WITH_CHECKPOINT:
            tf_ratio = cfg.LOWEST_TF_RATIO  # use lowest ratio when continuing training
        else:
            if epoch < 0.5 * cfg.EPOCHS:
                # warm-up phase (0-50% epochs): full teacher forcing
                tf_ratio = 1.0
            elif epoch < 0.9 * cfg.EPOCHS:
                # linear decay phase (50-90% epochs)
                progress = (epoch - 0.2 * cfg.EPOCHS) / (0.6 * cfg.EPOCHS)
                # decay from 1.0 down to cfg.LOWEST_TF_RATIO
                tf_ratio = 1.0 - (progress * (1.0 - cfg.LOWEST_TF_RATIO))
            else:
                # final phase (90-100% epochs): hold at lowest ratio
                tf_ratio = cfg.LOWEST_TF_RATIO

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1} | Teacher Forcing: {tf_ratio:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # --- Training ---
        model.train()
        total_train_loss = 0
        total_pos_loss = 0
        total_vel_loss = 0
        total_lang_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1} [Train]",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"]
            motion = batch["motion"]
            motion_mask = batch["motion_mask"]

            optimizer.zero_grad()
            loss_pos, loss_vel, loss_lang, total_loss, _ = model(
                input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio
            )
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # gradient clipping

            optimizer.step()

            # log loss
            total_train_loss += total_loss.item()
            total_pos_loss += loss_pos.item()
            total_vel_loss += loss_vel.item()
            total_lang_loss += loss_lang.item()
            progress_bar.set_postfix(
                {
                    "loss": total_loss.item(),
                    "pos_loss": loss_pos.item(),
                    "vel_loss": loss_vel.item(),
                    "lang_loss": loss_lang.item(),
                }
            )

        scheduler.step()

        # epoch averages
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_pos_loss = total_pos_loss / len(train_dataloader)
        avg_vel_loss = total_vel_loss / len(train_dataloader)

        train_losses_epoch.append(avg_train_loss)
        train_losses_pos_epoch.append(avg_pos_loss)
        train_losses_vel_epoch.append(avg_vel_loss)

        if cfg.LAMBDA_LANG > 0.0:
            avg_lang_loss = total_lang_loss / len(train_dataloader)
            train_losses_lang_epoch.append(avg_lang_loss)

        if accelerator.is_main_process:
            if cfg.LAMBDA_LANG > 0.0:
                print(
                    f"Epoch {epoch + 1} Done. Train Loss: {avg_train_loss:.4f} | Lang Loss: {avg_lang_loss:.4f}\n"
                )
            else:
                print(f"Epoch {epoch + 1} Done. Train Loss: {avg_train_loss:.4f}\n")

        # --- Validation + model save---
        if (epoch + 1) % 10 == 0 or epoch == cfg.EPOCHS - 1 or epoch == 0:
            # wait for all processes
            accelerator.wait_for_everyone()

            # unwrap model
            unwrapped_model = accelerator.unwrap_model(model)

            trainable_state_dict = {
                k: v for k, v in unwrapped_model.named_parameters() if v.requires_grad
            }

            if accelerator.is_main_process:
                print(f"\n--- Running Evaluation at Epoch {epoch + 1} ---")
                unwrapped_model.eval()

                gen_loader = get_qwen_model_loader(unwrapped_model, gt_loader, device)

                motion_loaders = {"MotionQwen": gen_loader}

                with open(eval_log_path, "a") as f:
                    print(f"Epoch {epoch+1}", file=f, flush=True)

                    # Matching Score & R-Precision
                    (
                        mat_score_dict,
                        R_precision_dict,
                        acti_dict,
                    ) = evaluate_matching_score(eval_wrapper, motion_loaders, f)

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
                validate_visual(
                    unwrapped_model, epoch + 1, cfg.CHECKPOINT_DIR + "/visualizations"
                )

                # plot
                plot_metrics(
                    train_losses_epoch,
                    train_losses_pos_epoch,
                    train_losses_vel_epoch,
                    train_losses_lang_epoch,
                    val_metrics,
                )

                # save model params
                torch.save(
                    trainable_state_dict,
                    os.path.join(PARAMS_DIRECTORY, f"trained_params_ep{epoch+1}.pt"),
                )

                # save training history
                save_history(
                    epoch + 1,
                    train_losses_epoch,
                    train_losses_pos_epoch,
                    train_losses_vel_epoch,
                    train_losses_lang_epoch,
                    val_metrics,
                )

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Training complete.")
