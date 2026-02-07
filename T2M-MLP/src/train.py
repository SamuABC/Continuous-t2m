import codecs
import gc
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import timedelta

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from evaluation import evaluate_diversity, evaluate_fid, evaluate_matching_score
from guoevaluation.dataset_motion_loader import get_dataset_motion_loader
from guoevaluation.evaluator_wrapper import EvaluatorModelWrapper
from guoevaluation.get_opt import get_opt
from model import MotionModelCont
from model_motion_loader import get_qwen_model_loader
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualization.visualization import visualize_transformer_motion

from dataset import HumanML3DDataset


def save_history(epoch, loss_history_dict, val_metrics):
    """Saves raw data to JSON."""
    data = {
        "number_of_epochs": epoch,
        "loss_history": loss_history_dict,
        "validation_metrics": val_metrics,
    }
    try:
        with open(os.path.join(cfg.CHECKPOINT_DIR, "training_history.json"), "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save training history: {e}")


def plot_metrics(history_dict, val_metrics, tf_ratios):
    """
    Plots metrics
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))

    epochs = range(1, len(history_dict["loss"]) + 1)
    val_epochs = val_metrics["epochs"]

    # --- Row 1: Losses ---

    # 1. Total Loss
    if "loss" in history_dict:
        axs[0, 0].plot(
            epochs, history_dict["loss"], label="Total Loss", color="blue", linewidth=2
        )
    axs[0, 0].set_title("Total Training Loss")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 2. Component Losses
    colors = ["orange", "green", "red", "purple", "brown", "darkcyan"]
    color_idx = 0

    for key, values in history_dict.items():
        if key == "loss":
            continue  # Total loss already plotted

        if np.mean(values) < 0:
            continue  # skip invalid losses

        style = "--"
        lbl = f"{key} Loss"

        axs[0, 1].plot(
            epochs,
            values,
            label=lbl,
            linestyle=style,
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    axs[0, 1].set_title("Component Losses")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # --- Row 2: Hyperparams & FID ---

    # 3. Teacher Forcing Ratio
    axs[1, 0].plot(epochs, tf_ratios, label="TF Ratio", color="teal", linewidth=2)
    axs[1, 0].set_title("Teacher Forcing Schedule")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Ratio")
    axs[1, 0].set_ylim(-0.1, 1.1)
    axs[1, 0].grid(True)

    # 4. FID
    if val_epochs:
        axs[1, 1].plot(
            val_epochs, val_metrics["fid"], label="FID", marker="o", color="red"
        )
    axs[1, 1].axhline(
        y=0.0016, color="red", linestyle="--", label="Ground Truth (0.002)"
    )
    axs[1, 1].set_title("FID (Lower is better)")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("FID")
    axs[1, 1].grid(True)

    # --- Row 3: Diversity & Matching ---

    # 5. Diversity
    if val_epochs:
        axs[2, 0].plot(
            val_epochs,
            val_metrics["diversity"],
            label="Diversity",
            marker="o",
            color="green",
        )
    axs[2, 0].axhline(
        y=9.5225, color="green", linestyle="--", label="Ground Truth (9.52)"
    )
    axs[2, 0].set_title("Diversity (Closer to GT is better)")
    axs[2, 0].set_xlabel("Epochs")
    axs[2, 0].set_ylabel("Score")
    axs[2, 0].grid(True)

    # 6. Matching
    if val_epochs:
        axs[2, 1].plot(
            val_epochs,
            val_metrics["matching"],
            label="Matching",
            marker="o",
            color="purple",
        )
    axs[2, 1].axhline(
        y=2.9554, color="purple", linestyle="--", label="Ground Truth (2.96)"
    )
    axs[2, 1].set_title("Matching Score (Lower is better)")
    axs[2, 1].set_xlabel("Epochs")
    axs[2, 1].set_ylabel("Score")
    axs[2, 1].grid(True)

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

    # longer timeout
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=120))

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, process_group_kwargs])
    device = accelerator.device

    model = MotionModelCont(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM)

    # save VRAM by enabling gradient checkpointing
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False

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
        num_workers=2,
        pin_memory=True,
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

    tf_decay_start = int(cfg.EPOCHS * (1 / 3))  # Start decay at 1/3
    tf_decay_end = int(
        cfg.EPOCHS * (5 / 6)
    )  # End decay at 5/6 (keeping last 1/6 stable)

    # scheduler
    scheduler1 = ConstantLR(optimizer, factor=1.0, total_iters=tf_decay_start)
    scheduler2 = CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS - tf_decay_start, eta_min=cfg.LR_MIN
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[tf_decay_start]
    )

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

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
    loss_history_dict = {}
    tf_ratios_epoch = []
    val_metrics = {"epochs": [], "fid": [], "diversity": [], "matching": []}

    # --- Training Loop ---
    print(f"Starting training on {device}...")

    if accelerator.is_main_process:
        print(f"Training with backbone {cfg.BASE_MODEL_ID}")
        if cfg.LAMBDA_LANG > 0.0:
            print("Language loss enabled.")
        if cfg.USE_CFG:
            print("Classifier Free Guidance enabled.")

    if cfg.RUN_BASELINE_LOSS_CHECK:
        if accelerator.is_main_process:
            print("\n" + "=" * 40)
            print("--- RUNNING BASELINE LOSS CHECK (No Optimization) ---")
            print("=" * 40)

        model.train()
        baseline_accumulators = {}
        num_baseline_batches = 50
        limit_batches = min(num_baseline_batches, len(train_dataloader))

        with torch.no_grad():
            check_bar = tqdm(
                train_dataloader,
                total=limit_batches,
                desc="Baseline Check",
                disable=not accelerator.is_main_process,
            )

            for i, batch in enumerate(check_bar):
                if i >= limit_batches:
                    break

                input_ids = batch["input_ids"]
                motion = batch["motion"]
                motion_mask = batch["motion_mask"]

                loss_logs, _, _ = model(
                    input_ids, motion, motion_mask, teacher_forcing_ratio=1.0
                )

                for k, v in loss_logs.items():
                    if k not in baseline_accumulators:
                        baseline_accumulators[k] = 0.0
                    baseline_accumulators[k] += v

        if accelerator.is_main_process:
            print("\n" + "-" * 20 + " RESULTS " + "-" * 20)
            print(f"Mean Losses over the first {limit_batches} Batches:")

            for k, v_sum in baseline_accumulators.items():
                avg_val = v_sum / limit_batches
                print(f"  {k:.<20}: {avg_val:.6f}")

    for epoch in range(cfg.EPOCHS):
        if cfg.CONTINUE_WITH_CHECKPOINT:
            tf_ratio = cfg.LOWEST_TF_RATIO
        else:
            if epoch < tf_decay_start:
                # Phase 1: Stable at 1.0
                tf_ratio = 1.0
            elif epoch < tf_decay_end:
                # Phase 2: Linear decay
                progress = (epoch - tf_decay_start) / (tf_decay_end - tf_decay_start)
                tf_ratio = 1.0 - (progress * (1.0 - cfg.LOWEST_TF_RATIO))
            else:
                # Phase 3: Stable at lowest ratio
                tf_ratio = cfg.LOWEST_TF_RATIO

        tf_ratios_epoch.append(tf_ratio)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1} | Teacher Forcing: {tf_ratio:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # --- Training ---
        model.train()
        epoch_accumulators = {}
        num_batches = len(train_dataloader)
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
            loss_logs, total_loss, _ = model(
                input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio
            )
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # gradient clipping

            optimizer.step()

            # log loss
            for k, v in loss_logs.items():
                if k not in epoch_accumulators:
                    epoch_accumulators[k] = 0.0
                epoch_accumulators[k] += v
            progress_bar.set_postfix(loss_logs)

        scheduler.step()

        # epoch averages
        log_string = f"Epoch {epoch + 1} Done."

        for k, v_sum in epoch_accumulators.items():
            avg_val = v_sum / num_batches

            # save to history
            if k not in loss_history_dict:
                loss_history_dict[k] = []
            loss_history_dict[k].append(avg_val)

            log_string += f" | {k}: {avg_val:.4f}"

        if accelerator.is_main_process:
            print(log_string + "\n")

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
                    loss_history_dict,
                    val_metrics,
                    tf_ratios_epoch,
                )

                # save model params
                torch.save(
                    trainable_state_dict,
                    os.path.join(PARAMS_DIRECTORY, f"trained_params_ep{epoch+1}.pt"),
                )

                # save training history
                save_history(
                    epoch + 1,
                    loss_history_dict,
                    val_metrics,
                )

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Training complete.")
