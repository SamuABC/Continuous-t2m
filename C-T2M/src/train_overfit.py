import os

import config as cfg
import torch
import torch.optim as optim
from dataset import HumanML3DDataset
from model import MotionModelCont
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import validate_visual

if __name__ == "__main__":
    # --- Setup ---
    model = MotionModelCont(
        base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM
    ).to(cfg.DEVICE)
    assert (
        cfg.TRAIN_BATCH_SIZE == cfg.EVAL_BATCH_SIZE
    ), "train/eval batch sizes must be equal for overfit test"

    # load checkpoint if continuing training
    if cfg.CONTINUE_WITH_CHECKPOINT and cfg.CHECKPOINT_TO_CONTINUE_PATH is not None:
        print(f"Loading checkpoint from {cfg.CHECKPOINT_TO_CONTINUE_PATH}...")
        model.load_state_dict(
            torch.load(cfg.CHECKPOINT_TO_CONTINUE_PATH, map_location=cfg.DEVICE),
            strict=False,
        )
        print("Checkpoint loaded successfully. Continuing previous training session.")

    train_dataset = HumanML3DDataset(
        cfg.DATA_ROOT, cfg.DATA_ROOT + "/train.txt", model.tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
    )
    first_batch = next(iter(train_dataloader))
    train_ids = first_batch["input_ids"][0]
    train_prompt = model.tokenizer.decode(train_ids, skip_special_tokens=True)

    # start with the lowest lr if continuing training
    if cfg.CONTINUE_WITH_CHECKPOINT:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LR_MIN, weight_decay=cfg.WEIGHT_DECAY
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LR_START, weight_decay=cfg.WEIGHT_DECAY
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN
    )

    PARAMS_DIRECTORY = os.path.join(cfg.CHECKPOINT_DIR, "trained_params")

    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    if not os.path.exists(PARAMS_DIRECTORY):
        os.makedirs(PARAMS_DIRECTORY)

    # store plotting data
    train_losses_epoch = []
    train_losses_motion_epoch = []
    train_losses_lang_epoch = []
    val_metrics = {"epochs": [], "fid": [], "diversity": [], "matching": []}

    # --- Training Loop ---
    print(f"Starting training on {cfg.DEVICE}...")
    if cfg.LAMBDA_LANG > 0.0:
        print("Language loss enabled.")
    if cfg.USE_CFG:
        print("Classifier Free Guidance enabled.")

    for epoch in tqdm(range(cfg.EPOCHS), desc="Training Epochs"):
        if cfg.CONTINUE_WITH_CHECKPOINT:
            tf_ratio = cfg.LOWEST_TF_RATIO  # use lowest ratio when continuing training
        else:
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
        total_motion_loss = 0
        total_lang_loss = 0

        input_ids = first_batch["input_ids"].to(cfg.DEVICE)
        motion = first_batch["motion"].to(cfg.DEVICE)
        motion_mask = first_batch["motion_mask"].to(cfg.DEVICE)

        optimizer.zero_grad()
        loss_motion, loss_lang, total_loss, _ = model(
            input_ids, motion, motion_mask, teacher_forcing_ratio=tf_ratio
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # gradient clipping
        optimizer.step()

        # log loss
        current_loss = total_loss.item()
        total_train_loss += current_loss
        total_motion_loss += loss_motion.item()
        total_lang_loss += loss_lang.item()

        scheduler.step()

        avg_train_loss = total_train_loss
        train_losses_epoch.append(avg_train_loss)

        if cfg.LAMBDA_LANG > 0.0:
            avg_motion_loss = total_motion_loss
            avg_lang_loss = total_lang_loss
            train_losses_motion_epoch.append(avg_motion_loss)
            train_losses_lang_epoch.append(avg_lang_loss)

        trainable_state_dict = {
            k: v for k, v in model.named_parameters() if v.requires_grad
        }

        if (epoch + 1) % 25 == 0:
            validate_visual(model, epoch + 1, cfg.CHECKPOINT_DIR + "/visualizations")
            torch.save(
                trainable_state_dict,
                os.path.join(PARAMS_DIRECTORY, f"trained_params_ep{epoch+1}.pt"),
            )

        # save latest model every epoch
        torch.save(
            trainable_state_dict,
            os.path.join(PARAMS_DIRECTORY, f"trained_params_latest.pt"),
        )

        if cfg.LAMBDA_LANG > 0.0:
            print(
                f"Epoch {epoch + 1} Done. Train Loss: {avg_train_loss:.4f} | Motion Loss: {avg_motion_loss:.4f} | Lang Loss: {avg_lang_loss:.4f}\n"
            )
        else:
            print(f"Epoch {epoch + 1} Done. Train Loss: {avg_train_loss:.4f}\n")

    print("Training complete.")
