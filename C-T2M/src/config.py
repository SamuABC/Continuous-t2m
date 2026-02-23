import argparse

import torch

# general
MOTION_DIM = 263
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# model
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
# "google/gemma-2-2b"
# "Qwen/Qwen1.5-0.5B"

# paths
# dataset
DATA_ROOT = "./dataset/HumanML3D"
# path to save current c-t2m training results
CHECKPOINT_DIR = "checkpoints/dir_not_configured"
# path to save current autoencoder pretraining results
AUTOENCODER_CHECKPOINT_DIR = "checkpoints_ae"
# path to pretrained t2m model to continue training from
CHECKPOINT_TO_CONTINUE_PATH = ""
# path to pretrained autoencoder to use in t2m training
AUTOENCODER_TO_USE_PATH = ""
# model used for inference and evaluation
INFERENCE_MODEL_PATH = "pretrained/10_trained_params_ep100.pt"
# path for visualizations during inference
OUTPUT_PATH = "output/motion.gif"

# autoencoder pretraining hyperparameters
AE_BATCH_SIZE = 64
AE_TRAIN_EPOCHS = 100
LAMBDA_CONTRASTIVE = 0.1  # weight for contrastive loss in autoencoder pretraining
LAMBDA_SMOOTHNESS = 0.1  # weight for smoothness loss in autoencoder pretraining
POSITIVE_WINDOW = 5  # frames within [t-window, t+window] are considered positive pairs
AE_NOISE_LEVEL = 0.1

# training
# flags
RUN_BASELINE_LOSS_CHECK = False
CONTINUE_WITH_CHECKPOINT = False
FREEZE_ENCODER = False
# hyperparameters
EVAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32  # (times number of gpus used for training)
WEIGHT_DECAY = 0.0
LR_START = 1e-4
LR_MIN = 1e-5
EPOCHS = 100
LOWEST_TF_RATIO = 0.2  # tf drops from 1.0 to this value linearly during training
TF_DECAY_START = 0.2  # percentage of epochs where decay of tf starts
TF_DECAY_END = 0.8  # percentage of epochs where tf stays at lowest value
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

LAMBDA_POS = 1.0  # weight for position loss
LAMBDA_VEL = 0.0  # weight for velocity loss
DROP_VEL_AT_TF_RATIO = (
    0.0  # if the tf ratio is smaller than this, the vel loss will not be applied
)

LAMBDA_SEMANTIC = 0.0  # weight for semantic loss
LAMBDA_LANG = 0.0  # weight for language loss


# Classifier Free Guidance
# flags
USE_CFG = False
# hyperparameters
COND_DROPOUT_RATE = 0.1  # probability of dropping text conditioning during training
GUIDANCE_SCALE = 2.5

# inference
VISUAL_VAL_EPOCH_PRINT = 100  # just to inform about the inference epoch number

PROMPT = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n### Input:\n"
PROMPT_END = "\n\nResponse: <Motion>"


def update_config_from_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--lambda_pos", type=float)
    parser.add_argument("--lambda_vel", type=float)
    parser.add_argument("--lambda_semantic", type=float)
    parser.add_argument("--lambda_lang", type=float)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--training_epochs", type=int)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--autoencoder_to_use_path", type=str)
    parser.add_argument("--base_model_id", type=str)

    args, unknown = parser.parse_known_args()

    if args.checkpoint_dir:
        global CHECKPOINT_DIR
        CHECKPOINT_DIR = args.checkpoint_dir
    if args.lambda_pos is not None:
        global LAMBDA_POS
        LAMBDA_POS = args.lambda_pos
    if args.lambda_vel is not None:
        global LAMBDA_VEL
        LAMBDA_VEL = args.lambda_vel
    if args.lambda_semantic is not None:
        global LAMBDA_SEMANTIC
        LAMBDA_SEMANTIC = args.lambda_semantic
    if args.lambda_lang is not None:
        global LAMBDA_LANG
        LAMBDA_LANG = args.lambda_lang
    if args.use_cfg:
        global USE_CFG
        USE_CFG = True
    if args.training_epochs is not None:
        global EPOCHS
        EPOCHS = args.training_epochs
    if args.freeze_encoder:
        global FREEZE_ENCODER
        FREEZE_ENCODER = True
    if args.autoencoder_to_use_path is not None:
        global AUTOENCODER_TO_USE_PATH
        AUTOENCODER_TO_USE_PATH = args.autoencoder_to_use_path
    if args.base_model_id is not None:
        global BASE_MODEL_ID
        BASE_MODEL_ID = args.base_model_id


def print_config():
    print("Configuration:")
    print("Base Model: " + BASE_MODEL_ID)
    print(
        "Autoencoder used: "
        + (AUTOENCODER_TO_USE_PATH if AUTOENCODER_TO_USE_PATH else "None")
    )
    if FREEZE_ENCODER:
        print("Encoder weights will be frozen during training.")
    if CONTINUE_WITH_CHECKPOINT:
        print("Continuing training from checkpoint: " + CHECKPOINT_TO_CONTINUE_PATH)
    if USE_CFG:
        print(
            "Classifier Free Guidance: Scale = "
            + str(GUIDANCE_SCALE)
            + ", Cond Dropout Rate = "
            + str(COND_DROPOUT_RATE)
        )
    print("Training Hyperparameters:")
    print("- Epochs: " + str(EPOCHS))
    print("- Train Batch Size: " + str(TRAIN_BATCH_SIZE) + " (times number of gpus)")
    print("- Eval Batch Size: " + str(EVAL_BATCH_SIZE))
    print("- Learning Rate: " + str(LR_START) + " to " + str(LR_MIN))
    print("- Weight Decay: " + str(WEIGHT_DECAY))
    print(
        "- Teacher Forcing: "
        + str(1.0)
        + " to "
        + str(LOWEST_TF_RATIO)
        + " (Decaystart at: "
        + str(TF_DECAY_START * 100)
        + "% of epochs, Stabilizationstart at: "
        + str(TF_DECAY_END * 100)
        + "% of epochs)"
    )
    print(
        "- Loss Weights: Position = "
        + str(LAMBDA_POS)
        + ", Semantic = "
        + str(LAMBDA_SEMANTIC)
        + ", Velocity = "
        + str(LAMBDA_VEL)
        + ", Language = "
        + str(LAMBDA_LANG)
    )
    print(
        "Dropping velocity loss when teacher forcing ratio < "
        + str(DROP_VEL_AT_TF_RATIO)
    )
    print(
        "- LoRA: Rank = "
        + str(LORA_RANK)
        + ", Alpha = "
        + str(LORA_ALPHA)
        + ", Dropout = "
        + str(LORA_DROPOUT)
    )
