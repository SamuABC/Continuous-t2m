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
# path to save current t2m training results
CHECKPOINT_DIR = "checkpoints/attempt_29_baseline"
# path to save current autoencoder pretraining results
AUTOENCODER_CHECKPOINT_DIR = "checkpoints_ae"
# path to pretrained t2m model to continue training from
CHECKPOINT_TO_CONTINUE_PATH = (
    "checkpoints/attempt_13_con_tf_0.8/trained_params/trained_params_ep200.pt"
)
# path to pretrained autoencoder to use in t2m training
AUTOENCODER_TO_USE_PATH = ""
# model used for inference and evaluation
INFERENCE_MODEL_PATH = "pretrained/10_trained_params_ep100.pt"
# path for stdout, errors and visualizations during inference
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
# hyperparameters
EVAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
WEIGHT_DECAY = 0.0
LR_START = 1e-4
LR_MIN = 1e-5
EPOCHS = 100
LOWEST_TF_RATIO = 0.2  # tf drops from 1.0 to this value linearly during training
LAMBDA_POS = 1.0  # weight for position loss
LAMBDA_SEMANTIC = 0.0  # weight for semantic loss
LAMBDA_VEL = 0.0  # weight for velocity loss
LAMBDA_LANG = 0.0  # weight for language loss
TF_WARMUP_PHASE = 1 / 5  # percentage of epochs where decay of tf starts
TF_STABILASATION_PHASE = 4 / 5  # percentage of epochs where tf stays at lowest value
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

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


def print_config():
    print("Configuration:")
    print("Base Model: " + BASE_MODEL_ID)
    print(
        "Autoencoder used: "
        + (AUTOENCODER_TO_USE_PATH if AUTOENCODER_TO_USE_PATH else "None")
    )
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
        + " (Warmupstart at: "
        + str(TF_WARMUP_PHASE * 100)
        + "% of epochs, Stabilizationstart at: "
        + str(TF_STABILASATION_PHASE * 100)
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
        "- LoRA: Rank = "
        + str(LORA_RANK)
        + ", Alpha = "
        + str(LORA_ALPHA)
        + ", Dropout = "
        + str(LORA_DROPOUT)
    )
