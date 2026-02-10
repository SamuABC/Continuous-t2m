import torch

# general
MOTION_DIM = 263
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model
BASE_MODEL_ID = "google/gemma-2-2b"
# "google/gemma-2-2b"
# "Qwen/Qwen1.5-0.5B"

# paths
# dataset
DATA_ROOT = "./dataset/HumanML3D"
# path to save current t2m training results
CHECKPOINT_DIR = "checkpoints/attempt_19_baseline"
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
EPOCHS = 150
LOWEST_TF_RATIO = 0.2  # tf drops from 1.0 to this value linearly during training
LAMBDA_POS = 1.0  # weight for position loss
LAMBDA_SEMANTIC = 5.0  # weight for semantic loss
LAMBDA_VEL = 3.0  # weight for velocity loss
LAMBDA_LANG = 0.0  # weight for language loss

# Classifier Free Guidance
# flags
USE_CFG = False
# hyperparameters
COND_DROPOUT_RATE = 0.1  # probability of dropping text conditioning during training
GUIDANCE_SCALE = 2.5

# inference
VISUAL_VAL_EPOCH_PRINT = 100  # just to inform about the inference epoch number
