import torch

# general
MOTION_DIM = 263

# model
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

# paths
DATA_ROOT = "./dataset/HumanML3D"
CHECKPOINT_DIR = "checkpoints/attempt_14"

# training
CONTINUE_WITH_CHECKPOINT = False
CHECKPOINT_TO_CONTINUE_PATH = (
    "checkpoints/attempt_13_con_tf_0.8/trained_params/trained_params_ep200.pt"
)
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LR = 1e-4
LR_MIN = 1e-5
WEIGHT_DECAY = 0.0
EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOWEST_TF_RATIO = (
    0.2  # teacher forcing drops from 1.0 to this value linearly during training
)

LAMBDA_VEL = 2.0  # weight for velocity loss

# language loss (set to 0.0 to disable language loss)
LAMBDA_LANG = 0.0  # weight for language loss

# Classifier Free Guidance
USE_CFG = False
COND_DROPOUT_RATE = 0.1  # probability of dropping text conditioning during training
GUIDANCE_SCALE = 2.5

# inference
INFERENCE_MODEL_PATH = "pretrained/10_trained_params_ep100.pt"
INFERENCE_MODEL_EPOCH = 100
OUTPUT_PATH = "output/motion.gif"
