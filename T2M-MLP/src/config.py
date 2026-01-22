import torch

# general
MOTION_DIM = 263

# model
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

# paths
DATA_ROOT = "./dataset/HumanML3D"
CHECKPOINT_DIR = "checkpoints/attempt_10_continue"

# training
CONTINUE_WITH_CHECKPOINT = True
CHECKPOINT_TO_CONTINUE_PATH = (
    "checkpoints/attempt_10/trained_params/trained_params_ep100.pt"
)
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LR = 1e-4
LR_MIN = 1e-5
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOWEST_TF_RATIO = 0.2

# inference
INFERENCE_MODEL_PATH = "pretrained/10_trained_params_ep100.pt"
INFERENCE_MODEL_EPOCH = 100
OUTPUT_PATH = "output/motion.gif"
