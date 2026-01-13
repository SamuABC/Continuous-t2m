import torch

# general
MOTION_DIM = 263

# model
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
STOP_PROB_THRESHOLD = 0.8
STOP_LOSS_WEIGHT = 0.1

# paths
DATA_ROOT = "./HumanML3D"
CHECKPOINT_DIR = "checkpoints"

# training
BATCH_SIZE = 16
LR = 1e-4
LR_MIN = 1e-5
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# inference
INFERENCE_MODEL_PATH = "pretrained/8_trained_params_ep60.pt"
INFERENCE_MODEL_EPOCH = 60
OUTPUT_PATH = "output/motion.gif"
