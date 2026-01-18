import torch

# general
MOTION_DIM = 263

# model
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

# paths
DATA_ROOT = "./dataset/HumanML3D"
CHECKPOINT_DIR = "checkpoints/attempt_9"

# training
BATCH_SIZE = 32
LR = 1e-4
LR_MIN = 1e-5
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# inference
INFERENCE_MODEL_PATH = "pretrained/7_trained_params_ep69.pt"
INFERENCE_MODEL_EPOCH = 69
OUTPUT_PATH = "output/motion.gif"
