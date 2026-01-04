import torch

# general
MOTION_DIM = 263
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

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
INFERENCE_MODEL_PATH = "pretrained/trained_params_ep69.pt"
OUTPUT_PATH = "output/motion.gif"
