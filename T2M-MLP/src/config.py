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
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF_START = 1.0
TF_END = 0.7

# inference
INFERENCE_MODEL_PATH = CHECKPOINT_DIR + "/4_scheduled_sampling_ckpt_ep20.pt"
OUTPUT_PATH = "output/motion.gif"
