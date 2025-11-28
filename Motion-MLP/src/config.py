import os

import numpy as np

# Paths
DATA_DIR = "data/HumanML3D"
VEC_DIR = os.path.join(DATA_DIR, "new_joint_vecs")
CHECKPOINT_DIR = "checkpoints"

# Model Checkpoint for Inference
INFERENCE_MODEL_PATH = os.path.join("pretrained", "best_model.pth")

# Data Params
DIMENSION = 263
SEQ_LEN = 30
MEAN = np.load(os.path.join(DATA_DIR, "Mean.npy"))
STD = np.load(os.path.join(DATA_DIR, "Std.npy"))

# Model Params
HIDDEN_DIM = 1024
NUM_LAYERS = 3

# Training Params
BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 40
NUM_WORKERS = 4
PATIENCE = 5

# Inference
GENERATED_FRAMES = 30
