import torch

# general
MOTION_DIM = 263

# model
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
# "google/gemma-2-2b"
# "Qwen/Qwen1.5-0.5B"

# paths
DATA_ROOT = "./dataset/HumanML3D"
CHECKPOINT_DIR = "checkpoints/attempt_18_test_2"

# autoencoder pretraining
AUTOENCODER_CHECKPOINT_DIR = "checkpoints_ae"
AE_BATCH_SIZE = 64
AE_TRAIN_EPOCHS = 100
LAMBDA_CONTRASTIVE = 0.1  # weight for contrastive loss in autoencoder pretraining
LAMBDA_SMOOTHNESS = 0.1  # weight for smoothness loss in autoencoder pretraining
POSITIVE_WINDOW = 5  # frames within [t-window, t+window] are considered positive pairs
AE_NOISE_LEVEL = 0.1

# training
AUTOENCODER_TO_USE_PATH = ""  # "checkpoints_ae/google_gemma-2-2b/motion_ae_smooth.pt"
RUN_BASELINE_LOSS_CHECK = False
CONTINUE_WITH_CHECKPOINT = False
CHECKPOINT_TO_CONTINUE_PATH = (
    "checkpoints/attempt_13_con_tf_0.8/trained_params/trained_params_ep200.pt"
)
WEIGHT_DECAY = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
LR = 1e-4
LR_MIN = 1e-5
EPOCHS = 10
LOWEST_TF_RATIO = (
    0.2  # teacher forcing drops from 1.0 to this value linearly during training
)

LAMBDA_POS = 1.0  # weight for position loss
LAMBDA_SEMANTIC = 5.0  # weight for semantic loss
LAMBDA_VEL = 3.0  # weight for velocity loss
LAMBDA_LANG = 0.0  # weight for language loss

# Classifier Free Guidance
USE_CFG = False
COND_DROPOUT_RATE = 0.1  # probability of dropping text conditioning during training
GUIDANCE_SCALE = 2.5

# inference
INFERENCE_MODEL_PATH = "pretrained/10_trained_params_ep100.pt"
INFERENCE_MODEL_EPOCH = 100
OUTPUT_PATH = "output/motion.gif"
