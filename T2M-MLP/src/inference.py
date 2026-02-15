import sys

import config as cfg
import numpy as np
import torch
from model import MotionModelCont
from train import validate_visual
from visualization.visualization import visualize_transformer_motion


def generate(prompt: str):
    """
    generates and visualizes a motion sequence from a text prompt
    """
    print("generating motion with model:", cfg.INFERENCE_MODEL_PATH)
    model = MotionModelCont(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM)
    model.load_state_dict(
        torch.load(cfg.INFERENCE_MODEL_PATH, map_location=cfg.DEVICE), strict=False
    )

    model.to(cfg.DEVICE)

    model.eval()

    print(f"Generating motion for: '{prompt}'...")
    mean = np.load(cfg.DATA_ROOT + "/Mean.npy")
    std = np.load(cfg.DATA_ROOT + "/Std.npy")

    prompt = cfg.PROMPT + prompt + cfg.PROMPT_END

    # returns tensor of shape (1, Seq_Len, Motion_Dim)
    with torch.no_grad():
        generated_motion = model.generate(prompt)

    # remove batch dimension -> (Seq_Len, Motion_Dim)
    y_pred = generated_motion[0].cpu().numpy()

    # denormalize
    y_pred_denorm = y_pred * std + mean

    visualize_transformer_motion(y_pred_denorm, prompt)


def generate_val_motions(epoch: int = cfg.VISUAL_VAL_EPOCH_PRINT):
    """
    generates and visualizes motions for test prompts.
    Arg:
        epoch: current epoch number
    """
    print("generating validation motion with model:", cfg.INFERENCE_MODEL_PATH)
    model = MotionModelCont(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM)
    model.load_state_dict(
        torch.load(cfg.INFERENCE_MODEL_PATH, map_location=cfg.DEVICE), strict=False
    )

    model.to(cfg.DEVICE)

    model.eval()
    validate_visual(model, epoch, "output/visualizations")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        generate(prompt)
    else:
        print("No prompt provided. Generating default validation motions...")
        generate_val_motions()
