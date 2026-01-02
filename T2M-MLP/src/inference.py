import sys

import config as cfg
import numpy as np
import torch
from model import MotionQwen
from visualization.visualization import visualize_transformer_motion


def generate(prompt: str):
    """
    generates and visualizes a motion sequence from a text prompt
    """
    model = MotionQwen(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM)
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(cfg.INFERENCE_MODEL_PATH, map_location=cfg.DEVICE), 
        strict=False
    )

    # safety check if all critical modules are loaded
    critical_modules = ["motion_encoder", "motion_decoder", "lora_A", "lora_B"]
    for key in missing_keys:
        for critical in critical_modules:
            if critical in key:
                print(f"WARNING: Critical key not loaded: {key}")

    model.to(cfg.DEVICE)

    model.eval()

    print(f"Generating motion for: '{prompt}'...")
    mean = np.load(cfg.DATA_ROOT + "/Mean.npy")
    std = np.load(cfg.DATA_ROOT + "/Std.npy")

    # returns tensor of shape (1, Seq_Len, Motion_Dim)
    with torch.no_grad():
        generated_motion = model.generate(prompt, max_new_tokens=120)

    # remove batch dimension -> (Seq_Len, Motion_Dim)
    y_pred = generated_motion[0].cpu().numpy()

    # denormalize
    y_pred_denorm = y_pred * std + mean

    visualize_transformer_motion(y_pred_denorm, prompt)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        generate(prompt)
    else:
        print("No prompt provided.")
