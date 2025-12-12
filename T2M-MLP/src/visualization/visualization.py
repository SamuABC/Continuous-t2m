import torch
import numpy as np

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric


def visualize_transformer_motion(model, prompt, output_path="output/qwen_motion.gif"):
    """
    Generates motion from text using the MotionQwen model and visualizes it.
    
    Args:
        model: Instance of MotionQwen
        prompt: Text description (str)
        mean: Normalization mean (numpy array)
        std: Normalization std (numpy array)
        output_path: Path to save the GIF
    """
    print(f"Generating motion for: '{prompt}'...")
    mean = np.load("HumanML3D/Mean.npy")
    std = np.load("HumanML3D/Std.npy")
    
    # 1. Generate Motion
    # Uses the generate() method implemented in the previous step
    # Returns tensor of shape (1, Seq_Len, Motion_Dim)
    with torch.no_grad():
        generated_motion = model.generate(prompt, max_new_tokens=120)
    
    # 2. Prepare Data
    # Remove batch dimension -> (Seq_Len, Motion_Dim)
    y_pred = generated_motion[0].cpu().numpy()
    
    # 3. Denormalize
    y_pred_denorm = y_pred * std + mean
    
    # 4. Recover XYZ (Skeleton)
    # recover_from_ric expects a FloatTensor
    # 22 is the standard joint count for HumanML3D
    pred_xyz = recover_from_ric(torch.from_numpy(y_pred_denorm).float(), 22)
    
    # Reshape for plot_3d: (Batch, Seq_Len, Joints, Coords) -> (1, T, 22, 3)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    
    # 5. Plot
    # We pass the prompt as the title for the GIF
    plot_3d.draw_to_batch(
        xyz.detach().cpu().numpy(), 
        [prompt], 
        [output_path]
    )
    print(f"Successfully generated motion and saved it to {output_path}")