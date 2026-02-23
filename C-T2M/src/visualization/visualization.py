import config as cfg
import torch
import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric


def visualize_transformer_motion(y_pred_denorm, prompt, output_path=cfg.OUTPUT_PATH):
    """
    Visualizes a motion as a gif.
    """

    joint_count = 22  # HumanML3D has 22 joints
    pred_xyz = recover_from_ric(torch.from_numpy(y_pred_denorm).float(), joint_count)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), [prompt], [output_path])
    print(f"Successfully generated motion and saved it to {output_path}")
