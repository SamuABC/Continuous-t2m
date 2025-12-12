import torch
import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric


def visualize_mlp_motion(y_pred, mean, std):
    # y_pred is a Tensor (GPU or CPU), mean/std are numpy arrays
    y_pred = y_pred.cpu().numpy()

    # Denormalize
    y_pred_denorm = y_pred * std + mean

    # Recover & Plot
    pred_xyz = recover_from_ric(torch.from_numpy(y_pred_denorm).float(), 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)

    plot_3d.draw_to_batch(
        xyz.detach().cpu().numpy(), ["Autoregressive Motion"], ["output/motion.gif"]
    )
    print("successfully generated motion and saved it to output/motion.gif")
