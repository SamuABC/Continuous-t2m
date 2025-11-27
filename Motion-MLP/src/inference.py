from model import MotionMLP
from dataset import get_loaders, dimension, mean, std
import torch
from visualization.visualization import visualize_mlp_motion
import random

_, _, _, val_dataset = get_loaders(batch_size=1)


def sample_start_pose(device="cuda"):
    """
    Sample a random starting pose from the validation dataset.
    """
    # 1. take a random sequence
    idx = random.randint(0, len(val_dataset) - 1)
    x, _ = val_dataset[idx]  # x shape: (Seq_Len, D)

    # 2. take a random time step from the sequence
    t = random.randint(0, x.shape[0] - 1)
    start_pose = x[t]  # Shape: (D,)

    # 3. Reshape for the model: (Batch=1, Time=1, D)
    return start_pose.view(1, 1, -1).to(device)


# load model
model = MotionMLP(input_dim=dimension)
model.load_state_dict(torch.load("checkpoints/mlp_autoreg.pth"))
model.eval().cuda()

# setup for generation
generated_frames = []
current_input = sample_start_pose()  # (1, 1, D)
frames = 40

print("Generating motion...")
# Autoregressive loop
with torch.no_grad():
    for _ in range(frames):
        # Prediction
        next_pose = model(current_input)  # (1, 1, D)
        generated_frames.append(next_pose)

        # Update input for next step
        current_input = next_pose

    # Concatenate all generated frames
    y_pred = torch.cat(generated_frames, dim=1)  # (1, T, D)
    y_pred = y_pred[0]  # (T, D)

visualize_mlp_motion(y_pred, mean, std)
