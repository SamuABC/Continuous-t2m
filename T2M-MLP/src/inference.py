import torch
from visualization.visualization import visualize_transformer_motion
from model import MotionQwen

# --- Usage Example ---
# Assuming 'cfg' has MEAN and STD loaded from the dataset meta files
# prompt = "A person walks in a circle"
# visualize_transformer_motion(model, prompt, cfg.MEAN, cfg.STD)
prompt = "A person walking forward."
model = MotionQwen(base_model_id="Qwen/Qwen1.5-0.5B", motion_dim=263)
model.load_state_dict(torch.load("checkpoints/motion_qwen_epoch_4.pt"))
model.eval()
visualize_transformer_motion(model, prompt, output_path="output/motion.gif")
