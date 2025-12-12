import torch
import os
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import MotionQwen
from dataset import HumanML3DDataset, collate_fn

# --- Configuration ---
BATCH_SIZE = 16 # Adjust based on GPU VRAM (Qwen 0.5 is small, but gradients add up)
LR = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./HumanML3D"
SPLIT_FILE = "./HumanML3D/train.txt"
MODEL_ID = "Qwen/Qwen1.5-0.5B"
MOTION_DIM = 263 # HumanML3D standard dimension

# --- Setup ---
# Initialize Model (Assuming MotionQwen class from previous turn is available)
model = MotionQwen(base_model_id=MODEL_ID, motion_dim=MOTION_DIM).to(DEVICE)
model.train()

# Tokenizer is already inside the model class, but we need it for the dataset
tokenizer = model.tokenizer
# Ensure padding token is set (Qwen sometimes lacks a default pad token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset & Loader
dataset = HumanML3DDataset(DATA_ROOT, SPLIT_FILE, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Optimization
optimizer = optim.AdamW(model.parameters(), lr=LR)

# checkpoints directory
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# --- Training Loop ---
print(f"Starting training on {DEVICE}...")

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(DEVICE)
        motion = batch['motion'].to(DEVICE)
        motion_mask = batch['motion_mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward calculates loss internally
        loss, predictions = model(input_ids, motion, motion_mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    torch.save(model.state_dict(), f"checkpoints/motion_qwen_epoch_{epoch+1}.pt")

print("Training complete.")