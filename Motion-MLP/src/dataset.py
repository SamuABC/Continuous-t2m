import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/HumanML3D"
VEC_DIR = os.path.join(DATA_DIR, "new_joint_vecs")

mean = np.load(os.path.join(DATA_DIR, "Mean.npy"))
std  = np.load(os.path.join(DATA_DIR, "Std.npy"))
dimension = 263

class HumanML3DAutoRegDataset(Dataset):
    def __init__(self, list_file, seq_len=30):
        self.seq_len = seq_len
        with open(list_file, "r") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        vec_path = os.path.join(VEC_DIR, sid + ".npy")
        data = np.load(vec_path)  # shape: (T, D)

        # Normalize (if not already done)
        data = (data - self.mean) / (self.std + 1e-8)

        # ensure correct length
        T = data.shape[0]
        if T <= self.seq_len + 1:
            # repeat pad
            pad = np.repeat(data[-1:], self.seq_len + 1 - T, axis=0)
            data = np.concatenate([data, pad], axis=0)

        # we take the first seq_len + 1 frames
        data = data[:self.seq_len + 1]  # (seq_len+1, D)

        # autoregressive: x_t -> x_{t+1}
        x = data[:-1]  # (seq_len, D)
        y = data[1:]   # (seq_len, D)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def get_loaders(batch_size=64):
    # training
    train_dataset = HumanML3DAutoRegDataset(os.path.join(DATA_DIR, "train.txt"), seq_len=30)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # validation
    val_dataset = HumanML3DAutoRegDataset(os.path.join(DATA_DIR, "val.txt"), seq_len=30)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset, val_dataset