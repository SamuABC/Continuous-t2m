import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionAutoEncoder(nn.Module):
    def __init__(self, motion_dim, hidden_dim):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.motion_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motion_dim),
        )

    def forward(self, x):
        z = self.motion_encoder(x)
        recon = self.motion_decoder(z)
        return recon, z, F.normalize(z, dim=-1)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        # Labels: i -> i + batch_size
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim.masked_fill_(mask, -9e15)

        return self.criterion(sim, labels)
