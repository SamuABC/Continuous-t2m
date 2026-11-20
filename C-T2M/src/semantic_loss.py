import os

import config as cfg
import numpy as np
import torch
import torch.nn as nn
from guoevaluation.modules import MotionEncoderBiGRUCo, MovementConvEncoder


class SemanticMotionLoss(nn.Module):
    def __init__(self, device, dataset_name="t2m"):
        super().__init__()
        self.device = device

        dim_pose = 263 if dataset_name == "t2m" else 251
        dim_movement_enc_hidden = 512
        dim_movement_latent = 512
        dim_motion_hidden = 1024
        dim_coemb_hidden = 512

        self.movement_encoder = MovementConvEncoder(
            dim_pose - 4, dim_movement_enc_hidden, dim_movement_latent
        )

        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=dim_movement_latent,
            hidden_size=dim_motion_hidden,
            output_size=dim_coemb_hidden,
            device=device,
        )

        checkpoint_path = "./checkpoints/t2m/text_mot_match/model/finest.tar"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.movement_encoder.load_state_dict(checkpoint["movement_encoder"])
        self.motion_encoder.load_state_dict(checkpoint["motion_encoder"])

        # Load normalization stats for conversion from model space -> evaluator space
        model_mean = torch.from_numpy(np.load(os.path.join(cfg.DATA_ROOT, "Mean.npy"))).float()
        model_std = torch.from_numpy(np.load(os.path.join(cfg.DATA_ROOT, "Std.npy"))).float() + 1e-8

        meta_dir = "./checkpoints/t2m/text_mot_match/meta"
        if not os.path.exists(os.path.join(meta_dir, "mean.npy")):
            meta_dir = "./checkpoints/t2m/Comp_v6_KLD01/meta"
        eval_mean = torch.from_numpy(np.load(os.path.join(meta_dir, "mean.npy"))).float()
        eval_std = torch.from_numpy(np.load(os.path.join(meta_dir, "std.npy"))).float() + 1e-8

        self.register_buffer("model_mean", model_mean)
        self.register_buffer("model_std", model_std)
        self.register_buffer("eval_mean", eval_mean)
        self.register_buffer("eval_std", eval_std)

        for param in self.movement_encoder.parameters():
            param.requires_grad = False
        for param in self.motion_encoder.parameters():
            param.requires_grad = False

        def disable_inplace(module):
            if isinstance(module, (nn.LeakyReLU, nn.ReLU, nn.Dropout)):
                module.inplace = False

        self.movement_encoder.apply(disable_inplace)
        self.motion_encoder.apply(disable_inplace)

        self.movement_encoder.eval()
        self.motion_encoder.eval()

        self.movement_encoder.to(device, dtype=torch.float32)
        self.motion_encoder.to(device, dtype=torch.float32)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep MovementConvEncoder in eval() so nn.Dropout stays disabled
        self.movement_encoder.eval()
        # MotionEncoderBiGRUCo has no Dropout/BatchNorm; cuDNN requires GRU in train() for backward()
        self.motion_encoder.train(mode)
        return self

    def forward(self, motions, m_lens):
        """
        input: motions (B, T, D)
        output: features (B, D_emb)
        """
        # Convert from model normalization to evaluator normalization
        motions = (motions * self.model_std + self.model_mean - self.eval_mean) / self.eval_std

        # Sort batch by length (descending) required for pack_padded_sequence
        m_lens_sorted, sort_idx = m_lens.sort(descending=True)
        motions_sorted = motions[sort_idx]

        # Slice Foot Contacts
        motions_sliced = motions_sorted[..., :-4]

        # Scale lengths
        m_lens_scaled = (m_lens_sorted // 4).clamp(min=1)

        # Encoder Forward
        movements = self.movement_encoder(motions_sliced)
        motion_embedding = self.motion_encoder(movements, m_lens_scaled)

        # Restore original order
        unsort_idx = torch.argsort(sort_idx)
        motion_embedding = motion_embedding[unsort_idx]

        return motion_embedding