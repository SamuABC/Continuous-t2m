import codecs
import os

import config as cfg
import numpy as np
import torch
from torch.utils.data import Dataset


class HumanML3DDataset(Dataset):
    def __init__(self, data_root, split_file, tokenizer, max_motion_len=196):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.max_motion_len = max_motion_len
        self.mean = np.load(os.path.join(data_root, "Mean.npy"))
        self.std = (
            np.load(os.path.join(data_root, "Std.npy")) + 1e-8
        )  # add epsilon to avoid division by zero

        # load all sample ids
        with open(split_file, "r") as f:
            self.id_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        name = self.id_list[idx]

        # load motion
        motion_path = os.path.join(self.data_root, "new_joint_vecs", name + ".npy")
        motion = np.load(motion_path)
        motion = torch.tensor(motion, dtype=torch.float32)  # (Seq_Len, Dim)
        motion = (motion - self.mean) / self.std  # normalize

        # crop motion if too long
        if motion.shape[0] > self.max_motion_len:
            motion = motion[: self.max_motion_len]

        # load text
        text_path = os.path.join(self.data_root, "texts", name + ".txt")
        with codecs.open(text_path, "r", encoding="utf-8") as f:
            # HumanML3D often has multiple descriptions. Take the first one.
            texts = [line.strip().split("#")[0] for line in f.readlines()]
            text = texts[0]

        # adjust prompt
        text = cfg.PROMPT + text + cfg.PROMPT_END

        # tokenize text
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
        )

        return {
            "motion": motion.float(),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }

    def collate_fn(self, batch):
        motions = [item["motion"] for item in batch]
        text_features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in batch
        ]

        # motion padding
        motions_padded = torch.nn.utils.rnn.pad_sequence(
            motions, batch_first=True, padding_value=0.0
        )

        # create binary motion mask
        B, T, D = motions_padded.shape
        motion_mask = torch.zeros(
            (B, T), dtype=torch.long
        )  # Use long/int for masks generally
        for i, m in enumerate(motions):
            motion_mask[i, : m.shape[0]] = 1

        # pad text
        text_batch = self.tokenizer.pad(
            text_features, padding=True, return_tensors="pt"
        )

        return {
            "motion": motions_padded,
            "motion_mask": motion_mask,  # (B, Motion_Len)
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],  # (B, Text_Len)
        }
