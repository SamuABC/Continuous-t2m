import os
import torch
import numpy as np
import codecs
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class HumanML3DDataset(Dataset):
    def __init__(self, data_root, split_file, tokenizer, max_motion_len=196):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_motion_len = max_motion_len
        self.mean = np.load(os.path.join(data_root, "Mean.npy"))
        self.std = np.load(os.path.join(data_root, "Std.npy"))

        # Load ID list
        with open(split_file, "r") as f:
            self.id_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        name = self.id_list[idx]

        # 1. Load Motion (numpy array: seq_len, dim)
        motion_path = os.path.join(self.data_root, "new_joint_vecs", name + ".npy")
        motion = np.load(motion_path)
        motion = torch.tensor(motion, dtype=torch.float32)
        # Normalize
        motion = (motion - self.mean) / self.std

        # Crop if too long
        if motion.shape[0] > self.max_motion_len:
            motion = motion[: self.max_motion_len]

        # 2. Load Text
        text_path = os.path.join(self.data_root, "texts", name + ".txt")
        with codecs.open(text_path, "r", encoding="utf-8") as f:
            # HumanML3D often has multiple descriptions; pick the first one or random
            texts = [line.strip().split("#")[0] for line in f.readlines()]
            text = texts[0]

        # 3. Tokenize Text
        # Note: We don't add special tokens here, the model handles structure
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=32,  # Keep text short for efficiency
        )

        return {
            "motion": torch.tensor(motion, dtype=torch.float32),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    # Separate lists
    motions = [item["motion"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]

    # 1. Pad Motion
    # returns (Batch, Max_Seq_Len, Dim)
    motions_padded = pad_sequence(motions, batch_first=True, padding_value=0.0)

    # Create Motion Mask (1 for real data, 0 for padding)
    motion_lengths = torch.tensor([m.shape[0] for m in motions])
    B, T, D = motions_padded.shape
    motion_mask = torch.zeros((B, T), dtype=torch.bool)
    for i, length in enumerate(motion_lengths):
        motion_mask[i, :length] = 1

    # 2. Pad Text
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )  # 0 is usually pad for Qwen, check tokenizer

    return {
        "motion": motions_padded,
        "motion_mask": motion_mask,
        "input_ids": input_ids_padded,
    }
