import codecs
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class HumanML3DDataset(Dataset):
    def __init__(self, data_root, split_file, tokenizer, max_motion_len=196):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_motion_len = max_motion_len
        self.mean = np.load(os.path.join(data_root, "Mean.npy"))
        self.std = np.load(os.path.join(data_root, "Std.npy"))

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
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        # pad motion
        # returns (Batch, Max_Seq_Len, Dim)
        motions_padded = torch.nn.utils.rnn.pad_sequence(
            motions, batch_first=True, padding_value=0.0
        )

        # motion mask
        B, T, D = motions_padded.shape
        motion_mask = torch.zeros((B, T), dtype=torch.float32)

        motion_lengths = [m.shape[0] for m in motions]
        for i, length in enumerate(motion_lengths):
            motion_mask[i, :length] = 1.0

        # pad text input ids (left padding)
        if self.tokenizer.pad_token_id is None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = self.tokenizer.pad_token_id

        input_ids_flipped = [t.flip(0) for t in input_ids]
        input_ids_padded_flipped = torch.nn.utils.rnn.pad_sequence(
            input_ids_flipped, batch_first=True, padding_value=pad_id
        )
        input_ids_padded = input_ids_padded_flipped.flip(1)

        # pad text attention masks (left padding)
        attention_masks_flipped = [t.flip(0) for t in attention_masks]
        attention_masks_padded_flipped = torch.nn.utils.rnn.pad_sequence(
            attention_masks_flipped, batch_first=True, padding_value=0
        )
        attention_masks_padded = attention_masks_padded_flipped.flip(1)

        return {
            "motion": motions_padded,
            "motion_mask": motion_mask,
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
        }
