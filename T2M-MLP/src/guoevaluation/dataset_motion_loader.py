from os.path import join as pjoin

import numpy as np
import torch
from guoevaluation.dataset import Text2MotionDatasetV2, collate_fn
from guoevaluation.get_opt import get_opt
from guoevaluation.word_vectorizer import WordVectorizer
from torch.utils.data import DataLoader


def get_dataset_motion_loader(opt_path, batch_size, device, _split_file="test.txt"):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == "t2m" or opt.dataset_name == "kit":
        print("Loading dataset %s ..." % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
        std = np.load(pjoin(opt.meta_dir, "std.npy"))

        w_vectorizer = WordVectorizer("./glove", "our_vab")
        split_file = pjoin(opt.data_root, _split_file)
        dataset = Text2MotionDatasetV2(opt, mean, std, split_file, w_vectorizer)
        if _split_file == "train.txt":
            # cut the training set for quick testing of overfitting
            dataset = torch.utils.data.Subset(dataset, list(range(1460)))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=True,
        )
    else:
        raise KeyError("Dataset not Recognized !!")

    print("Ground Truth Dataset Loading Completed!!!")
    return dataloader, dataset
