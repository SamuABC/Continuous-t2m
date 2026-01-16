import os
import warnings

import config as cfg
import numpy as np
import torch
from evaluation.evaluator_wrapper import EvaluatorModelWrapper
from evaluation.get_eval_option import get_opt
from evaluation.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
    calculate_R_precision,
)
from evaluation.word_vectorizer import WordVectorizer
from model import MotionQwen
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HumanML3DDataset

warnings.filterwarnings("ignore")


def prepare_text_for_evaluator(raw_texts, w_vectorizer, device, max_text_len=20):
    batch_size = len(raw_texts)
    dim_word = 300
    dim_pos = 15

    word_embeddings = torch.zeros((batch_size, max_text_len, dim_word), device=device)
    pos_one_hots = torch.zeros((batch_size, max_text_len, dim_pos), device=device)
    cap_lens = []

    for i, text in enumerate(raw_texts):
        text = text.replace(".", "").replace("!", "").replace("?", "").replace(",", "")
        words = text.split()

        if len(words) < max_text_len:
            cap_len = len(words)
        else:
            cap_len = max_text_len
            words = words[:max_text_len]

        cap_lens.append(cap_len)

        for w_i, word in enumerate(words):
            # remove '/' to not confuse the vectorizer
            if "/" in word:
                word = word.replace("/", "")
                if not word:
                    word = "unk"

            try:
                vec, pos_vec = w_vectorizer[f"{word}/OTHER"]
            except (KeyError, ValueError):
                # Fallback for unknown words
                vec, pos_vec = w_vectorizer["unk/OTHER"]

            word_embeddings[i, w_i] = torch.from_numpy(vec)
            pos_one_hots[i, w_i] = torch.from_numpy(pos_vec)

    return word_embeddings, pos_one_hots, torch.tensor(cap_lens, device=device)


def evaluation_qwen_loop(val_loader, model, eval_wrapper, w_vectorizer, device):
    model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    text_embed_list = []

    mean = torch.tensor(np.load(cfg.DATA_ROOT + "/Mean.npy")).to(device)
    std = torch.tensor(np.load(cfg.DATA_ROOT + "/Std.npy")).to(device)

    pbar = tqdm(val_loader, desc="Generating Motions")
    for batch in pbar:
        # --- 1. Prepare Data ---
        input_ids = batch["input_ids"].to(device)
        # decode text for the evaluator
        raw_texts = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        gt_motion = batch["motion"].to(device)
        gt_motion = gt_motion * std + mean  # denormalize

        # --- 2. Prediction ---
        with torch.no_grad():
            pred_motion = model.generate(raw_texts, max_new_tokens=196)

        # denormalize
        pred_motion = pred_motion * std + mean

        # get actual generated length
        curr_batch_size = pred_motion.shape[0]
        curr_seq_len = pred_motion.shape[1]

        # gt lengths for this batch
        gt_mask = batch["motion_mask"].to(device)
        gt_lengths = gt_mask.sum(1).long()

        # predicted lengths for this batch
        pred_lengths = torch.full(
            (curr_batch_size,), curr_seq_len, device=device, dtype=torch.long
        )

        # --- 3. Vectorize Text using helper function ---
        word_embs, pos_ohot, cap_lens = prepare_text_for_evaluator(
            raw_texts, w_vectorizer, device
        )

        # sort indices based on cap_lens (descending)
        sorted_cap_lens, sorted_idx = torch.sort(cap_lens, descending=True)

        # reorder all tensors accordingly
        word_embs = word_embs[sorted_idx]
        pos_ohot = pos_ohot[sorted_idx]

        # reorder motion and gt
        gt_motion = gt_motion[sorted_idx]
        gt_lengths = gt_lengths[sorted_idx]
        pred_motion = pred_motion[sorted_idx]
        pred_lengths = pred_lengths[sorted_idx]

        # --- 4. Extract Embeddings ---

        # gt embeddings
        _, gt_motion_emb = eval_wrapper.get_co_embeddings(
            word_embs, pos_ohot, sorted_cap_lens, gt_motion, gt_lengths
        )

        # pred embeddings
        pred_text_emb, pred_motion_emb = eval_wrapper.get_co_embeddings(
            word_embs, pos_ohot, sorted_cap_lens, pred_motion, pred_lengths
        )

        motion_annotation_list.append(gt_motion_emb)
        motion_pred_list.append(pred_motion_emb)
        text_embed_list.append(pred_text_emb)

    # --- 5. Calculate Metrics ---
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    text_embed_np = torch.cat(text_embed_list, dim=0).cpu().numpy()

    # FID
    mu_gt, sigma_gt = calculate_activation_statistics(motion_annotation_np)
    mu_pred, sigma_pred = calculate_activation_statistics(motion_pred_np)
    fid = calculate_frechet_distance(mu_gt, sigma_gt, mu_pred, sigma_pred)

    # Diversity
    div = calculate_diversity(motion_pred_np, diversity_times=300)

    # R-Precision
    raw_r_precision = calculate_R_precision(
        motion_pred_np, text_embed_np, top_k=3, sum_all=True
    )
    r_precision = raw_r_precision / len(motion_pred_np)

    # Matching Score
    raw_matching_score = calculate_matching_score(motion_pred_np, text_embed_np)
    matching_score = raw_matching_score.mean()

    top1, top2, top3 = r_precision[0], r_precision[1], r_precision[2]

    return fid, div, top1, top2, top3, matching_score, 0.0
    # (multimodality is 0 because the model currently is deterministic)


def main():
    # --- Config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "./output/eval_results"
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = cfg.INFERENCE_MODEL_PATH

    # --- Model Loading ---
    print(f"Loading MotionQwen from {ckpt_path}...")
    model = MotionQwen(base_model_id=cfg.BASE_MODEL_ID, motion_dim=cfg.MOTION_DIM).to(
        device
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Padding side must be left for generation
    model.tokenizer.padding_side = "left"

    # --- Evaluator Loading (Guo Wrapper) ---
    dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD01/opt.txt"

    glove_root = "./dataset/HumanML3D/glove"

    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # --- Word Vectorizer Loading ---
    print("Loading Word Vectorizer...")
    try:
        w_vectorizer = WordVectorizer(glove_root, "our_vab")
    except Exception as e:
        print(f"Error loading WordVectorizer from {glove_root}: {e}")
        print(
            "Please check where your 'our_vab_data.npy', 'our_vab_words.pkl' files are."
        )
        return

    # --- Dataloader ---
    test_dataset = HumanML3DDataset(
        cfg.DATA_ROOT, cfg.DATA_ROOT + "/test.txt", model.tokenizer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        drop_last=True,
    )

    # --- Evaluation Loop ---
    print("Starting Evaluation...")

    fid_list, div_list = [], []
    top1_list, top2_list, top3_list = [], [], []
    match_list, multi_list = [], []

    repeat_time = 1

    for i in range(repeat_time):
        print(f"Run {i+1}/{repeat_time}")

        fid, div, top1, top2, top3, matching, multi = evaluation_qwen_loop(
            test_loader, model, eval_wrapper, w_vectorizer, device
        )

        print(f"  --> FID: {fid:.3f} | Div: {div:.3f} | Top1: {top1:.3f}")

        fid_list.append(fid)
        div_list.append(div)
        top1_list.append(top1)
        top2_list.append(top2)
        top3_list.append(top3)
        match_list.append(matching)
        multi_list.append(multi)

    # --- Final Stats ---
    def get_stat(data):
        mean = np.mean(data)
        conf = np.std(data) * 1.96 / np.sqrt(len(data))
        return mean, conf

    f_mean, f_conf = get_stat(fid_list)
    d_mean, d_conf = get_stat(div_list)
    t1_mean, t1_conf = get_stat(top1_list)
    t2_mean, t2_conf = get_stat(top2_list)
    t3_mean, t3_conf = get_stat(top3_list)
    m_mean, m_conf = get_stat(match_list)

    msg_final = (
        f"Evaluation of model: {ckpt_path}\n"
        f"Final Results over {repeat_time} runs:\n"
        f"FID: {f_mean:.3f} ± {f_conf:.3f}\n"
        f"Diversity: {d_mean:.3f} ± {d_conf:.3f}\n"
        f"Top1: {t1_mean:.3f} ± {t1_conf:.3f}\n"
        f"Top2: {t2_mean:.3f} ± {t2_conf:.3f}\n"
        f"Top3: {t3_mean:.3f} ± {t3_conf:.3f}\n"
        f"Matching: {m_mean:.3f} ± {m_conf:.3f}"
    )

    print("==================================")
    print(msg_final)
    print("==================================")

    with open(os.path.join(out_dir, "eval_log.txt"), "w") as f:
        f.write(msg_final)


if __name__ == "__main__":
    main()
