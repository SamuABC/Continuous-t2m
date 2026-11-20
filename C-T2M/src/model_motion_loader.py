import torch
from tqdm import tqdm


class MotionQwenLoader:
    def __init__(self, model, gt_loader, device):
        self.model = model
        self.gt_loader = gt_loader
        self.device = device

        eval_dataset = (
            gt_loader.dataset.dataset
            if hasattr(gt_loader.dataset, "dataset")
            else gt_loader.dataset
        )
        self.eval_mean = torch.from_numpy(eval_dataset.mean).float().to(device)
        self.eval_std = torch.from_numpy(eval_dataset.std).float().to(device)

    def __len__(self):
        return len(self.gt_loader)

    def __iter__(self):
        for batch in tqdm(self.gt_loader, desc="Generating Motions"):
            (
                word_embeddings,
                pos_one_hots,
                caption,
                sent_lens,
                motions,
                m_lens,
                tokens,
            ) = batch

            text_prompts = list(caption)

            current_batch_max_len = max(m_lens).item()

            # generate motions
            self.model.eval()
            with torch.no_grad():
                # note: for fair comparison, motions are generated with same length as GT
                generated_motions = self.model.generate(
                    text_prompts,
                    max_new_tokens=current_batch_max_len,
                )

                # convert from model normalization (Mean/Std) to evaluator normalization (Comp_v6_KLD01 meta)
                model_mean = self.model.mean.to(self.device)
                model_std = self.model.std.to(self.device)
                generated_motions = (generated_motions * model_std + model_mean - self.eval_mean) / (
                    self.eval_std + 1e-8
                )

                # zero-pad frames beyond actual motion length to match gt_loader format
                max_len = generated_motions.shape[1]
                valid_mask = (
                    torch.arange(max_len, device=self.device).unsqueeze(0)
                    < m_lens.to(self.device).unsqueeze(1)
                ).unsqueeze(-1)
                generated_motions = generated_motions * valid_mask

            # return the batch with generated motions
            yield (
                word_embeddings,
                pos_one_hots,
                caption,
                sent_lens,
                generated_motions,
                m_lens,
                tokens,
            )


def get_qwen_model_loader(model, gt_loader, device):
    """
    Factory function to be used in eval_motion_loaders dict
    """
    return MotionQwenLoader(model, gt_loader, device)