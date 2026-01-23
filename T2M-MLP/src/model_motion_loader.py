import torch
from tqdm import tqdm


class MotionQwenLoader:
    def __init__(self, model, gt_loader, device):
        self.model = model
        self.gt_loader = gt_loader
        self.device = device

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
