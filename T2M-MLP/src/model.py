import config as cfg
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class MotionQwen(nn.Module):
    def __init__(self, base_model_id, motion_dim, hidden_dim=1024, r=32, lora_alpha=64):
        super().__init__()

        # load Qwen Backbone & Tokenizer
        self.qwen = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # add a new pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            # Important: Resize token embeddings because vocab size changed
            self.qwen.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.padding_side = "left"

        # configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules="all-linear",
        )
        self.qwen = get_peft_model(self.qwen, peft_config)

        # motion Adapter (MLP)
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
            nn.Linear(hidden_dim, motion_dim),
            nn.GELU(),
            nn.Linear(motion_dim, motion_dim),
        )

        # define start motion token
        self.start_motion_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, input_ids, motion, motion_mask, teacher_forcing_ratio=1.0):
        # --- Conditional Dropout ---
        current_input_ids = input_ids.clone()

        if self.training and cfg.COND_DROPOUT_RATE > 0:
            # Mask: 1 = keep, 0 = drop
            keep_prob = 1.0 - cfg.COND_DROPOUT_RATE
            B = input_ids.shape[0]
            keep_mask = torch.bernoulli(
                torch.full((B, 1), keep_prob, device=input_ids.device)
            ).long()

            # Replace text with pad_token_id where mask is 0
            pad_token_id = self.tokenizer.pad_token_id
            current_input_ids = input_ids * keep_mask + pad_token_id * (1 - keep_mask)

        # --- Standard training --- ( 1 pass)
        if teacher_forcing_ratio == 1.0 or not self.training:
            return self._run_forward_pass(current_input_ids, motion, motion_mask)

        # --- Scheduled Sampling --- (2 passes)
        # get predicted motion without gradient tracking
        with torch.no_grad():
            _, _, _, predicted_motion = self._run_forward_pass(
                current_input_ids, motion, motion_mask
            )

        # mixing
        B, T, D = motion.shape
        mixing_mask = torch.bernoulli(
            torch.full((B, T, 1), teacher_forcing_ratio, device=motion.device)
        )

        mixed_motion = mixing_mask * motion + (1 - mixing_mask) * predicted_motion

        return self._run_forward_pass(current_input_ids, mixed_motion, motion_mask)

    def _run_forward_pass(self, input_ids, motion, motion_mask):
        device = input_ids.device
        B, T, D = motion.shape

        # embeddings
        transformer = self.qwen.base_model.model.model
        text_embeds = transformer.embed_tokens(input_ids)
        motion_embeds = self.motion_encoder(motion)

        start_token = self.start_motion_token.expand(B, -1, -1)
        # shift motion input
        motion_input = motion_embeds[:, :-1, :]

        inputs_embeds = torch.cat([text_embeds, start_token, motion_input], dim=1)

        # binary 2D attention mask
        # Qwen automatically applies the Causal Lower-Triangular Mask on top of this.
        text_mask = (input_ids != self.tokenizer.pad_token_id).long()
        start_mask = torch.ones((B, 1), device=device, dtype=torch.long)
        motion_mask_in = motion_mask[:, :-1].long()

        attention_mask = torch.cat([text_mask, start_mask, motion_mask_in], dim=1)

        # position IDs (needed because of left padding in text)
        text_mask_bool = text_mask.bool()
        text_positions = (text_mask_bool.cumsum(dim=1) - 1).clamp(min=0)

        # motion starts after the last text token
        last_text_pos = text_positions.max(dim=1, keepdim=True).values
        start_pos = last_text_pos + 1

        motion_range = torch.arange(
            1, motion_input.shape[1] + 1, device=device
        ).unsqueeze(0)
        motion_positions = start_pos + motion_range

        position_ids = torch.cat(
            [text_positions, start_pos, motion_positions], dim=1
        ).long()

        # forward
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        # --- Motion loss ---
        # extract motion hidden states & decode
        last_hidden_state = outputs.hidden_states[-1]
        text_len = text_embeds.shape[1]
        motion_hidden_out = last_hidden_state[:, text_len:, :]
        predicted_motion = self.motion_decoder(motion_hidden_out)

        # motion loss (MSE)
        loss_fn = nn.MSELoss(reduction="none")
        loss_unreduced = loss_fn(predicted_motion, motion)
        loss_per_frame = loss_unreduced.mean(dim=-1)
        loss_motion = (loss_per_frame * motion_mask).sum() / motion_mask.sum()

        if cfg.LAMBDA_LANG == 0.0:
            # skip language loss computation if weight is zero
            total_loss = loss_motion
            loss_lang = torch.tensor(-1.0, device=device)
            return loss_motion, loss_lang, total_loss, predicted_motion

        # --- Language loss ---
        logits = outputs.logits
        text_logits = logits[:, : text_len - 1, :].contiguous()
        text_labels = input_ids[:, 1:].contiguous()
        vocab_size = text_logits.shape[-1]

        loss_lang_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss_lang = loss_lang_fn(text_logits.view(-1, vocab_size), text_labels.view(-1))

        # --- Total loss ---
        total_loss = loss_motion + (cfg.LAMBDA_LANG * loss_lang)

        return loss_motion, loss_lang, total_loss, predicted_motion

    @torch.no_grad()
    def generate(self, text, max_new_tokens=196, min_new_tokens=10):
        self.eval()
        device = self.qwen.device

        # prepare text inputs
        cond_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(device)
        cond_input_ids = cond_inputs.input_ids
        cond_mask = cond_inputs.attention_mask

        # prepare uncond inputs (all pad tokens)
        uncond_input_ids = torch.full_like(cond_input_ids, self.tokenizer.pad_token_id)
        uncond_mask = (uncond_input_ids != self.tokenizer.pad_token_id).long()

        # get embeddings for both cond and uncond
        transformer = self.qwen.base_model.model.model
        B = cond_input_ids.shape[0]
        start_token = self.start_motion_token.expand(B, -1, -1)

        cond_embeds = torch.cat(
            [transformer.embed_tokens(cond_input_ids), start_token], dim=1
        )
        uncond_embeds = torch.cat(
            [transformer.embed_tokens(uncond_input_ids), start_token], dim=1
        )

        # extend attention masks with start token
        start_mask = torch.ones((B, 1), device=device, dtype=torch.long)
        cond_att_mask = torch.cat([cond_mask, start_mask], dim=1)
        uncond_att_mask = torch.cat([uncond_mask, start_mask], dim=1)

        # position ids
        cond_pos_ids = (cond_att_mask.cumsum(dim=1) - 1).clamp(min=0)
        uncond_pos_ids = (uncond_att_mask.cumsum(dim=1) - 1).clamp(min=0)

        past_kv_cond = DynamicCache()
        past_kv_uncond = DynamicCache()
        generated_frames = []

        for i in range(max_new_tokens):
            # conditional pass
            out_cond = self.qwen(
                inputs_embeds=cond_embeds,
                past_key_values=past_kv_cond,
                attention_mask=cond_att_mask,
                position_ids=cond_pos_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv_cond = out_cond.past_key_values
            hidden_cond = out_cond.hidden_states[-1][:, -1:, :]
            motion_cond = self.motion_decoder(hidden_cond)

            # unconditional pass
            out_uncond = self.qwen(
                inputs_embeds=uncond_embeds,
                past_key_values=past_kv_uncond,
                attention_mask=uncond_att_mask,
                position_ids=uncond_pos_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv_uncond = out_uncond.past_key_values
            hidden_uncond = out_uncond.hidden_states[-1][:, -1:, :]
            motion_uncond = self.motion_decoder(hidden_uncond)

            # classifier-free guidance scaling
            pred_motion = motion_uncond + cfg.GUIDANCE_SCALE * (
                motion_cond - motion_uncond
            )
            generated_frames.append(pred_motion)

            # prepare embeddings for next iteration
            cond_embeds = self.motion_encoder(pred_motion)
            uncond_embeds = self.motion_encoder(pred_motion)

            # update attention masks (+1 column of ones)
            new_mask = torch.ones((B, 1), device=device, dtype=torch.long)
            cond_att_mask = torch.cat([cond_att_mask, new_mask], dim=1)
            uncond_att_mask = torch.cat([uncond_att_mask, new_mask], dim=1)

            # update position ids (+1)
            cond_pos_ids = cond_pos_ids[:, -1:] + 1
            uncond_pos_ids = uncond_pos_ids[:, -1:] + 1

        return torch.cat(generated_frames, dim=1)
