import config as cfg
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from semantic_loss import SemanticMotionLoss


class MotionModelCont(nn.Module):
    def __init__(self, base_model_id, motion_dim, r=32, lora_alpha=64):
        super().__init__()

        config = AutoConfig.from_pretrained(base_model_id)
        hidden_dim = config.hidden_size

        # load Backbone & Tokenizer
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        mean = torch.from_numpy(np.load(cfg.DATA_ROOT + "/Mean.npy")).float()
        std = torch.from_numpy(np.load(cfg.DATA_ROOT + "/Std.npy")).float() + 1e-8

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # add a new pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backbone.config.pad_token_id = self.tokenizer.pad_token_id

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
        self.backbone = get_peft_model(self.backbone, peft_config)

        # initialize semantic loss module
        if cfg.LAMBDA_SEMANTIC > 0.0:
            self.semantic_loss_module = SemanticMotionLoss(
                device=self.backbone.device, dataset_name="t2m"
            )

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

        # cast to bfloat16 to fit to backbone's dtype
        self.motion_encoder = self.motion_encoder.to(torch.bfloat16)
        self.motion_decoder = self.motion_decoder.to(torch.bfloat16)
        self.start_motion_token.data = self.start_motion_token.data.to(torch.bfloat16)

    def forward(self, input_ids, motion, motion_mask, teacher_forcing_ratio=1.0):
        # --- Conditional Dropout ---
        current_input_ids = input_ids.clone()

        motion = motion.to(torch.bfloat16)

        if self.training and cfg.COND_DROPOUT_RATE > 0 and cfg.USE_CFG:
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
            _, _, predicted_motion = self._run_forward_pass(
                current_input_ids, motion, motion_mask
            )

        # mixing
        B, T, D = motion.shape
        mixing_mask = torch.bernoulli(
            torch.full((B, T, 1), teacher_forcing_ratio, device=motion.device)
        ).to(torch.bfloat16)

        mixed_motion = mixing_mask * motion + (1 - mixing_mask) * predicted_motion

        return self._run_forward_pass(current_input_ids, mixed_motion, motion_mask)

    def _run_forward_pass(self, input_ids, motion, motion_mask):
        device = input_ids.device
        B, T, D = motion.shape

        # embeddings
        base_model = self.backbone.get_base_model()
        input_embedding_layer = base_model.get_input_embeddings()

        text_embeds = input_embedding_layer(input_ids)
        motion_embeds = self.motion_encoder(motion)

        start_token = self.start_motion_token.expand(B, -1, -1)
        # shift motion input
        motion_input = motion_embeds[:, :-1, :]

        inputs_embeds = torch.cat([text_embeds, start_token, motion_input], dim=1)

        # binary 2D attention mask
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
        outputs = self.backbone(
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

        # position loss (MSE)
        loss_fn = nn.MSELoss(reduction="none")
        loss_unreduced = loss_fn(predicted_motion, motion)
        loss_per_frame = loss_unreduced.mean(dim=-1)
        loss_pos = (loss_per_frame * motion_mask).sum() / (motion_mask.sum() + 1e-8)

        # velocity loss (MSE), calculate on frame differences
        target_vel = motion[:, 1:] - motion[:, :-1]
        pred_vel = predicted_motion[:, 1:] - predicted_motion[:, :-1]

        # adjust mask for velocity (one frame shorter)
        vel_mask = motion_mask[:, 1:]

        loss_vel_unreduced = loss_fn(pred_vel, target_vel)
        loss_vel_per_frame = loss_vel_unreduced.mean(dim=-1)
        loss_vel = (loss_vel_per_frame * vel_mask).sum() / (vel_mask.sum() + 1e-8)

        # --- Foot Contact Consistency Loss ---
        # Indices based on HumanML3D:
        #   Positions: 157:163 (L_Foot 157:160, R_Foot 160:163)
        #   Contacts:  259:263
        mean_pos_l = self.mean[..., 157:160]
        std_pos_l = self.std[..., 157:160]
        mean_pos_r = self.mean[..., 160:163]
        std_pos_r = self.std[..., 160:163]

        mean_cont = self.mean[..., 259:263]
        std_cont = self.std[..., 259:263]

        # 2. Foot Skate Loss
        # Derive velocity from positions
        pred_pos_l_norm = predicted_motion[:, :, 157:160]
        pred_pos_r_norm = predicted_motion[:, :, 160:163]

        pred_pos_l_phys = pred_pos_l_norm * std_pos_l + mean_pos_l
        pred_pos_r_phys = pred_pos_r_norm * std_pos_r + mean_pos_r

        # Calculate velocity: v[t] = p[t] - p[t-1]
        derived_vel_l = pred_pos_l_phys[:, 1:] - pred_pos_l_phys[:, :-1]
        derived_vel_r = pred_pos_r_phys[:, 1:] - pred_pos_r_phys[:, :-1]

        derived_vel_l_mag = torch.norm(derived_vel_l, dim=-1)
        derived_vel_r_mag = torch.norm(derived_vel_r, dim=-1)

        # 3. Foot Contacts
        pred_contact_norm = predicted_motion[:, :, 259:263]
        pred_contact_phys = pred_contact_norm * std_cont + mean_cont
        scale_factor = 10.0
        contact_logits = (pred_contact_phys - 0.5) * scale_factor
        pred_contact_prob = torch.sigmoid(contact_logits)

        # Align contacts with derived velocity (remove first frame)
        # Max over Heel/Toe for Left (0:2) and Right (2:4)
        pred_contact_l = torch.max(pred_contact_prob[:, 1:, 0:2], dim=-1).values
        pred_contact_r = torch.max(pred_contact_prob[:, 1:, 2:4], dim=-1).values

        fc_loss_l = (derived_vel_l_mag * pred_contact_l * vel_mask).sum()
        fc_loss_r = (derived_vel_r_mag * pred_contact_r * vel_mask).sum()

        loss_fc = (fc_loss_l + fc_loss_r) / (vel_mask.sum() + 1e-8)

        # 4. Foot Contact Classification Loss
        # Create binary GT
        gt_contacts_norm = motion[:, :, 259:263]
        gt_contacts_phys = gt_contacts_norm * std_cont + mean_cont
        gt_contacts_binary = (gt_contacts_phys > 0.5).float()

        loss_contact_fn = nn.BCEWithLogitsLoss(reduction="none")
        loss_contact_unreduced = loss_contact_fn(contact_logits, gt_contacts_binary)

        loss_contact_per_frame = loss_contact_unreduced.mean(dim=-1)

        # Apply mask
        loss_contact_cls = (loss_contact_per_frame * motion_mask).sum() / (
            motion_mask.sum() + 1e-8
        )

        # --- Semantic Loss ---
        with torch.autocast(device_type=device.type, enabled=False):
            if cfg.LAMBDA_SEMANTIC > 0.0:
                # cast inputs to float32 for semantic loss module
                motion_f32 = motion.to(dtype=torch.float32)
                predicted_motion_f32 = predicted_motion.to(dtype=torch.float32)

                # Calculate lengths from mask
                m_lens = motion_mask.sum(dim=1).long()

                # Extract features for Ground Truth
                with torch.no_grad():
                    gt_features = self.semantic_loss_module(motion_f32, m_lens)

                # Extract features for Predicted Motion
                pred_features = self.semantic_loss_module(predicted_motion_f32, m_lens)

                # Calculate MSE between feature vectors
                loss_semantic = nn.MSELoss()(pred_features, gt_features)
            else:
                loss_semantic = torch.tensor(0.0, device=device)

        # --- Total Motion Loss ---
        loss_motion = (
            (cfg.LAMBDA_POS * loss_pos)
            + (cfg.LAMBDA_VEL * loss_vel)
            + (cfg.LAMBDA_FOOT_SKATE * loss_fc)
            + (cfg.LAMBDA_FOOT_CONTACT * loss_contact_cls)
            + (cfg.LAMBDA_SEMANTIC * loss_semantic)
        )

        if cfg.LAMBDA_LANG > 0.0:
            # --- Language loss ---
            logits = outputs.logits
            text_logits = logits[:, : text_len - 1, :].contiguous()
            text_labels = input_ids[:, 1:].contiguous()
            vocab_size = text_logits.shape[-1]
            loss_lang_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

            loss_lang = loss_lang_fn(
                text_logits.view(-1, vocab_size), text_labels.view(-1)
            )
            total_loss = loss_motion + (cfg.LAMBDA_LANG * loss_lang)
        else:
            # skip language loss computation if weight is zero
            loss_lang = torch.tensor(-1.0, device=device)
            total_loss = loss_motion

        loss_logs = {
            "loss": total_loss.item(),  # note: if renamed, also change in train.py
            "pos": loss_pos.item(),
            "vel": loss_vel.item(),
            "foot skate": loss_fc.item(),
            "contact": loss_contact_cls.item(),
            "lang": loss_lang.item(),
            "semantic": loss_semantic.item(),
        }

        return loss_logs, total_loss, predicted_motion

    @torch.no_grad()
    def generate_without_cfg(self, text, max_new_tokens=196):
        self.eval()
        device = self.backbone.device

        # prepare text inputs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # get text embeddings
        base_model = self.backbone.get_base_model()
        input_embedding_layer = base_model.get_input_embeddings()
        text_embeds = input_embedding_layer(input_ids)

        B, L, _ = text_embeds.shape
        start_token = self.start_motion_token.expand(B, -1, -1)

        # concat: [text, start_token]
        current_inputs_embeds = torch.cat([text_embeds, start_token], dim=1)

        # update attention mask for start token (add 1s to the right)
        start_mask = torch.ones((B, 1), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, start_mask], dim=1)

        # get position ids of valid tokens (0-based indexing)
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        past_key_values = DynamicCache()
        generated_frames = []

        # generation loop
        for i in range(max_new_tokens):
            outputs = self.backbone(
                inputs_embeds=current_inputs_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=True,
            )

            past_key_values = outputs.past_key_values

            last_hidden_state = outputs.hidden_states[-1][:, -1:, :]

            pred_motion = self.motion_decoder(last_hidden_state)

            generated_frames.append(pred_motion)

            # encode output to latent space for next step
            pred_motion = pred_motion.to(torch.bfloat16)
            current_inputs_embeds = self.motion_encoder(pred_motion)

            # add 1 column to attention mask for the new frame
            new_mask_col = torch.ones((B, 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

            # update position ids (increment the last pos by 1)
            position_ids = position_ids[:, -1:] + 1

        full_motion = torch.cat(generated_frames, dim=1)
        return full_motion.float()

    @torch.no_grad()
    def generate_with_cfg(self, text, max_new_tokens=196):
        self.eval()
        device = self.backbone.device

        # prepare text inputs
        cond_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(device)
        cond_input_ids = cond_inputs.input_ids
        cond_mask = cond_inputs.attention_mask

        # prepare uncond inputs (all pad tokens)
        uncond_input_ids = torch.full_like(cond_input_ids, self.tokenizer.pad_token_id)
        uncond_mask = (uncond_input_ids != self.tokenizer.pad_token_id).long()

        # get embeddings for both cond and uncond
        B = cond_input_ids.shape[0]

        input_embedding_layer = self.backbone.get_input_embeddings()

        start_token = self.start_motion_token.expand(B, -1, -1)

        cond_embeds = torch.cat(
            [input_embedding_layer(cond_input_ids), start_token], dim=1
        )
        uncond_embeds = torch.cat(
            [input_embedding_layer(uncond_input_ids), start_token], dim=1
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
            out_cond = self.backbone(
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
            out_uncond = self.backbone(
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
            pred_motion = pred_motion.to(torch.bfloat16)
            cond_embeds = self.motion_encoder(pred_motion)
            uncond_embeds = self.motion_encoder(pred_motion)

            # update attention masks (+1 column of ones)
            new_mask = torch.ones((B, 1), device=device, dtype=torch.long)
            cond_att_mask = torch.cat([cond_att_mask, new_mask], dim=1)
            uncond_att_mask = torch.cat([uncond_att_mask, new_mask], dim=1)

            # update position ids (+1)
            cond_pos_ids = cond_pos_ids[:, -1:] + 1
            uncond_pos_ids = uncond_pos_ids[:, -1:] + 1

        return torch.cat(generated_frames, dim=1).float()

    @torch.no_grad()
    def generate(self, text, max_new_tokens=196):
        if cfg.USE_CFG:
            return self.generate_with_cfg(text, max_new_tokens)
        else:
            return self.generate_without_cfg(text, max_new_tokens)
