import config as cfg
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class MotionQwen(nn.Module):
    def __init__(self, base_model_id, motion_dim, hidden_dim=1024, r=8, lora_alpha=32):
        super().__init__()

        # load Qwen Backbone & Tokenizer
        self.qwen = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        self.qwen = get_peft_model(self.qwen, peft_config)

        # motion Adapter (MLP)
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.motion_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motion_dim),
        )

        # define start motion token
        self.start_motion_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, input_ids, motion, motion_mask, teacher_forcing_ratio=1.0):
        # standard training ( 1 pass)
        if teacher_forcing_ratio == 1.0 or not self.training:
            return self._run_forward_pass(input_ids, motion, motion_mask)

        # Scheduled Sampling (2 passes)
        # get predicted motion without gradient tracking
        with torch.no_grad():
            _, predicted_motion = self._run_forward_pass(input_ids, motion, motion_mask)

        # mixing
        B, T, D = motion.shape
        mixing_mask = torch.bernoulli(
            torch.full((B, T, 1), teacher_forcing_ratio, device=motion.device)
        )

        mixed_motion = mixing_mask * motion + (1 - mixing_mask) * predicted_motion

        return self._run_forward_pass(input_ids, mixed_motion, motion_mask)

    def _run_forward_pass(self, input_ids, motion, motion_mask):
        """
        Helper method containing the core logic to avoid code duplication.
        """
        device = input_ids.device
        B, T, D = motion.shape

        # embeddings
        transformer = self.qwen.base_model.model.model
        text_embeds = transformer.embed_tokens(input_ids)
        motion_embeds = self.motion_encoder(motion)

        start_token = self.start_motion_token.expand(B, -1, -1)
        # shift inputs
        motion_input = motion_embeds[:, :-1, :]

        inputs_embeds = torch.cat([text_embeds, start_token, motion_input], dim=1)

        # masks
        text_mask = input_ids != self.tokenizer.pad_token_id
        text_mask_float = text_mask.to(dtype=inputs_embeds.dtype)
        start_mask_float = torch.ones((B, 1), device=device, dtype=inputs_embeds.dtype)
        motion_mask_in_float = motion_mask[:, :-1].to(dtype=inputs_embeds.dtype)

        mask_2d = torch.cat(
            [text_mask_float, start_mask_float, motion_mask_in_float], dim=1
        )
        seq_len = mask_2d.shape[1]

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(
            1, 1, seq_len, seq_len
        )
        padding_mask = mask_2d.view(B, 1, 1, seq_len)
        combined_mask = causal_mask * padding_mask

        min_dtype = torch.finfo(inputs_embeds.dtype).min
        inverted_mask = (1.0 - combined_mask) * min_dtype

        # position ids
        text_positions = (text_mask.cumsum(dim=1) - 1).clamp(min=0)
        text_lengths = text_mask.sum(dim=1, keepdim=True)
        start_position = text_lengths
        motion_range = torch.arange(
            1, motion_input.shape[1] + 1, device=device
        ).unsqueeze(0)
        motion_positions = start_position + motion_range
        position_ids = torch.cat(
            [text_positions, start_position, motion_positions], dim=1
        ).long()

        # forward
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=inverted_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        text_len = text_embeds.shape[1]
        motion_hidden_out = last_hidden_state[:, text_len:, :]

        predicted_motion = self.motion_decoder(motion_hidden_out)

        # motion loss (MSE)
        loss_fn = nn.MSELoss(reduction="none")
        loss_unreduced = loss_fn(predicted_motion, motion)
        loss_per_frame = loss_unreduced.mean(dim=-1)
        loss = (loss_per_frame * motion_mask).sum() / motion_mask.sum()

        return loss, predicted_motion

    @torch.no_grad()
    def generate(self, text, max_new_tokens=200, min_new_tokens=10):
        self.eval()
        device = self.qwen.device

        # prepare text inputs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # get text embeddings
        transformer = self.qwen.base_model.model.model
        text_embeds = transformer.embed_tokens(input_ids)

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
            outputs = self.qwen(
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
            current_inputs_embeds = self.motion_encoder(pred_motion)

            # add 1 column to attention mask for the new frame
            new_mask_col = torch.ones((B, 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

            # update position ids (increment the last pos by 1)
            position_ids = position_ids[:, -1:] + 1

        full_motion = torch.cat(generated_frames, dim=1)
        return full_motion
