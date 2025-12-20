import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import get_peft_model, LoraConfig, TaskType


class MotionQwen(nn.Module):
    def __init__(self, base_model_id, motion_dim, hidden_dim=1024, r=8, lora_alpha=32):
        super().__init__()

        # 1. Load Qwen Backbone & Tokenizer
        self.qwen = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # 2. Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        self.qwen = get_peft_model(self.qwen, peft_config)

        # 3. Motion Adapter (MLP)
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

        # 4. Special Tokens
        self.start_motion_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # End token is trained implicitly as the target of the last frame if needed,
        # or used during inference as a stopping condition.
        self.end_motion_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

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

        # learn with mixed motion
        loss, predicted_motion = self._run_forward_pass(
            input_ids, mixed_motion, motion_mask
        )

        return loss, predicted_motion

    def _run_forward_pass(self, input_ids, motion, motion_mask):
        """
        Helper method containing the core logic to avoid code duplication.
        """
        device = input_ids.device
        B, T, D = motion.shape

        # 1. Embeddings
        transformer = self.qwen.base_model.model.model
        text_embeds = transformer.embed_tokens(input_ids)
        motion_embeds = self.motion_encoder(motion)

        start_token = self.start_motion_token.expand(B, -1, -1)
        # Shift inputs: M_0 ... M_{T-2} predicts M_1 ... M_{T-1}
        motion_input = motion_embeds[:, :-1, :]

        inputs_embeds = torch.cat([text_embeds, start_token, motion_input], dim=1)

        # 2. Masks
        text_mask = (input_ids != self.tokenizer.pad_token_id).long()
        start_mask = torch.ones((B, 1), device=device)
        motion_mask_in = motion_mask[:, :-1]

        mask_2d = torch.cat([text_mask, start_mask, motion_mask_in], dim=1)
        seq_len = mask_2d.shape[1]

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(
            1, 1, seq_len, seq_len
        )
        padding_mask = mask_2d.view(B, 1, 1, seq_len)
        combined_mask = causal_mask * padding_mask

        min_dtype = torch.finfo(inputs_embeds.dtype).min
        inverted_mask = (1.0 - combined_mask) * min_dtype

        # 3. Position Ids
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

        # 4. Forward
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

        # 5. Loss
        loss_fn = nn.MSELoss(reduction="none")
        loss_unreduced = loss_fn(predicted_motion, motion)
        loss_per_frame = loss_unreduced.mean(dim=-1)
        loss = (loss_per_frame * motion_mask).sum() / motion_mask.sum()

        return loss, predicted_motion

    @torch.no_grad()
    def generate(self, text, max_new_tokens=120, temperature=1.0):
        """
        text: Input string
        max_new_tokens: Number of motion frames to generate
        """
        self.eval()
        device = self.qwen.device

        # 1. Prepare Text Inputs
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        # 2. Get Initial Embeddings [Text + Start_Token]
        transformer = self.qwen.base_model.model.model
        text_embeds = transformer.embed_tokens(input_ids)

        B = text_embeds.shape[0]
        start_token = self.start_motion_token.expand(B, -1, -1)

        # Concatenate: [Text, Start]
        # We feed this first to prime the KV cache
        current_inputs_embeds = torch.cat([text_embeds, start_token], dim=1)

        past_key_values = DynamicCache()
        generated_frames = []

        # 3. Autoregressive Loop
        for _ in range(max_new_tokens):
            # Forward pass
            # output_hidden_states=True is required to get the embedding for decoding
            outputs = self.qwen(
                inputs_embeds=current_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

            past_key_values = outputs.past_key_values

            # Get the last hidden state (corresponding to the last input token)
            # Shape: (Batch, 1, Hidden_Dim)
            last_hidden_state = outputs.hidden_states[-1][:, -1:, :]

            # Decode to Motion Space
            pred_motion = self.motion_decoder(last_hidden_state)
            generated_frames.append(pred_motion)

            # Prepare input for the next step
            # The output of step T becomes the input for step T+1
            # We must encode it back to embedding space
            current_inputs_embeds = self.motion_encoder(pred_motion)

        # 4. Assemble Result
        # Shape: (Batch, Seq_Len, Motion_Dim)
        full_motion = torch.cat(generated_frames, dim=1)

        return full_motion
