"""
VITA-Audio RL+SFT Trainer (Fused Forward)

Goal:
- Avoid double forward of the main model by fusing RL (GRPO) and SFT passes.
- Build a single combined batch that contains:
  - RL prompt+generated completion sequences (no labels, used for RL loss)
  - SFT prompt+ground-truth completion sequences (labels mask only on GT part)
- Run one forward on the policy model to obtain logits for both objectives.
- Still run one separate forward on the reference model for KL (RL only).

Notes:
- Preserves full SFT preprocessing (history text, audio question, text+audio answer),
  including discrete/contiguous audio injection and audio_indices construction.
  Focuses on reducing the duplicate forward on the policy model while matching behavior.

Usage:
- Import and use VitaAudioRLSFTTrainerFused instead of the clean trainer in your launcher
  if you want the fused-forward optimization.
"""

from __future__ import annotations

import logging
import copy
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from trl.models import unwrap_model_for_generation
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from utils.import_utils import ensure_vita_audio_importable

ensure_vita_audio_importable()
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous

from .vita_audio_rl_sft_trainer_clean import (
    VitaAudioRLSFTTrainer,
)

logger = logging.getLogger(__name__)


class VitaAudioRLSFTTrainerFused(VitaAudioRLSFTTrainer):
    """
    RL+SFT trainer that fuses the policy forward for RL and SFT into one pass.
    """

    def compute_loss(self, model: PreTrainedModel, inputs, return_outputs: bool = False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The VitaAudioRLSFTTrainerFused does not support returning outputs")

        # 1) Build RL generated sequences and metadata (copied/adapted from clean _compute_grpo_loss)
        rl_context = self._build_rl_context(model, inputs)

        # 2) Build SFT sequences (text-only) and labels for fused pass
        sft_context = self._build_sft_context_text_only(model, inputs)

        # If no RL rows and no SFT rows, return zero
        if rl_context.total_rows == 0 and sft_context.total_rows == 0:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # 3) Create a single combined batch for the policy model
        comb = self._combine_rl_sft_batches(rl_context, sft_context, model)

        # Single forward for policy model; pass labels to trigger label-dependent branches
        # (RL rows are fully -100 so they contribute no CE). We still ignore outputs.loss
        # and compute our own SFT + RL losses from logits.
        model_kwargs = {
            "input_ids": comb["input_ids"],
            "attention_mask": comb["attention_mask"],
            "use_cache": False,
        }
        # Ensure labels are passed to preserve original VITA-Audio behavior with labels
        if "labels" in comb:
            model_kwargs["labels"] = comb["labels"]
        if "audios" in comb:
            model_kwargs["audios"] = comb["audios"]
        if "audio_indices" in comb:
            model_kwargs["audio_indices"] = comb["audio_indices"]
        outputs = model(**model_kwargs)

        logits = outputs.logits  # (B, L, V)
        # Compute per-token logps for actual next tokens
        target_ids = comb["input_ids"][:, 1:]
        logps = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # 4) Compute SFT loss from logits using labels (only SFT rows contribute)
        sft_loss = self._compute_sft_loss_from_logits(logits, comb["labels"]) if sft_context.total_rows > 0 else torch.tensor(0.0, device=model.device)

        # 5) Compute RL loss using policy logits and ref logits (one extra ref forward)
        rl_loss = self._compute_rl_loss_from_logits(model, rl_context, logps, policy_logits=logits)

        # 6) Combine losses
        if hasattr(self, "get_effective_mix_weights"):
            mix_weights = self.get_effective_mix_weights()
            effective_rl_weight = mix_weights["rl_weight"]
            effective_sft_weight = mix_weights["sft_weight"]
        else:
            effective_rl_weight = float(self.rl_loss_weight)
            effective_sft_weight = float(self.sft_loss_weight)
        total_loss = effective_rl_weight * rl_loss + effective_sft_weight * sft_loss

        # 7) Metrics (global means)
        rl_loss_g = self.accelerator.gather_for_metrics(rl_loss.detach()).mean().item()
        sft_loss_g = self.accelerator.gather_for_metrics(sft_loss.detach()).mean().item()
        total_loss_g = self.accelerator.gather_for_metrics(total_loss.detach()).mean().item()
        self._metrics["rl_loss"].append(rl_loss_g)
        self._metrics["sft_loss"].append(sft_loss_g)
        self._metrics["combined_loss"].append(total_loss_g)
        self._metrics["num_samples"].append(self.accelerator.gather_for_metrics(torch.tensor(len(inputs), device=total_loss.device, dtype=torch.float32)).mean().item())
        self._metrics["effective_rl_weight"].append(effective_rl_weight)
        self._metrics["effective_sft_weight"].append(effective_sft_weight)
        if hasattr(self, "_last_adaptive_stats") and self._last_adaptive_stats is not None:
            self._metrics["adaptive_lambda_raw"].append(self._last_adaptive_stats["lambda_raw"])
            self._metrics["adaptive_lambda"].append(self._last_adaptive_stats["lambda"])
            self._metrics["adaptive_reward_gate"].append(self._last_adaptive_stats["reward_gate"])
            self._metrics["adaptive_info_gate"].append(self._last_adaptive_stats["info_gate"])
            self._metrics["adaptive_reward_max"].append(self._last_adaptive_stats["reward_max"])
            self._metrics["adaptive_reward_var"].append(self._last_adaptive_stats["reward_var"])

        logger.info(
            "[Fused] GRPO: %.4f, SFT: %.4f, Combined: %.4f, mix=(%.3f RL / %.3f SFT)",
            rl_loss_g,
            sft_loss_g,
            total_loss_g,
            effective_rl_weight,
            effective_sft_weight,
        )

        return total_loss

    # -------------------- helpers --------------------
    class _RLContext:
        def __init__(self):
            self.all_generated_sequences: List[torch.Tensor] = []  # (1, P+C)
            self.all_prompt_lengths: List[int] = []
            self.all_completions_text: List[str] = []
            self.all_messages_for_reward: List[str] = []
            self.all_decoded_audios: List[Optional[Any]] = []
            self.audios_per_row: List[List[torch.Tensor]] = []
            self.audio_indices_per_row: List[List[torch.Tensor]] = []
            self.batch_sequences: Optional[torch.Tensor] = None  # (N_rl, L)
            self.batch_attention_masks: Optional[torch.Tensor] = None  # (N_rl, L)
            self.batch_completion_masks: Optional[torch.Tensor] = None  # (N_rl, C_max)
            self.num_generations: int = 0
            self.total_rows: int = 0

    class _SFTContext:
        def __init__(self):
            self.sequences: List[torch.Tensor] = []
            self.labels: List[torch.Tensor] = []
            self.total_rows: int = 0
            self.audios_per_sample: List[List[torch.Tensor]] = []
            self.audio_indices_per_sample: List[List[torch.Tensor]] = []

    def _build_rl_context(self, model: PreTrainedModel, rl_inputs) -> _RLContext:
        ctx = self._RLContext()
        batch_size = len(rl_inputs)
        num_generations = self.args.num_generations
        ctx.num_generations = num_generations

        for sample_idx in range(batch_size):
            current_sample = rl_inputs[sample_idx]
            messages = current_sample["messages"]
            audios = current_sample.get("audios", [])

            processed_messages = self._prepare_s2s_messages(messages, audios)
            prompt_input_ids = self.processing_class.apply_chat_template(
                processed_messages, tokenize=True, add_generation_prompt=self.add_generation_prompt
            )

            # Handle audio in prompt (contiguous or discrete) like clean trainer
            audios_processed = None
            audio_indices = None
            if audios and self.audio_tokenizer and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
                prompt_input_ids, audios_processed, audio_indices = add_audio_input_contiguous(
                    prompt_input_ids, audios, self.processing_class, self.audio_tokenizer
                )
                logger.debug(f"Processed contiguous audio: {len(audios)} files")
            elif audios and self.audio_tokenizer and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                processed_messages = self._process_discrete_audio_tokens(processed_messages, audios)
                prompt_input_ids = self.processing_class.apply_chat_template(
                    processed_messages, tokenize=True, add_generation_prompt=self.add_generation_prompt
                )
                logger.debug("Using discrete audio tokens in text")

            prompt_inputs_dict = {"input_ids": torch.tensor([prompt_input_ids], dtype=torch.long)}
            if audios_processed is not None:
                prompt_inputs_dict["audios"] = audios_processed
            if audio_indices is not None:
                prompt_inputs_dict["audio_indices"] = audio_indices
            for k, v in list(prompt_inputs_dict.items()):
                if torch.is_tensor(v):
                    prompt_inputs_dict[k] = v.to(model.device)
            prompt_length = prompt_inputs_dict["input_ids"].size(1)

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped, torch.no_grad():
                generation_config = copy.deepcopy(self.generation_config)
                generation_config.pad_token_id = self.processing_class.pad_token_id
                generation_config.eos_token_id = self.processing_class.eos_token_id
                generated_sequences = self._generate_sequences_for_variant(
                    generation_model=unwrapped,
                    prompt_inputs=prompt_inputs_dict,
                    generation_config=generation_config,
                    num_generations=num_generations,
                )

            # Pad same length for this sample's generations and store
            max_len = max(seq.size(0) for seq in generated_sequences)
            padded = []
            for seq in generated_sequences:
                if seq.size(0) < max_len:
                    pad = F.pad(seq, (0, max_len - seq.size(0)), value=self.processing_class.pad_token_id)
                else:
                    pad = seq
                padded.append(pad.unsqueeze(0))
            generated_sequences = torch.cat(padded, dim=0)  # (G, P+C)

            for g in range(num_generations):
                seq = generated_sequences[g:g+1]
                ctx.all_generated_sequences.append(seq)
                ctx.all_prompt_lengths.append(prompt_length)

                completion_ids = seq[:, prompt_length:]
                speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
                text_only_ids = [tid for tid in completion_ids[0].tolist() if tid < speech_token_offset]
                completion_text = self.processing_class.decode(text_only_ids, skip_special_tokens=True)
                ctx.all_completions_text.append(completion_text)

                reward_prompt = current_sample.get("question_text")
                if not reward_prompt:
                    user_segments = []
                    for msg in processed_messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if content:
                                user_segments.append(content.replace("<|audio|>", "").strip())
                    reward_prompt = "\n\n".join(seg for seg in user_segments if seg) or "[audio-only prompt]"
                ctx.all_messages_for_reward.append(reward_prompt)

                # Decode audio tokens (if any) for reward functions that need audio
                decoded_audio = None
                if self.audio_tokenizer is not None:
                    audio_tokens = []
                    for token_id in completion_ids[0].tolist():
                        if token_id >= speech_token_offset:
                            audio_tokens.append(token_id - speech_token_offset)
                    if len(audio_tokens) > 0:
                        try:
                            decoded_audio = self.audio_tokenizer.decode(audio_tokens)
                        except Exception as e:
                            logger.debug(f"Audio decode failed in fused trainer: {e}")
                            decoded_audio = None
                ctx.all_decoded_audios.append(decoded_audio)

                # Keep prompt audio lists for this RL row (so combined forward gets audio features)
                if audios_processed is not None and audio_indices is not None:
                    ctx.audios_per_row.append([audios_processed] if isinstance(audios_processed, torch.Tensor) else audios_processed)
                    # audio_indices can be a single tensor or list; normalize to list
                    if isinstance(audio_indices, torch.Tensor):
                        ctx.audio_indices_per_row.append([audio_indices])
                    else:
                        ctx.audio_indices_per_row.append(audio_indices)
                else:
                    ctx.audios_per_row.append([])
                    ctx.audio_indices_per_row.append([])
        # import os, torch.distributed as dist, pdb
        # if int(os.environ.get("RANK", 0)) == 0:
        #     breakpoint()
        # else:
        #     dist.barrier()
        # Batch RL sequences (B*G)
        if ctx.all_generated_sequences:
            max_seq_length = max(seq.size(1) for seq in ctx.all_generated_sequences)
            pad_list = []
            for seq in ctx.all_generated_sequences:
                if seq.size(1) < max_seq_length:
                    pad_seq = F.pad(seq, (0, max_seq_length - seq.size(1)), value=self.processing_class.pad_token_id)
                else:
                    pad_seq = seq
                pad_list.append(pad_seq)
            ctx.batch_sequences = torch.cat(pad_list, dim=0).to(model.device)
            ctx.batch_attention_masks = (ctx.batch_sequences != self.processing_class.pad_token_id).float()

            # Build completion masks (up to first EOS or last non-pad)
            comp_masks = []
            prompt_lens = torch.tensor(ctx.all_prompt_lengths, device=ctx.batch_sequences.device)
            for i in range(ctx.batch_sequences.size(0)):
                p_len = int(prompt_lens[i].item())
                comp = torch.zeros(ctx.batch_sequences.size(1) - p_len, device=ctx.batch_sequences.device)
                part = ctx.batch_sequences[i, p_len:]
                eos_mask = (part == self.processing_class.eos_token_id)
                if eos_mask.any():
                    eos_idx = int(eos_mask.nonzero(as_tuple=True)[0][0].item())
                    comp[: eos_idx + 1] = 1
                else:
                    non_pad = (part != self.processing_class.pad_token_id)
                    if non_pad.any():
                        last = int(non_pad.nonzero(as_tuple=True)[0][-1].item())
                        comp[: last + 1] = 1
                comp_masks.append(comp.unsqueeze(0))
            ctx.batch_completion_masks = torch.cat(comp_masks, dim=0)
            ctx.total_rows = ctx.batch_sequences.size(0)
        else:
            ctx.total_rows = 0

        return ctx

    def _build_sft_context_text_only(self, model: PreTrainedModel, inputs) -> _SFTContext:
        """Build full SFT sequences and labels for fused forward, mirroring clean implementation.
        Handles history as text, current question possibly as audio, and answer as text+audio.
        Supports discrete and contiguous audio injection with audio_indices construction.
        """
        ctx = self._SFTContext()
        IGNORE_TOKEN_ID = -100

        # Import VITA-Audio constants (consistent with clean)
        from vita_audio.constants import (
            AUD_START_TOKEN, AUD_END_TOKEN, AUD_TAG_TOKEN, AUD_CONTEXT_TOKEN
        )

        # Token ids
        AUD_CONTEXT_ID = self.processing_class(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids[0]
        AUD_START_ID = self.processing_class(AUD_START_TOKEN, add_special_tokens=False).input_ids[0]
        AUD_END_ID = self.processing_class(AUD_END_TOKEN, add_special_tokens=False).input_ids[0]
        AUD_TAG_ID = self.processing_class(AUD_TAG_TOKEN, add_special_tokens=False).input_ids[0]

        IM_START = "<|im_start|>"
        IM_END = "<|im_end|>"
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

        nl_tokens = self.processing_class("\n", add_special_tokens=False).input_ids
        IM_START_IDS = self.processing_class(IM_START, add_special_tokens=False).input_ids
        IM_END_IDS = self.processing_class(IM_END, add_special_tokens=False).input_ids
        USER_IDS = self.processing_class(USER, add_special_tokens=False).input_ids
        ASSISTANT_IDS = self.processing_class(ASSISTANT, add_special_tokens=False).input_ids
        SYSTEM_IDS = self.processing_class(SYSTEM, add_special_tokens=False).input_ids

        for sample in inputs:
            sft_target_text = sample.get("sft_target_text", "")
            sft_target_audio_path = sample.get("sft_target_audio")
            if not (sft_target_text or sft_target_audio_path):
                continue

            # try:
            messages = sample["messages"].copy()
            audios = sample.get("audios", []).copy()

            # Build assistant target content: text + optional audio
            target_content = ""
            if sft_target_text:
                target_content += sft_target_text
            if sft_target_audio_path:
                if target_content:
                    target_content += "\n<|audio|>"
                else:
                    target_content = "<|audio|>"
                audios.append(sft_target_audio_path)
            messages.append({"role": "assistant", "content": target_content})

            contiguous_audio_idxs: List[int] = []

            # Discrete audio token injection
            if audios and self.audio_tokenizer and getattr(self.audio_tokenizer, 'is_discrete', False):
                audio_tokens_list = [
                    self.audio_tokenizer.encode(x, is_discrete=True, is_contiguous=False)
                    for x in audios
                ]
                audio_tokens_list = ["".join(f"<|audio_{i}|>" for i in x) for x in audio_tokens_list]
                audio_idx = 0
                for j, sentence in enumerate(messages):
                    content = sentence["content"]
                    role = sentence["role"]
                    if self.audio_tokenizer.apply_to_role(role, is_discrete=True):
                        while AUD_TAG_TOKEN in content:
                            content = content.replace(
                                AUD_TAG_TOKEN,
                                f"{AUD_START_TOKEN}{audio_tokens_list[audio_idx]}{AUD_END_TOKEN}",
                                1,
                            )
                            audio_idx += 1
                    else:
                        audio_count = content.count(AUD_TAG_TOKEN)
                        contiguous_audio_idxs.extend(range(audio_idx, audio_idx + audio_count))
                        audio_idx += audio_count
                    sentence["content"] = content

            # Build input_ids and targets with im template
            input_ids: List[int] = []
            targets: List[int] = []
            audios_processed: List[torch.Tensor] = []
            audio_indices_processed: List[torch.Tensor] = []

            processed_messages = self._prepare_s2s_messages(messages, audios)

            for j, sentence in enumerate(processed_messages):
                role = sentence["role"]
                content = sentence["content"]
                if role == "user":
                    _input_id = (
                        IM_START_IDS + USER_IDS + nl_tokens +
                        self.processing_class(content, add_special_tokens=False).input_ids +
                        IM_END_IDS + nl_tokens
                    )
                    _target = [IGNORE_TOKEN_ID] * len(_input_id)
                elif role == "assistant":
                    content_input_id = self.processing_class(content, add_special_tokens=False).input_ids
                    if self.audio_tokenizer is not None:
                        content_input_id = self._apply_text_audio_interval(content_input_id, AUD_START_ID, AUD_END_ID)
                    _input_id = IM_START_IDS + ASSISTANT_IDS + nl_tokens + content_input_id + IM_END_IDS + nl_tokens
                    # Supervise assistant content INCLUDING IM_END and following newline,
                    # matching original VITA-Audio SFT behavior.
                    _target = (
                        [IGNORE_TOKEN_ID] * len(IM_START_IDS) +
                        [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS) +
                        [IGNORE_TOKEN_ID] * len(nl_tokens) +
                        content_input_id + IM_END_IDS + nl_tokens
                    )
                else:  # system
                    _input_id = (
                        IM_START_IDS + SYSTEM_IDS + nl_tokens +
                        self.processing_class(content, add_special_tokens=False).input_ids +
                        IM_END_IDS + nl_tokens
                    )
                    _target = [IGNORE_TOKEN_ID] * len(_input_id)
                input_ids += _input_id
                targets += _target

            # Contiguous audio injection (create audio_indices)
            if contiguous_audio_idxs and self.audio_tokenizer and getattr(self.audio_tokenizer, 'is_contiguous', False):
                aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
                if len(aud_positions) > 0:
                    new_input_ids: List[int] = []
                    new_targets: List[int] = []
                    st = 0
                    for pos_idx, aud_pos in enumerate(aud_positions):
                        if pos_idx < len(contiguous_audio_idxs):
                            actual_audio_idx = contiguous_audio_idxs[pos_idx]
                            audio = self.audio_tokenizer.encode(audios[actual_audio_idx], is_contiguous=True)
                            audios_processed.append(audio)
                            audio_token_length = audio.size(0) + 4

                            new_input_ids += input_ids[st:aud_pos]
                            new_targets += targets[st:aud_pos]

                            new_input_ids += [AUD_START_ID]
                            new_targets += [IGNORE_TOKEN_ID]

                            audio_indice_b = torch.zeros(1, audio_token_length, dtype=torch.int64)
                            audio_indice_s = torch.arange(
                                len(new_input_ids), len(new_input_ids) + audio_token_length
                            ).unsqueeze(0).repeat(1, 1)
                            audio_indice_b_s = torch.stack([audio_indice_b, audio_indice_s], dim=0)
                            audio_indices_processed.append(audio_indice_b_s)

                            new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
                            new_targets += [IGNORE_TOKEN_ID] * audio_token_length

                            new_input_ids += [AUD_END_ID]
                            new_targets += [IGNORE_TOKEN_ID]

                            st = aud_pos + 1
                    new_input_ids += input_ids[st:]
                    new_targets += targets[st:]
                    input_ids = new_input_ids
                    targets = new_targets

            # Final tensors per-sample
            seq = torch.tensor([input_ids], dtype=torch.long, device=model.device)
            lab = torch.tensor([targets], dtype=torch.long, device=model.device)
            ctx.sequences.append(seq)
            ctx.labels.append(lab)
            ctx.audios_per_sample.append(audios_processed)
            ctx.audio_indices_per_sample.append(audio_indices_processed)

            # except Exception as e:
            #     logger.warning(f"SFT processing failed for sample in fused: {e}")
            #     print(e)
            #     continue

        # Pad SFT rows to same length (independently from RL)
        if ctx.sequences:
            max_len = max(s.size(1) for s in ctx.sequences)
            seqs = []
            labs = []
            for s, l in zip(ctx.sequences, ctx.labels):
                if s.size(1) < max_len:
                    pad = max_len - s.size(1)
                    s = F.pad(s, (0, pad), value=self.processing_class.pad_token_id)
                    l = F.pad(l, (0, pad), value=IGNORE_TOKEN_ID)
                seqs.append(s)
                labs.append(l)
            ctx.sequences = [torch.cat(seqs, dim=0)]
            ctx.labels = [torch.cat(labs, dim=0)]
            ctx.total_rows = ctx.sequences[0].size(0)
        else:
            ctx.total_rows = 0

        return ctx

    def _combine_rl_sft_batches(self, rl: _RLContext, sft: _SFTContext, model: PreTrainedModel) -> Dict[str, Any]:
        has_rl = rl.total_rows > 0
        has_sft = sft.total_rows > 0

        if has_rl and has_sft:
            sft_seqs = sft.sequences[0]
            sft_labs = sft.labels[0]
            max_len = max(rl.batch_sequences.size(1), sft_seqs.size(1))

            def pad_to(x: torch.Tensor, pad_id: int, length: int) -> torch.Tensor:
                if x.size(1) < length:
                    pad = length - x.size(1)
                    return F.pad(x, (0, pad), value=pad_id)
                return x

            rl_inputs_padded = pad_to(rl.batch_sequences, self.processing_class.pad_token_id, max_len)
            sft_inputs_padded = pad_to(sft_seqs, self.processing_class.pad_token_id, max_len)
            input_ids = torch.cat([rl_inputs_padded, sft_inputs_padded], dim=0)

            rl_labels = torch.full_like(rl_inputs_padded, -100)
            sft_labels = sft_labs
            if sft_labels.size(1) < max_len:
                pad = max_len - sft_labels.size(1)
                sft_labels = F.pad(sft_labels, (0, pad), value=-100)
            labels = torch.cat([rl_labels, sft_labels], dim=0)

            attention_mask = (input_ids != self.processing_class.pad_token_id).float()

            # Merge audios and audio_indices like data_collator, preserving strict 1:1 order
            audios: Optional[List[torch.Tensor]] = []
            audio_indices: Optional[List[torch.Tensor]] = []
            # 1) RL rows first
            for row_idx in range(rl.batch_sequences.size(0)):
                a_list = rl.audios_per_row[row_idx]
                i_list = rl.audio_indices_per_row[row_idx]
                if len(a_list) != len(i_list):
                    logger.warning(f"Mismatch audios/indices count in RL row {row_idx}: {len(a_list)} vs {len(i_list)}; truncating to min.")
                for a, idx_tensor in zip(a_list, i_list):
                    audios.append(a)
                    t = idx_tensor.clone()
                    t[0, :, :] = row_idx
                    audio_indices.append(t)
            # 2) Then SFT rows
            base_idx = rl.batch_sequences.size(0)
            for sample_row in range(sft.total_rows):
                a_list = sft.audios_per_sample[sample_row]
                i_list = sft.audio_indices_per_sample[sample_row]
                if len(a_list) != len(i_list):
                    logger.warning(f"Mismatch audios/indices count in SFT row {sample_row}: {len(a_list)} vs {len(i_list)}; truncating to min.")
                for a, idx_tensor in zip(a_list, i_list):
                    audios.append(a)
                    t = idx_tensor.clone()
                    t[0, :, :] = base_idx + sample_row
                    audio_indices.append(t)
            if len(audios) == 0:
                audios = None
            if len(audio_indices) == 0:
                audio_indices = None
        elif has_rl:
            input_ids = rl.batch_sequences
            attention_mask = rl.batch_attention_masks
            labels = torch.full_like(input_ids, -100)
            # RL-only: attach RL prompt audios/indices, preserving order
            audios = []
            audio_indices = []
            for row_idx in range(rl.batch_sequences.size(0)):
                a_list = rl.audios_per_row[row_idx]
                i_list = rl.audio_indices_per_row[row_idx]
                if len(a_list) != len(i_list):
                    logger.warning(f"Mismatch audios/indices count in RL row {row_idx}: {len(a_list)} vs {len(i_list)}; truncating to min.")
                for a, idx_tensor in zip(a_list, i_list):
                    audios.append(a)
                    t = idx_tensor.clone()
                    t[0, :, :] = row_idx
                    audio_indices.append(t)
            if len(audios) == 0:
                audios = None
            if len(audio_indices) == 0:
                audio_indices = None
        else:
            input_ids = sft.sequences[0]
            attention_mask = (input_ids != self.processing_class.pad_token_id).float()
            labels = sft.labels[0]
            audios = []
            for lst in sft.audios_per_sample:
                audios.extend(lst)
            audio_indices = []
            for sample_row, lst in enumerate(sft.audio_indices_per_sample):
                for idx_tensor in lst:
                    idx_tensor = idx_tensor.clone()
                    idx_tensor[0, :, :] = sample_row
                    audio_indices.append(idx_tensor)

        out: Dict[str, Any] = {
            "input_ids": input_ids.to(model.device),
            "attention_mask": attention_mask.to(model.device),
            "labels": labels.to(model.device),
        }
        if audios is not None and len(audios) > 0:
            out["audios"] = audios
        if audio_indices is not None and len(audio_indices) > 0:
            out["audio_indices"] = audio_indices
        return out

    def _compute_sft_loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-sample SFT loss from logits and labels (-100 ignored), return mean over SFT rows only.
        Assumes labels for non-SFT rows are -100 entirely.
        """
        with torch.no_grad():
            # Identify SFT rows
            row_mask = (labels != -100).any(dim=1)  # rows with any supervised token
        if not row_mask.any():
            return torch.tensor(0.0, device=logits.device)

        logprobs = torch.log_softmax(logits, dim=-1)
        shift_logprobs = logprobs[:, :-1, :]
        shift_labels = labels[:, 1:]
        # mask invalid positions and avoid gather on -100
        valid_mask = (shift_labels != -100)
        safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
        picked = shift_logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        nll = -(picked * valid_mask.float())
        tok_counts = valid_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = nll.sum(dim=1) / tok_counts
        sft_mean = per_sample_loss[row_mask].mean()
        return sft_mean

    def _compute_rl_loss_from_logits(
        self,
        model: PreTrainedModel,
        rl: _RLContext,
        policy_logps: torch.Tensor,
        policy_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Early out
        if rl.total_rows == 0:
            return torch.tensor(0.0, device=model.device)

        # Ref forward to get ref per-token logps (RL rows only)
        with torch.no_grad():
            if self.ref_model is not None:
                self.ref_model.train()
                # Build kwargs for reference forward; include audio for RL rows if available
                ref_kwargs = {
                    "input_ids": rl.batch_sequences,
                    "attention_mask": rl.batch_attention_masks,
                    "use_cache": False,
                }
                # Flatten RL audio lists and align indices' batch dimension to RL row indices
                ref_audios: List[torch.Tensor] = []
                ref_audio_indices: List[torch.Tensor] = []
                for row_idx in range(rl.batch_sequences.size(0)):
                    for a in rl.audios_per_row[row_idx]:
                        ref_audios.append(a)
                    for idx_tensor in rl.audio_indices_per_row[row_idx]:
                        t = idx_tensor.clone()
                        t[0, :, :] = row_idx
                        ref_audio_indices.append(t)
                if len(ref_audios) > 0 and len(ref_audio_indices) > 0:
                    ref_kwargs["audios"] = ref_audios
                    ref_kwargs["audio_indices"] = ref_audio_indices

                ref_out = self.ref_model(**ref_kwargs)
                ref_logits = ref_out.logits
                ref_logps = torch.log_softmax(ref_logits[:, :-1, :], dim=-1).gather(
                    -1, rl.batch_sequences[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
            else:
                # Zero-KL case: use model logps as ref to get kl=0
                ref_logps = policy_logps.detach()

        total_loss = 0.0
        batch_size = rl.total_rows // self.args.num_generations
        num_generations = self.args.num_generations

        for sample_idx in range(batch_size):
            st = sample_idx * num_generations
            ed = st + num_generations

            sample_model_logps = policy_logps[st:ed]            # (G, L-1)
            sample_ref_logps = ref_logps[st:ed]                 # (G, L-1)

            prompt_len = rl.all_prompt_lengths[st]
            completion_model_logps = sample_model_logps[:, prompt_len - 1:]  # (G, C)
            completion_ref_logps = sample_ref_logps[:, prompt_len - 1:]     # (G, C)

            target_logps_len = min(completion_model_logps.size(1), completion_ref_logps.size(1))
            if completion_model_logps.size(1) != completion_ref_logps.size(1):
                logger.debug(
                    "Adjusting completion logps length (model=%s, ref=%s, target=%s)",
                    completion_model_logps.size(1),
                    completion_ref_logps.size(1),
                    target_logps_len,
                )
            if completion_model_logps.size(1) != target_logps_len:
                completion_model_logps = completion_model_logps[:, :target_logps_len]
            if completion_ref_logps.size(1) != target_logps_len:
                completion_ref_logps = completion_ref_logps[:, :target_logps_len]

            sample_completion_masks = rl.batch_completion_masks[st:ed]
            max_logps_len = target_logps_len
            if sample_completion_masks.size(1) != max_logps_len:
                if sample_completion_masks.size(1) < max_logps_len:
                    pad = max_logps_len - sample_completion_masks.size(1)
                    sample_completion_masks = F.pad(sample_completion_masks, (0, pad), value=0)
                else:
                    sample_completion_masks = sample_completion_masks[:, :max_logps_len]

            # Rewards per generation
            rewards_f = torch.zeros(num_generations, len(self.reward_funcs), device=model.device)
            sample_completions = rl.all_completions_text[st:ed]
            sample_messages = rl.all_messages_for_reward[st:ed]
            sample_audios = rl.all_decoded_audios[st:ed] if rl.all_decoded_audios else [None] * num_generations
            for idx, reward_func in enumerate(self.reward_funcs):
                if hasattr(reward_func, "__call__") and not hasattr(reward_func, "config"):
                    reward_vals = reward_func(
                        prompts=sample_messages,
                        completions=sample_completions,
                        audios=sample_audios,
                    )
                    rewards_f[:, idx] = torch.tensor(reward_vals, dtype=torch.float32, device=model.device)
                else:
                    # Model-based reward
                    try:
                        tok = self.reward_processing_classes[idx]
                        texts = [sample_messages[g] + sample_completions[g] for g in range(num_generations)]
                        inp = tok(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
                        inp = {k: v.to(model.device) for k, v in inp.items()}
                        with torch.inference_mode():
                            rewards_f[:, idx] = reward_func(**inp).logits[:, 0]
                    except Exception as e:
                        logger.debug(f"Model-based reward failed in fused trainer: {e}")
                        rewards_f[:, idx] = 0.0

            rewards_f *= self.reward_weights.to(model.device).unsqueeze(0)
            rewards = rewards_f.sum(dim=1)  # (G,)

            grouped_mean = rewards.mean()
            grouped_std = rewards.std() + 1e-4
            adv = (rewards - grouped_mean) / grouped_std

            # Per-token KL
            per_token_kl = torch.exp(completion_ref_logps - completion_model_logps) - \
                           (completion_ref_logps - completion_model_logps) - 1

            # Text/Speech KL weighting parity with clean trainer
            if self._supports_token_type_separation():
                completion_ids_for_mask = rl.batch_sequences[st:ed, prompt_len:]
                speech_token_mask, text_token_mask = self._get_token_type_masks(
                    completion_ids_for_mask, sample_completion_masks
                )
                text_beta = getattr(self.args, 'text_beta', self.args.beta)
                speech_beta = getattr(self.args, 'speech_beta', self.args.beta * 0.5)
                weighted_kl = text_beta * per_token_kl * text_token_mask + \
                              speech_beta * per_token_kl * speech_token_mask
            else:
                weighted_kl = self.args.beta * per_token_kl

            per_tok_loss = torch.exp(completion_model_logps - completion_model_logps.detach()) * adv.unsqueeze(1)
            per_tok_loss = -(per_tok_loss - weighted_kl)
            sample_loss = ((per_tok_loss * sample_completion_masks).sum(dim=1) / sample_completion_masks.sum(dim=1).clamp(min=1)).mean()
            total_loss = total_loss + sample_loss

            # -------- metrics (global-aggregated) --------
            # completion length
            comp_len = sample_completion_masks.sum(1).float().mean()
            self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(comp_len).mean().item())

            # reward components and summary
            rw_gather = self.accelerator.gather_for_metrics(rewards_f).mean(0)
            for i, rf in enumerate(self.reward_funcs):
                nm = rf.config._name_or_path.split("/")[-1] if hasattr(rf, 'config') else getattr(rf, '__name__', f"reward_{i}")
                self._metrics[f"rewards/{nm}"].append(self.accelerator.gather_for_metrics(rw_gather[i]).mean().item())
            self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
            self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(grouped_std).mean().item())

            # KL metrics (and optional text/speech split)
            if self._supports_token_type_separation():
                completion_ids_for_mask = rl.batch_sequences[st:ed, prompt_len:]
                speech_token_mask, text_token_mask = self._get_token_type_masks(
                    completion_ids_for_mask, sample_completion_masks
                )
                text_kl = ((per_token_kl * text_token_mask).sum(1) / text_token_mask.sum(1).clamp(min=1)).mean()
                speech_kl = ((per_token_kl * speech_token_mask).sum(1) / speech_token_mask.sum(1).clamp(min=1)).mean()
                kl_overall = ((per_token_kl * sample_completion_masks).sum(1) / sample_completion_masks.sum(1).clamp(min=1)).mean()
                self._metrics["text_kl"].append(self.accelerator.gather_for_metrics(text_kl).mean().item())
                self._metrics["speech_kl"].append(self.accelerator.gather_for_metrics(speech_kl).mean().item())
                self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl_overall).mean().item())
                # Entropy by token type if logits available
                if policy_logits is not None:
                    with torch.no_grad():
                        completion_logits = policy_logits[st:ed, prompt_len - 1:-1, :]
                        entropy_stats = self._compute_entropy_by_token_type(
                            completion_logits, completion_ids_for_mask, sample_completion_masks
                        )
                        text_ent = torch.tensor(entropy_stats["text_entropy"], device=model.device)
                        speech_ent = torch.tensor(entropy_stats["speech_entropy"], device=model.device)
                        total_ent = torch.tensor(entropy_stats["total_entropy"], device=model.device)
                        self._metrics["text_entropy"].append(self.accelerator.gather_for_metrics(text_ent).mean().item())
                        self._metrics["speech_entropy"].append(self.accelerator.gather_for_metrics(speech_ent).mean().item())
                        self._metrics["total_entropy"].append(self.accelerator.gather_for_metrics(total_ent).mean().item())
            else:
                kl_overall = ((per_token_kl * sample_completion_masks).sum(1) / sample_completion_masks.sum(1).clamp(min=1)).mean()
                self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl_overall).mean().item())

        return total_loss / max(batch_size, 1)
