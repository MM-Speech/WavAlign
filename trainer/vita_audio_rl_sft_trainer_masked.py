import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from transformers import PreTrainedModel

from .vita_audio_rl_sft_trainer_fused import VitaAudioRLSFTTrainerFused

logger = logging.getLogger(__name__)


class VitaAudioRLSFTTrainerMasked(VitaAudioRLSFTTrainerFused):
    """
    RL+SFT trainer that allows selecting which token types (text/speech) contribute
    to the RL loss and the SFT loss independently, while preserving the fused-forward
    optimization from VitaAudioRLSFTTrainerFused.

    Args added:
    - rl_token_type:   "all" | "text" | "speech"   (default: "all")
    - sft_token_type:  "all" | "text" | "speech"   (default: "all")
    - include_audio_boundaries: bool                  (default: True)
        When True, <|begin_of_audio|> and <|end_of_audio|> are treated as speech tokens
        for masking decisions.
    - rl_speech_weight / rl_text_weight: scale RL loss per token type
    - sft_speech_weight / sft_text_weight: scale SFT loss per token type

    Usage:
        from trainer.vita_audio_rl_sft_trainer_masked import VitaAudioRLSFTTrainerMasked
        trainer = VitaAudioRLSFTTrainerMasked(
            ..., rl_token_type="speech", sft_token_type="text"
        )
    """

    def __init__(
        self,
        *args,
        rl_token_type: str = "all",
        sft_token_type: str = "all",
        include_audio_boundaries: bool = True,
        rl_speech_weight: float = 1.0,
        rl_text_weight: float = 1.0,
        sft_speech_weight: float = 1.0,
        sft_text_weight: float = 1.0,
        adaptive_mixing: bool = True,
        adaptive_lambda_max: float = 0.8,
        adaptive_ema_alpha: float = 0.9,
        adaptive_gate_slope: float = 5.0,
        adaptive_reward_threshold: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rl_token_type = (rl_token_type or "all").lower()
        self.sft_token_type = (sft_token_type or "all").lower()
        assert self.rl_token_type in ("all", "text", "speech"), "rl_token_type must be one of: all|text|speech"
        assert self.sft_token_type in ("all", "text", "speech"), "sft_token_type must be one of: all|text|speech"
        self.include_audio_boundaries = bool(include_audio_boundaries)
        self.rl_speech_weight = float(rl_speech_weight)
        self.rl_text_weight = float(rl_text_weight)
        self.sft_speech_weight = float(sft_speech_weight)
        self.sft_text_weight = float(sft_text_weight)
        self.adaptive_mixing = bool(adaptive_mixing)
        self.adaptive_lambda_max = float(adaptive_lambda_max)
        self.adaptive_ema_alpha = float(adaptive_ema_alpha)
        self.adaptive_gate_slope = float(adaptive_gate_slope)
        self.adaptive_reward_threshold = float(adaptive_reward_threshold)
        self._adaptive_lambda_state: Optional[float] = None
        self._last_adaptive_stats: Optional[Dict[str, float]] = None

    # --- token-type masking helpers ---
    def _get_token_type_masks(self, completion_ids: torch.Tensor, completion_masks: torch.Tensor):
        """
        Build text/speech token masks. Speech tokens include ids >= <|audio_0|> and
        optionally the <|begin_of_audio|>/<|end_of_audio|> markers if include_audio_boundaries.
        completion_ids: (B, L)
        completion_masks: (B, L) in {0,1}
        """
        speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
        vocab_size = getattr(self.model.config, "vocab_size", 168072)

        # Base classification
        speech_mask = ((completion_ids >= speech_token_offset) & (completion_ids < vocab_size))
        text_mask = (completion_ids < speech_token_offset)

        if self.include_audio_boundaries:
            begin_of_audio_id = self.processing_class.convert_tokens_to_ids("<|begin_of_audio|>")
            end_of_audio_id = self.processing_class.convert_tokens_to_ids("<|end_of_audio|>")
            speech_mask = speech_mask | (completion_ids == begin_of_audio_id) | (completion_ids == end_of_audio_id)
            # Ensure boundaries are not counted as text
            text_mask = text_mask & (completion_ids != begin_of_audio_id) & (completion_ids != end_of_audio_id)

        speech_mask = speech_mask.float() * completion_masks.float()
        text_mask = text_mask.float() * completion_masks.float()
        return speech_mask, text_mask

    # --- SFT ---
    def _compute_sft_loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SFT loss but restrict averaging to the selected token type (text/speech/all).
        Labels are -100 on positions that should be ignored.
        """
        shift_labels = labels[:, 1:]
        valid_mask = (shift_labels != -100)

        speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
        vocab_size = getattr(self.model.config, "vocab_size", 168072)
        speech_mask = ((shift_labels >= speech_token_offset) & (shift_labels < vocab_size))
        text_mask = (shift_labels < speech_token_offset)
        if self.include_audio_boundaries:
            boa = self.processing_class.convert_tokens_to_ids("<|begin_of_audio|>")
            eoa = self.processing_class.convert_tokens_to_ids("<|end_of_audio|>")
            speech_mask = speech_mask | (shift_labels == boa) | (shift_labels == eoa)
            text_mask = text_mask & (shift_labels != boa) & (shift_labels != eoa)

        speech_mask = speech_mask & valid_mask
        text_mask = text_mask & valid_mask

        if self.sft_token_type == "speech":
            allowed_mask = speech_mask
        elif self.sft_token_type == "text":
            allowed_mask = text_mask
        else:  # "all"
            allowed_mask = valid_mask

        row_mask = allowed_mask.any(dim=1)
        if not row_mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Standard NLL with masked positions
        logprobs = torch.log_softmax(logits, dim=-1)
        shift_logprobs = logprobs[:, :-1, :]

        # Avoid gather on invalid positions
        safe_labels = torch.where(allowed_mask, shift_labels, torch.zeros_like(shift_labels))
        picked = shift_logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        weight_mask = allowed_mask.float()
        if self.sft_token_type == "speech":
            weight_mask = weight_mask * self.sft_speech_weight
        elif self.sft_token_type == "text":
            weight_mask = weight_mask * self.sft_text_weight
        else:
            # all：根据 token 类型分别加权，其他 token 保持 1
            weight_mask = valid_mask.float()
            weight_mask = torch.where(speech_mask, self.sft_speech_weight * valid_mask.float(), weight_mask)
            weight_mask = torch.where(text_mask, self.sft_text_weight * valid_mask.float(), weight_mask)

        nll = -(picked * weight_mask)

        tok_counts = weight_mask.sum(dim=1).clamp(min=1e-6)
        per_sample_loss = nll.sum(dim=1) / tok_counts
        sft_mean = per_sample_loss[row_mask].mean()
        return sft_mean

    # --- RL ---
    def _compute_rl_loss_from_logits(
        self,
        model: PreTrainedModel,
        rl,
        policy_logps: torch.Tensor,
        policy_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Copy of fused RL loss with the final averaging mask restricted to the configured
        token type. KL weighting remains split across text/speech as in the base.
        """
        # Early out
        if rl.total_rows == 0:
            return torch.tensor(0.0, device=model.device)

        # Reference model forward (identical to base)
        with torch.no_grad():
            if self.ref_model is not None:
                self.ref_model.train()
                ref_kwargs = {
                    "input_ids": rl.batch_sequences,
                    "attention_mask": rl.batch_attention_masks,
                    "use_cache": False,
                }
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
                target_device = rl.batch_sequences.device
                self.ref_model = self.ref_model.to(target_device)
                ref_out = self.ref_model(**ref_kwargs)
                ref_logits = ref_out.logits
                ref_logps = torch.log_softmax(ref_logits[:, :-1, :], dim=-1).gather(
                    -1, rl.batch_sequences[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
            else:
                ref_logps = policy_logps.detach()

        total_loss = 0.0
        batch_size = rl.total_rows // self.args.num_generations
        num_generations = self.args.num_generations
        adaptive_raw_values: List[float] = []
        adaptive_values: List[float] = []
        adaptive_reward_gates: List[float] = []
        adaptive_info_gates: List[float] = []
        adaptive_reward_maxes: List[float] = []
        adaptive_reward_vars: List[float] = []

        for sample_idx in range(batch_size):
            st = sample_idx * num_generations
            ed = st + num_generations

            sample_model_logps = policy_logps[st:ed]
            sample_ref_logps = ref_logps[st:ed]

            prompt_len = rl.all_prompt_lengths[st]
            completion_model_logps = sample_model_logps[:, prompt_len - 1:]
            completion_ref_logps = sample_ref_logps[:, prompt_len - 1:]

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

            # Rewards
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
                    try:
                        tok = self.reward_processing_classes[idx]
                        texts = [sample_messages[g] + sample_completions[g] for g in range(num_generations)]
                        inp = tok(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
                        inp = {k: v.to(model.device) for k, v in inp.items()}
                        with torch.inference_mode():
                            rewards_f[:, idx] = reward_func(**inp).logits[:, 0]
                    except Exception:
                        rewards_f[:, idx] = 0.0

            rewards_f *= self.reward_weights.to(model.device).unsqueeze(0)
            rewards = rewards_f.sum(dim=1)
            grouped_mean = rewards.mean()
            grouped_std = rewards.std(unbiased=False) + 1e-4
            adv = (rewards - grouped_mean) / grouped_std

            if self.adaptive_mixing:
                adaptive_stats = self._compute_adaptive_lambda_stats(rewards)
                adaptive_raw_values.append(adaptive_stats["lambda_raw"])
                adaptive_values.append(adaptive_stats["lambda"])
                adaptive_reward_gates.append(adaptive_stats["reward_gate"])
                adaptive_info_gates.append(adaptive_stats["info_gate"])
                adaptive_reward_maxes.append(adaptive_stats["reward_max"])
                adaptive_reward_vars.append(adaptive_stats["reward_var"])

            # Per-token KL
            per_token_kl = torch.exp(completion_ref_logps - completion_model_logps) - (
                completion_ref_logps - completion_model_logps
            ) - 1

            # Token-type KL weighting consistent with base
            speech_token_mask = None
            text_token_mask = None
            if self._supports_token_type_separation():
                completion_ids_for_mask = rl.batch_sequences[st:ed, prompt_len : prompt_len + max_logps_len]
                if completion_ids_for_mask.size(1) != max_logps_len:
                    if completion_ids_for_mask.size(1) < max_logps_len:
                        pad = max_logps_len - completion_ids_for_mask.size(1)
                        completion_ids_for_mask = F.pad(
                            completion_ids_for_mask,
                            (0, pad),
                            value=self.processing_class.pad_token_id,
                        )
                    else:
                        completion_ids_for_mask = completion_ids_for_mask[:, :max_logps_len]
                speech_token_mask, text_token_mask = self._get_token_type_masks(
                    completion_ids_for_mask, sample_completion_masks
                )
                text_beta = getattr(self.args, "text_beta", self.args.beta)
                speech_beta = getattr(self.args, "speech_beta", self.args.beta * 0.5)
                weighted_kl = (
                    text_beta * per_token_kl * text_token_mask
                    + speech_beta * per_token_kl * speech_token_mask
                )
            else:
                weighted_kl = self.args.beta * per_token_kl

            per_tok_loss = torch.exp(completion_model_logps - completion_model_logps.detach()) * adv.unsqueeze(1)
            per_tok_loss = -(per_tok_loss - weighted_kl)

            # 构造带权重的掩码
            if self._supports_token_type_separation():
                if self.rl_token_type == "speech":
                    loss_mask_used = speech_token_mask * self.rl_speech_weight
                elif self.rl_token_type == "text":
                    loss_mask_used = text_token_mask * self.rl_text_weight
                else:  # all
                    loss_mask_used = sample_completion_masks + (
                        (self.rl_speech_weight - 1.0) * speech_token_mask
                        + (self.rl_text_weight - 1.0) * text_token_mask
                    )
            else:
                loss_mask_used = sample_completion_masks

            if torch.all(loss_mask_used <= 0):
                loss_mask_used = sample_completion_masks

            sample_loss = (
                (per_tok_loss * loss_mask_used).sum(dim=1)
                / loss_mask_used.sum(dim=1).clamp(min=1e-6)
            ).mean()
            total_loss = total_loss + sample_loss

            # metrics parity with base
            comp_len = sample_completion_masks.sum(1).float().mean()
            self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(comp_len).mean().item())

            rw_gather = self.accelerator.gather_for_metrics(rewards_f).mean(0)
            for i, rf in enumerate(self.reward_funcs):
                nm = rf.config._name_or_path.split("/")[-1] if hasattr(rf, "config") else getattr(rf, "__name__", f"reward_{i}")
                self._metrics[f"rewards/{nm}"].append(self.accelerator.gather_for_metrics(rw_gather[i]).mean().item())
            self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
            self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(grouped_std).mean().item())

            if self._supports_token_type_separation():
                text_kl = ((per_token_kl * text_token_mask).sum(1) / text_token_mask.sum(1).clamp(min=1)).mean()
                speech_kl = ((per_token_kl * speech_token_mask).sum(1) / speech_token_mask.sum(1).clamp(min=1)).mean()
                kl_overall = ((per_token_kl * sample_completion_masks).sum(1) / sample_completion_masks.sum(1).clamp(min=1)).mean()
                self._metrics["text_kl"].append(self.accelerator.gather_for_metrics(text_kl).mean().item())
                self._metrics["speech_kl"].append(self.accelerator.gather_for_metrics(speech_kl).mean().item())
                self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl_overall).mean().item())
                if policy_logits is not None:
                    with torch.no_grad():
                        logit_start = max(prompt_len - 1, 0)
                        logit_end = logit_start + max_logps_len
                        completion_logits = policy_logits[st:ed, logit_start:logit_end, :]
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

        if self.adaptive_mixing and adaptive_values:
            mean_lambda = sum(adaptive_values) / len(adaptive_values)
            self._adaptive_lambda_state = mean_lambda
            self._last_adaptive_stats = {
                "lambda_raw": sum(adaptive_raw_values) / len(adaptive_raw_values),
                "lambda": mean_lambda,
                "reward_gate": sum(adaptive_reward_gates) / len(adaptive_reward_gates),
                "info_gate": sum(adaptive_info_gates) / len(adaptive_info_gates),
                "reward_max": sum(adaptive_reward_maxes) / len(adaptive_reward_maxes),
                "reward_var": sum(adaptive_reward_vars) / len(adaptive_reward_vars),
            }
        else:
            self._last_adaptive_stats = None

        return total_loss / max(batch_size, 1)

    def get_effective_mix_weights(self) -> Dict[str, float]:
        rl_weight = float(self.rl_loss_weight)
        if self.adaptive_mixing and self._last_adaptive_stats is not None:
            rl_weight = float(self._last_adaptive_stats["lambda"])
        rl_weight = max(0.0, min(1.0, rl_weight))
        return {
            "rl_weight": rl_weight,
            "sft_weight": 1.0 - rl_weight,
        }

    def _compute_adaptive_lambda_stats(self, rewards: torch.Tensor) -> Dict[str, float]:
        rewards_detached = rewards.detach()
        reward_max = float(rewards_detached.max().item())
        reward_var = float(rewards_detached.var(unbiased=False).item())
        info_gate = max(0.0, min(1.0, reward_var / 4.0))
        reward_gate = float(
            torch.sigmoid(
                torch.tensor(
                    self.adaptive_gate_slope * (reward_max - self.adaptive_reward_threshold),
                    device=rewards.device,
                )
            ).item()
        )
        lambda_raw = float(self.adaptive_lambda_max * reward_gate * info_gate)
        if self._adaptive_lambda_state is None:
            lambda_value = lambda_raw
        else:
            lambda_value = float(
                (1.0 - self.adaptive_ema_alpha) * lambda_raw
                + self.adaptive_ema_alpha * self._adaptive_lambda_state
            )
        lambda_value = max(0.0, min(self.adaptive_lambda_max, lambda_value))
        return {
            "reward_max": reward_max,
            "reward_var": reward_var,
            "reward_gate": reward_gate,
            "info_gate": info_gate,
            "lambda_raw": lambda_raw,
            "lambda": lambda_value,
        }
