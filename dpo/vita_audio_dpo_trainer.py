"""Direct Preference Optimization trainer for VITA-Audio models."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from trl.models import create_reference_model, unwrap_model_for_generation

from trainer.vita_audio_rl_sft_trainer_clean import VitaAudioRLSFTTrainer
from utils.import_utils import ensure_vita_audio_importable

ensure_vita_audio_importable()
from vita_audio.constants import AUD_CONTEXT_TOKEN, AUD_END_TOKEN, AUD_START_TOKEN, AUD_TAG_TOKEN
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous

logger = logging.getLogger(__name__)


class VitaAudioDPOTrainer(VitaAudioRLSFTTrainer):
    """Trainer that reuses the VITA-Audio SFT preprocessing but optimizes a DPO objective."""

    _VALID_DPO_DATA_MODES = ("auto", "offline", "online")

    @staticmethod
    def _move_nested_tensors_to_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [VitaAudioDPOTrainer._move_nested_tensors_to_device(v, device) for v in value]
        return value

    def _move_audio_inputs_to_device(self, inputs: Dict[str, Any], device: torch.device) -> None:
        if "audios" in inputs:
            inputs["audios"] = self._move_nested_tensors_to_device(inputs["audios"], device)
        if "audio_indices" in inputs:
            inputs["audio_indices"] = self._move_nested_tensors_to_device(inputs["audio_indices"], device)

    def __init__(
        self,
        model,
        reward_funcs: Optional[List] = None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        data_collator=None,
        audio_tokenizer=None,
        audio_tokenizer_type: str = "glm4voice",
        flow_path: Optional[str] = None,
        model_variant: str = "balance",
        text_audio_interval_ratio: Optional[Tuple[int, ...]] = None,
        freeze_audio_components: bool = True,
        rl_loss_weight: float = 0.0,
        sft_loss_weight: float = 0.0,
        dpo_beta: float = 0.1,
        num_negative_samples: int = 1,
        negative_generation_kwargs: Optional[Dict] = None,
        dpo_token_type: str = "all",
        include_audio_boundaries: bool = True,
        dpo_data_mode: str = "auto",
    ) -> None:
        reward_funcs = reward_funcs or []
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            data_collator=data_collator,
            audio_tokenizer=audio_tokenizer,
            audio_tokenizer_type=audio_tokenizer_type,
            flow_path=flow_path,
            model_variant=model_variant,
            text_audio_interval_ratio=text_audio_interval_ratio,
            freeze_audio_components=freeze_audio_components,
            rl_loss_weight=rl_loss_weight,
            sft_loss_weight=sft_loss_weight,
        )

        self.dpo_beta = float(dpo_beta)
        self.num_negative_samples = max(1, int(num_negative_samples))
        self.negative_generation_kwargs = negative_generation_kwargs or {}
        self.ignore_token_id = -100
        self.dpo_token_type = (dpo_token_type or "all").lower()
        if self.dpo_token_type not in ("all", "text", "speech"):
            raise ValueError("dpo_token_type must be one of: all|text|speech")
        self.include_audio_boundaries = bool(include_audio_boundaries)

        dpo_data_mode = (dpo_data_mode or "auto").lower()
        if dpo_data_mode not in self._VALID_DPO_DATA_MODES:
            raise ValueError(f"dpo_data_mode must be one of: {', '.join(self._VALID_DPO_DATA_MODES)}")
        self.dpo_data_mode = dpo_data_mode

        self.speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
        self.begin_of_audio_id = self.processing_class.convert_tokens_to_ids("<|begin_of_audio|>")
        self.end_of_audio_id = self.processing_class.convert_tokens_to_ids("<|end_of_audio|>")

        if self.ref_model is None:
            try:
                self.ref_model = create_reference_model(self.model)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not create reference model automatically: %s", exc)
                self.ref_model = None

        self.dpo_generation_config = self._build_dpo_generation_config()

    @staticmethod
    def _extract_rejected_fields(sample: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        for source in (sample, sample.get("original_sample") or {}):
            if not isinstance(source, dict):
                continue
            rejected_text = source.get("rejected_text") or ""
            rejected_audio = source.get("rejected_audio")
            if rejected_text or rejected_audio:
                return str(rejected_text), rejected_audio
        return "", None

    def _build_candidate_inputs(
        self,
        sample: Dict[str, Any],
        device: torch.device,
        *,
        text: str,
        audio_path: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not (text or audio_path):
            return None
        candidate_sample = dict(sample)
        candidate_sample["sft_target_text"] = text
        candidate_sample["sft_target_audio"] = audio_path
        return self._build_chosen_inputs(candidate_sample, device)

    def _combine_sequence_inputs(self, seq_inputs_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
        if not seq_inputs_list:
            raise ValueError("seq_inputs_list must be non-empty")

        pad_id = int(self.processing_class.pad_token_id)
        max_len = max(int(inp["input_ids"].size(1)) for inp in seq_inputs_list)

        batch_input_ids: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        all_audios: List[Any] = []
        all_audio_indices: List[Any] = []

        for batch_idx, inp in enumerate(seq_inputs_list):
            input_ids = inp["input_ids"].to(device)
            labels = inp["labels"].to(device)

            if input_ids.size(0) != 1 or labels.size(0) != 1:
                raise ValueError("Expected each sequence inputs to have batch dimension 1.")

            seq_len = int(input_ids.size(1))
            if seq_len < max_len:
                pad_len = max_len - seq_len
                pad_tokens = torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device=device)
                input_ids = torch.cat([input_ids, pad_tokens], dim=1)

                pad_labels = torch.full(
                    (1, pad_len),
                    int(self.ignore_token_id),
                    dtype=labels.dtype,
                    device=device,
                )
                labels = torch.cat([labels, pad_labels], dim=1)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

            if "audios" in inp and inp["audios"] is not None:
                all_audios.extend(self._move_nested_tensors_to_device(inp["audios"], device))
            if "audio_indices" in inp and inp["audio_indices"] is not None:
                for idx_t in self._move_nested_tensors_to_device(inp["audio_indices"], device):
                    t = idx_t.clone()
                    t[0, :, :] = batch_idx
                    all_audio_indices.append(t)

        combined: Dict[str, Any] = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
        }
        if all_audios:
            combined["audios"] = all_audios
        if all_audio_indices:
            combined["audio_indices"] = all_audio_indices
        return combined

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        if return_outputs:
            raise ValueError("VitaAudioDPOTrainer does not return per-batch outputs.")

        if isinstance(inputs, dict):
            batch_samples = inputs.get("rl_samples") or inputs.get("sft_samples") or list(inputs.values())
        else:
            batch_samples = inputs

        if not isinstance(batch_samples, (list, tuple)):
            raise ValueError("Expected dataset samples to be a list of dicts.")

        device = model.device

        seq_inputs_list: List[Dict[str, Any]] = []
        pair_indices: List[Tuple[int, int]] = []

        for sample in batch_samples:
            prompt_inputs, prompt_length = self._prepare_prompt_inputs(sample, device)
            prompt_audio_inputs: Dict[str, Any] = {}
            if "audios" in prompt_inputs:
                prompt_audio_inputs["audios"] = prompt_inputs["audios"]
            if "audio_indices" in prompt_inputs:
                prompt_audio_inputs["audio_indices"] = prompt_inputs["audio_indices"]

            chosen_inputs = self._build_chosen_inputs(sample, device)
            if chosen_inputs is None:
                continue
            self._mask_prompt_labels(chosen_inputs, prompt_length)

            rejected_text, rejected_audio = self._extract_rejected_fields(sample)
            use_offline = self.dpo_data_mode in ("auto", "offline") and (rejected_text or rejected_audio)
            if self.dpo_data_mode == "online":
                use_offline = False

            if use_offline:
                rejected_inputs = self._build_candidate_inputs(
                    sample,
                    device,
                    text=rejected_text,
                    audio_path=rejected_audio,
                )
                if rejected_inputs is None:
                    continue
                self._mask_prompt_labels(rejected_inputs, prompt_length)

                chosen_idx = len(seq_inputs_list)
                seq_inputs_list.append(chosen_inputs)
                rejected_idx = len(seq_inputs_list)
                seq_inputs_list.append(rejected_inputs)
                pair_indices.append((chosen_idx, rejected_idx))
                continue

            if self.dpo_data_mode == "offline":
                continue

            rejected_sequences = self._sample_rejected_sequences(model, prompt_inputs)
            if not rejected_sequences:
                continue

            chosen_idx = len(seq_inputs_list)
            seq_inputs_list.append(chosen_inputs)
            for seq in rejected_sequences:
                rejected_inputs = self._build_rejected_inputs(seq, prompt_length, device, prompt_audio_inputs)
                rejected_idx = len(seq_inputs_list)
                seq_inputs_list.append(rejected_inputs)
                pair_indices.append((chosen_idx, rejected_idx))

        if not seq_inputs_list or not pair_indices:
            return torch.tensor(0.0, device=device, requires_grad=True)

        combined_inputs = self._combine_sequence_inputs(seq_inputs_list, device)
        policy_logps, policy_counts = self._sequence_logps(model, combined_inputs)

        ref_logps: Optional[torch.Tensor] = None
        if self.ref_model is not None:
            self._ensure_ref_model_device(device)
            self.ref_model.train()
            with torch.no_grad():
                ref_logps, _ = self._sequence_logps(self.ref_model, combined_inputs)

        losses: List[torch.Tensor] = []
        policy_pos_logps: List[torch.Tensor] = []
        policy_neg_logps: List[torch.Tensor] = []
        ref_pos_logps: List[torch.Tensor] = []
        ref_neg_logps: List[torch.Tensor] = []
        for chosen_idx, rejected_idx in pair_indices:
            chosen_policy_logp = policy_logps[chosen_idx]
            rejected_policy_logp = policy_logps[rejected_idx]
            chosen_count = policy_counts[chosen_idx]
            rejected_count = policy_counts[rejected_idx]

            if self.dpo_token_type != "all" and (chosen_count.item() == 0 or rejected_count.item() == 0):
                continue

            chosen_ref_logp = ref_logps[chosen_idx] if ref_logps is not None else None
            rejected_ref_logp = ref_logps[rejected_idx] if ref_logps is not None else None

            policy_delta = chosen_policy_logp - rejected_policy_logp
            ref_delta = 0.0
            if chosen_ref_logp is not None and rejected_ref_logp is not None:
                ref_delta = chosen_ref_logp - rejected_ref_logp

            logits = self.dpo_beta * (policy_delta - ref_delta)
            loss = -F.logsigmoid(logits)
            losses.append(loss)
            policy_pos_logps.append(chosen_policy_logp.detach())
            policy_neg_logps.append(rejected_policy_logp.detach())
            if chosen_ref_logp is not None:
                ref_pos_logps.append(chosen_ref_logp.detach())
            if rejected_ref_logp is not None:
                ref_neg_logps.append(rejected_ref_logp.detach())

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        batch_loss = torch.stack(losses).mean()
        self._record_metrics(batch_loss, policy_pos_logps, policy_neg_logps, ref_pos_logps, ref_neg_logps)
        return batch_loss

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _record_metrics(
        self,
        batch_loss: torch.Tensor,
        policy_pos_logps: List[torch.Tensor],
        policy_neg_logps: List[torch.Tensor],
        ref_pos_logps: List[torch.Tensor],
        ref_neg_logps: List[torch.Tensor],
    ) -> None:
        loss_avg = self.accelerator.gather_for_metrics(batch_loss.detach()).mean().item()
        self._metrics["dpo/loss"].append(loss_avg)
        self._metrics["dpo/pairs"].append(float(len(policy_neg_logps)))

        if policy_pos_logps:
            pos_val = torch.stack(policy_pos_logps).mean()
            self._metrics["dpo/chosen_logp"].append(
                self.accelerator.gather_for_metrics(pos_val).mean().item()
            )
        if policy_neg_logps:
            neg_val = torch.stack(policy_neg_logps).mean()
            self._metrics["dpo/rejected_logp"].append(
                self.accelerator.gather_for_metrics(neg_val).mean().item()
            )
        if ref_pos_logps:
            ref_pos_val = torch.stack(ref_pos_logps).mean()
            self._metrics["dpo/ref_chosen_logp"].append(
                self.accelerator.gather_for_metrics(ref_pos_val).mean().item()
            )
        if ref_neg_logps:
            ref_neg_val = torch.stack(ref_neg_logps).mean()
            self._metrics["dpo/ref_rejected_logp"].append(
                self.accelerator.gather_for_metrics(ref_neg_val).mean().item()
            )

    def _build_dpo_generation_config(self) -> GenerationConfig:
        base_config = copy.deepcopy(self.generation_config) if self.generation_config else GenerationConfig()
        base_config.do_sample = True
        base_config.num_return_sequences = self.num_negative_samples
        base_config.max_new_tokens = self.negative_generation_kwargs.get(
            "max_new_tokens", getattr(base_config, "max_new_tokens", 256)
        )
        base_config.temperature = self.negative_generation_kwargs.get(
            "temperature", getattr(base_config, "temperature", 0.7)
        )
        base_config.top_p = self.negative_generation_kwargs.get(
            "top_p", getattr(base_config, "top_p", 0.9)
        )
        base_config.top_k = self.negative_generation_kwargs.get(
            "top_k", getattr(base_config, "top_k", 0)
        )
        base_config.repetition_penalty = self.negative_generation_kwargs.get(
            "repetition_penalty", getattr(base_config, "repetition_penalty", 1.0)
        )
        base_config.pad_token_id = self.processing_class.pad_token_id
        base_config.eos_token_id = self.processing_class.eos_token_id
        return base_config

    def _prepare_prompt_inputs(self, sample: Dict, device: torch.device) -> Tuple[Dict, int]:
        messages = copy.deepcopy(sample.get("messages", []))
        audios = list(sample.get("audios", []))
        processed_messages = self._prepare_s2s_messages(messages, audios)

        prompt_input_ids = self.processing_class.apply_chat_template(
            processed_messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        audios_processed = None
        audio_indices = None

        if (
            audios
            and self.audio_tokenizer
            and self.audio_tokenizer.apply_to_role("user", is_contiguous=True)
        ):
            prompt_input_ids, audios_processed, audio_indices = add_audio_input_contiguous(
                prompt_input_ids, audios, self.processing_class, self.audio_tokenizer
            )
        elif (
            audios
            and self.audio_tokenizer
            and self.audio_tokenizer.apply_to_role("user", is_discrete=True)
        ):
            processed_messages = self._process_discrete_audio_tokens(processed_messages, audios)
            prompt_input_ids = self.processing_class.apply_chat_template(
                processed_messages,
                tokenize=True,
                add_generation_prompt=self.add_generation_prompt,
            )

        inputs = {"input_ids": torch.tensor([prompt_input_ids], dtype=torch.long, device=device)}
        if audios_processed is not None:
            inputs["audios"] = audios_processed
            inputs["audio_indices"] = audio_indices
            self._move_audio_inputs_to_device(inputs, device)
        return inputs, inputs["input_ids"].size(1)

    def _build_chosen_inputs(self, sample: Dict, device: torch.device) -> Optional[Dict]:
        sft_target_text = sample.get("sft_target_text", "")
        sft_target_audio_path = sample.get("sft_target_audio")
        if not (sft_target_text or sft_target_audio_path):
            return None

        messages = copy.deepcopy(sample.get("messages", []))
        audios = list(sample.get("audios", []))

        target_content = ""
        if sft_target_text:
            target_content += sft_target_text
        if sft_target_audio_path:
            target_content = f"{target_content}\n<|audio|>" if target_content else "<|audio|>"
            audios.append(sft_target_audio_path)
        messages.append({"role": "assistant", "content": target_content})

        contiguous_audio_idxs: List[int] = []

        if audios and self.audio_tokenizer and self.audio_tokenizer.is_discrete:
            audio_tokens: List[List[int]] = []
            sample_id = sample.get("id", "<unknown>")
            for audio_path in audios:
                if not audio_path:
                    audio_tokens.append([])
                    continue
                audio_path_str = str(audio_path)
                if not os.path.exists(audio_path_str):
                    logger.warning(
                        "Skipping sample id=%s because audio file is missing: %s",
                        sample_id,
                        audio_path_str,
                    )
                    return None
                try:
                    tokens = self.audio_tokenizer.encode(
                        audio_path_str,
                        is_discrete=True,
                        is_contiguous=False,
                    )
                except Exception as exc:  # pragma: no cover - best effort for corrupted audio
                    logger.warning(
                        "Skipping sample id=%s because audio tokenization failed for %s: %s",
                        sample_id,
                        audio_path_str,
                        exc,
                    )
                    return None
                audio_tokens.append(tokens)
            audio_tokens_list = ["".join(f"<|audio_{i}|>" for i in tokens) for tokens in audio_tokens]

            audio_idx = 0
            for sentence in messages:
                content = sentence["content"]
                role = sentence["role"]
                if self.audio_tokenizer.apply_to_role(role, is_discrete=True):
                    while AUD_TAG_TOKEN in content and audio_idx < len(audio_tokens_list):
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

        processed_messages = self._prepare_s2s_messages(messages, audios)

        audios_processed: List[torch.Tensor] = []
        audio_indices_processed: List[torch.Tensor] = []
        input_ids: List[int] = []
        targets: List[int] = []

        tokenizer = self.processing_class
        AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids[0]
        AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids[0]
        AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids[0]
        AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids[0]

        IM_START = "<|im_start|>"
        IM_END = "<|im_end|>"
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
        IM_START_IDS = tokenizer(IM_START, add_special_tokens=False).input_ids
        IM_END_IDS = tokenizer(IM_END, add_special_tokens=False).input_ids
        USER_IDS = tokenizer(USER, add_special_tokens=False).input_ids
        ASSISTANT_IDS = tokenizer(ASSISTANT, add_special_tokens=False).input_ids
        SYSTEM_IDS = tokenizer(SYSTEM, add_special_tokens=False).input_ids

        for sentence in processed_messages:
            role = sentence["role"]
            content = sentence["content"]

            if role == "user":
                _input = (
                    IM_START_IDS + USER_IDS + nl_tokens
                    + tokenizer(content, add_special_tokens=False).input_ids
                    + IM_END_IDS + nl_tokens
                )
                _target = [self.ignore_token_id] * len(_input)
            elif role == "assistant":
                content_input_ids = tokenizer(content, add_special_tokens=False).input_ids
                if self.audio_tokenizer is not None:
                    content_input_ids = self._apply_text_audio_interval(
                        content_input_ids, AUD_START_ID, AUD_END_ID
                    )
                _input = (
                    IM_START_IDS + ASSISTANT_IDS + nl_tokens
                    + content_input_ids + IM_END_IDS + nl_tokens
                )
                _target = (
                    [self.ignore_token_id] * (len(IM_START_IDS) + len(ASSISTANT_IDS) + len(nl_tokens))
                    + content_input_ids + IM_END_IDS + nl_tokens
                )
            else:  # system
                _input = (
                    IM_START_IDS + SYSTEM_IDS + nl_tokens
                    + tokenizer(content, add_special_tokens=False).input_ids
                    + IM_END_IDS + nl_tokens
                )
                _target = [self.ignore_token_id] * len(_input)

            input_ids += _input
            targets += _target

        if contiguous_audio_idxs and self.audio_tokenizer and self.audio_tokenizer.is_contiguous:
            aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
            if aud_positions:
                new_input_ids: List[int] = []
                new_targets: List[int] = []
                cursor = 0
                for pos_idx, aud_pos in enumerate(aud_positions):
                    if pos_idx >= len(contiguous_audio_idxs):
                        break
                    actual_audio_idx = contiguous_audio_idxs[pos_idx]
                    audio = self.audio_tokenizer.encode(
                        audios[actual_audio_idx], is_contiguous=True
                    )
                    audios_processed.append(audio)
                    audio_token_length = audio.size(0) + 4

                    new_input_ids += input_ids[cursor:aud_pos]
                    new_targets += targets[cursor:aud_pos]

                    new_input_ids += [AUD_START_ID]
                    new_targets += [self.ignore_token_id]

                    audio_indice_b = torch.zeros(1, audio_token_length, dtype=torch.int64)
                    audio_indice_s = torch.arange(
                        len(new_input_ids), len(new_input_ids) + audio_token_length
                    ).unsqueeze(0)
                    audio_indice = torch.stack([audio_indice_b, audio_indice_s], dim=0)
                    audio_indices_processed.append(audio_indice)

                    new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
                    new_targets += [self.ignore_token_id] * audio_token_length

                    new_input_ids += [AUD_END_ID]
                    new_targets += [self.ignore_token_id]
                    cursor = aud_pos + 1

                new_input_ids += input_ids[cursor:]
                new_targets += targets[cursor:]
                input_ids = new_input_ids
                targets = new_targets

        inputs_dict = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "labels": torch.tensor([targets], dtype=torch.long, device=device),
        }
        if audios_processed:
            inputs_dict["audios"] = audios_processed
        if audio_indices_processed:
            inputs_dict["audio_indices"] = audio_indices_processed
        self._move_audio_inputs_to_device(inputs_dict, device)
        return inputs_dict

    def _build_rejected_inputs(
        self,
        sequence: torch.Tensor,
        prompt_length: int,
        device: torch.device,
        prompt_audio_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        seq = sequence.to(device)
        labels = seq.clone()
        labels[:, :prompt_length] = self.ignore_token_id
        labels[seq == self.processing_class.pad_token_id] = self.ignore_token_id
        rejected_inputs: Dict[str, Any] = {"input_ids": seq, "labels": labels}

        if prompt_audio_inputs:
            if "audios" in prompt_audio_inputs:
                rejected_inputs["audios"] = prompt_audio_inputs["audios"]
            if "audio_indices" in prompt_audio_inputs:
                rejected_inputs["audio_indices"] = prompt_audio_inputs["audio_indices"]
        self._move_audio_inputs_to_device(rejected_inputs, device)

        return rejected_inputs

    def _mask_prompt_labels(self, inputs: Dict[str, Any], prompt_length: int) -> None:
        labels = inputs.get("labels")
        if not torch.is_tensor(labels) or labels.dim() != 2:
            return
        cutoff = min(int(prompt_length), labels.size(1))
        if cutoff > 0:
            labels[:, :cutoff] = self.ignore_token_id
        input_ids = inputs.get("input_ids")
        if torch.is_tensor(input_ids) and self.processing_class.pad_token_id is not None:
            labels[input_ids == self.processing_class.pad_token_id] = self.ignore_token_id

    def _sequence_logps(self, model, seq_inputs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = seq_inputs["input_ids"]
        labels = seq_inputs["labels"]
        attention_mask = (input_ids != self.processing_class.pad_token_id).long()
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "use_cache": False,
        }

        if "audios" in seq_inputs:
            model_kwargs["audios"] = self._move_nested_tensors_to_device(seq_inputs["audios"], input_ids.device)
        if "audio_indices" in seq_inputs:
            model_kwargs["audio_indices"] = self._move_nested_tensors_to_device(
                seq_inputs["audio_indices"], input_ids.device
            )

        outputs = model(**model_kwargs)
        logits = outputs.logits
        logps, counts = self._select_label_logps(
            logits,
            labels,
            token_type=self.dpo_token_type,
            include_audio_boundaries=self.include_audio_boundaries,
        )
        return logps.squeeze(-1), counts

    def _reference_logp(self, seq_inputs: Dict) -> Optional[torch.Tensor]:
        if self.ref_model is None:
            return None
        device = seq_inputs["input_ids"].device
        self._ensure_ref_model_device(device)
        self.ref_model.train()
        with torch.no_grad():
            logp, _ = self._sequence_logps(self.ref_model, seq_inputs)
        return logp.detach()

    def _select_label_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_type: str = "all",
        include_audio_boundaries: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != self.ignore_token_id

        if token_type == "all":
            allowed_mask = valid_mask
        else:
            speech_token_offset = int(self.speech_token_offset)
            vocab_size = getattr(self.model.config, "vocab_size", shift_logits.size(-1))
            speech_mask = (shift_labels >= speech_token_offset) & (shift_labels < vocab_size)
            text_mask = shift_labels < speech_token_offset

            if include_audio_boundaries:
                begin_of_audio_id = int(self.begin_of_audio_id)
                end_of_audio_id = int(self.end_of_audio_id)
                speech_mask = speech_mask | (shift_labels == begin_of_audio_id) | (shift_labels == end_of_audio_id)
                text_mask = text_mask & (shift_labels != begin_of_audio_id) & (shift_labels != end_of_audio_id)

            speech_mask = speech_mask & valid_mask
            text_mask = text_mask & valid_mask
            allowed_mask = speech_mask if token_type == "speech" else text_mask

        safe_labels = shift_labels.masked_fill(~allowed_mask, 0)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * allowed_mask

        token_count = allowed_mask.sum(dim=-1)
        seq_logps = token_logps.sum(dim=-1)
        return seq_logps.unsqueeze(-1), token_count

    def _ensure_ref_model_device(self, device: torch.device) -> None:
        if self.ref_model is None:
            return
        current_device = next(self.ref_model.parameters()).device
        if current_device != device:
            self.ref_model = self.ref_model.to(device)

    def _sample_rejected_sequences(self, model, prompt_inputs: Dict) -> List[torch.Tensor]:
        prompt_inputs = {
            k: v.to(model.device) if torch.is_tensor(v) else v for k, v in prompt_inputs.items()
        }
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped, torch.no_grad():
            sequences = self._generate_sequences_for_variant(
                generation_model=unwrapped,
                prompt_inputs=prompt_inputs,
                generation_config=self.dpo_generation_config,
                num_generations=self.num_negative_samples,
            )
        return [sequences[i : i + 1] for i in range(sequences.size(0))]
