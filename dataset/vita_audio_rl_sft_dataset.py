"""Generic dataset utilities for WavAlign post-training."""

from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_from_disk
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VitaAudioRLSFTDataset(Dataset):
    """Loads the JSONL/HF dataset schema used by the released training code."""

    def __init__(
        self,
        dataset_path: str,
        task_types: Optional[List[str]] = None,
        use_luke_system: bool = True,
        force_s2s: bool = False,
        audio_base_path: Optional[str] = None,
        validate_audio: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path
        self.task_types = set(task_types or ["s2s"])
        self.use_luke_system = use_luke_system
        self.force_s2s = force_s2s
        self.audio_base_path = Path(audio_base_path or Path(dataset_path).resolve().parent)
        self.validate_audio = validate_audio
        self.max_samples = max_samples

        random.seed(seed)

        self.luke_system_message = {
            "role": "system",
            "content": "You are Luke, the voice AI assistant. You can speak and listen.",
        }

        self.data = self._process_samples(self._load_dataset())
        logger.info("Loaded %d samples from %s", len(self.data), self.dataset_path)

    def _load_dataset(self) -> List[Dict[str, Any]]:
        path = Path(self.dataset_path)
        if path.is_dir():
            dataset = load_from_disk(str(path))
            records = [dict(sample) for sample in dataset]
        elif path.suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("Expected a list when loading JSON dataset.")
            records = payload

        if self.max_samples and len(records) > self.max_samples:
            records = random.sample(records, self.max_samples)
        return records

    def _process_samples(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for sample in raw_data:
            item = self._process_single_sample(sample)
            if item is None:
                continue
            task_type = item.get("task_type", "unknown")
            if self.task_types and task_type not in self.task_types:
                continue
            processed.append(item)
        return processed

    def _process_single_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "messages" in sample:
            return self._process_release_format(sample)
        return self._process_legacy_format(sample)

    def _process_release_format(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = deepcopy(sample.get("messages") or [])
        if not isinstance(messages, list) or not messages:
            return None

        audios = self._resolve_audio_list(sample.get("audios") or [])
        sft_target_text = self._clean_text(sample.get("sft_target_text"))
        sft_target_audio = self._resolve_audio_path(sample.get("sft_target_audio"))
        rejected_audio = self._resolve_audio_path(sample.get("rejected_audio"))
        orig_sft_target_audio = self._resolve_audio_path(sample.get("orig_sft_target_audio"))

        prompt_messages = deepcopy(messages)
        while prompt_messages and prompt_messages[-1].get("role") == "assistant":
            prompt_messages.pop()

        if self.use_luke_system and (
            not prompt_messages or prompt_messages[0].get("role") != "system"
        ):
            prompt_messages = [self.luke_system_message] + prompt_messages

        if not any(msg.get("role") == "user" for msg in prompt_messages):
            return None

        if self.validate_audio:
            audios = self._validate_audio_files(audios)
            if sft_target_audio and not os.path.exists(sft_target_audio):
                sft_target_audio = None
            if rejected_audio and not os.path.exists(rejected_audio):
                rejected_audio = None
            if orig_sft_target_audio and not os.path.exists(orig_sft_target_audio):
                orig_sft_target_audio = None

        question_text = self._clean_text(sample.get("question_text")) or self._derive_question_text(
            prompt_messages
        )
        history_text = self._clean_text(sample.get("history_text"))

        processed = {
            "messages": prompt_messages,
            "audios": audios,
            "sft_target_text": sft_target_text,
            "sft_target_audio": sft_target_audio,
            "question_text": question_text,
            "question_text_raw": self._clean_text(sample.get("question_text_raw")),
            "history_text": history_text,
            "task_type": sample.get("task_type")
            or self._determine_task_type(prompt_messages, audios, sft_target_text, sft_target_audio),
            "original_sample": sample,
        }
        for key in (
            "rejected_text",
            "chosen_score",
            "rejected_score",
            "score_gap",
            "chosen_key",
            "rejected_key",
            "source_dataset",
            "id",
        ):
            if key in sample:
                processed[key] = sample[key]
        if rejected_audio:
            processed["rejected_audio"] = rejected_audio
        if orig_sft_target_audio:
            processed["orig_sft_target_audio"] = orig_sft_target_audio
        if "orig_sft_target_text" in sample:
            processed["orig_sft_target_text"] = sample["orig_sft_target_text"]
        return processed

    def _process_legacy_format(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question_text = self._clean_text(sample.get("question_text"))
        history_text = self._extract_history_text(sample.get("history") or sample.get("history_text"))
        answer_text = self._clean_text(sample.get("answer_text") or sample.get("sft_target_text"))

        question_audio = self._resolve_audio_path(sample.get("question_audio"))
        answer_audio = self._resolve_audio_path(sample.get("answer_audio_path") or sample.get("sft_target_audio"))

        user_segments = [segment for segment in [history_text, question_text] if segment]
        audios: List[str] = []
        if self.force_s2s or question_audio:
            user_segments.append("<|audio|>")
            if question_audio:
                audios.append(question_audio)
        user_content = "\n\n".join(user_segments)
        if not user_content:
            return None

        messages = [{"role": "user", "content": user_content}]
        if self.use_luke_system:
            messages = [self.luke_system_message] + messages

        if self.validate_audio:
            audios = self._validate_audio_files(audios)
            if answer_audio and not os.path.exists(answer_audio):
                answer_audio = None

        return {
            "messages": messages,
            "audios": audios,
            "sft_target_text": answer_text,
            "sft_target_audio": answer_audio,
            "question_text": "\n\n".join([segment for segment in [history_text, question_text] if segment]),
            "history_text": history_text,
            "task_type": self._determine_task_type(messages, audios, answer_text, answer_audio),
            "original_sample": sample,
        }

    def _resolve_audio_list(self, audio_paths: List[Any]) -> List[str]:
        resolved = []
        for item in audio_paths:
            resolved_path = self._resolve_audio_path(item)
            if resolved_path:
                resolved.append(resolved_path)
        return resolved

    def _resolve_audio_path(self, audio_path: Any) -> Optional[str]:
        if not audio_path:
            return None
        path = str(audio_path).strip()
        if not path:
            return None
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str((self.audio_base_path / path_obj).resolve())

    def _validate_audio_files(self, audio_paths: List[str]) -> List[str]:
        valid: List[str] = []
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                valid.append(audio_path)
            else:
                logger.warning("Audio file not found: %s", audio_path)
        return valid

    @staticmethod
    def _clean_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _derive_question_text(messages: List[Dict[str, Any]]) -> str:
        user_segments: List[str] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", "")).replace("<|audio|>", "").strip()
            if content:
                user_segments.append(content)
        return "\n\n".join(user_segments)

    @staticmethod
    def _determine_task_type(
        messages: List[Dict[str, Any]],
        audios: List[str],
        sft_target_text: Optional[str] = None,
        sft_target_audio: Optional[str] = None,
    ) -> str:
        has_input_audio = bool(audios) or any(
            "<|audio|>" in str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"
        )
        has_output_audio = bool(sft_target_audio)
        has_output_text = bool(sft_target_text and sft_target_text.strip())

        if has_input_audio and has_output_audio and has_output_text:
            return "s2s"
        if has_input_audio and has_output_text and not has_output_audio:
            return "asr"
        if not has_input_audio and has_output_audio:
            return "tts"
        if has_input_audio and has_output_text:
            return "sqa"
        if has_output_text:
            return "text"
        return "unknown"

    @staticmethod
    def _extract_history_text(history: Optional[str]) -> str:
        if not history:
            return ""
        history = history.strip()
        if not history:
            return ""
        if "[USER]" not in history or "[ASSISTANT]" not in history:
            return history

        turns: List[str] = []
        parts = history.split("[USER]")[1:]
        for part in parts:
            if "[ASSISTANT]" in part:
                user_part, assistant_part = part.split("[ASSISTANT]", 1)
                user_text = user_part.strip()
                assistant_text = assistant_part.split("[USER]")[0].strip()
                if user_text:
                    turns.append(f"User: {user_text}")
                if assistant_text:
                    turns.append(f"Assistant: {assistant_text}")
            else:
                user_text = part.strip()
                if user_text:
                    turns.append(f"User: {user_text}")
        return "\n".join(turns)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx].copy()


def collate_fn_rl_sft(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    rl_samples = list(batch)
    sft_samples = [
        sample for sample in batch if sample.get("sft_target_text") or sample.get("sft_target_audio")
    ]
    return {
        "rl_samples": rl_samples,
        "sft_samples": sft_samples,
        "batch_size": len(batch),
        "num_rl_samples": len(rl_samples),
        "num_sft_samples": len(sft_samples),
    }


def collate_fn_simple(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch
