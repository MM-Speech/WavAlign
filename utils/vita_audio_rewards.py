"""Reward model wrappers for WavAlign RL training.

The reward client expects an OpenAI-compatible multimodal chat endpoint.
Configure it with environment variables instead of hard-coded credentials:

- `WAVALIGN_REWARD_API_KEY`
- `WAVALIGN_REWARD_API_BASE`  # full request URL, despite the historical name
- `WAVALIGN_REWARD_MODEL`

Legacy aliases are also accepted:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from io import BytesIO
from typing import List, Optional

import requests
import torch
from scipy.io.wavfile import write as write_wav

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """
**Task: Evaluate the Quality of a Spoken Answer**

You will be provided with a spoken question and a spoken answer. Your goal is to assess the quality of the answer.

Score from 1 to 5 and return:
<justification>...</justification>
<score>...</score>
"""

ACOUSTIC_EVALUATION_PROMPT = """
**Task: Evaluate Paralinguistic Quality of the Spoken Voice**

Judge clarity, fluency, accent, emotion, pacing, and listening comfort, while ignoring semantic correctness.

Score from 1 to 5 and return:
<justification>...</justification>
<score>...</score>
"""

SEMANTIC_EVALUATION_PROMPT = """
**Task: Evaluate Semantic Quality of the Spoken Answer**

Judge only the content quality and ignore the acoustic quality.

Score from 1 to 5 and return:
<justification>...</justification>
<score>...</score>
"""


class GPT4oRewardFunction:
    """Backward-compatible reward wrapper around an OpenAI-style chat API."""

    _PROMPTS = {
        "holistic": EVALUATION_PROMPT,
        "acoustic": ACOUSTIC_EVALUATION_PROMPT,
        "semantic": SEMANTIC_EVALUATION_PROMPT,
    }

    def __init__(
        self,
        evaluation_type: str = "holistic",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
        retry_delay: int = 5,
        default_score: float = 2.5,
    ) -> None:
        evaluation_type = (evaluation_type or "holistic").lower()
        if evaluation_type not in self._PROMPTS:
            raise ValueError(f"Invalid evaluation_type: {evaluation_type}")

        self.evaluation_type = evaluation_type
        self.evaluation_prompt = self._PROMPTS[evaluation_type]
        self.api_key = api_key or os.getenv("WAVALIGN_REWARD_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_endpoint = (
            api_endpoint
            or os.getenv("WAVALIGN_REWARD_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1/chat/completions"
        )
        self.model_name = (
            model_name
            or os.getenv("WAVALIGN_REWARD_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_score = default_score
        self.__name__ = f"{evaluation_type}_reward"

        if not self.api_key:
            raise ValueError(
                "Missing reward API key. Set WAVALIGN_REWARD_API_KEY (or OPENAI_API_KEY)."
            )

    @staticmethod
    def _encode_audio_tensor_base64(
        audio_tensor: Optional[torch.Tensor], sampling_rate: int = 24000
    ) -> Optional[str]:
        if audio_tensor is None or not torch.is_tensor(audio_tensor):
            return None
        try:
            audio_tensor = audio_tensor.detach().cpu().float()
            if audio_tensor.dim() == 2 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            peak = torch.max(torch.abs(audio_tensor))
            if peak > 1.0:
                audio_tensor = audio_tensor / peak
            audio_int16 = (audio_tensor.numpy() * 32767).astype("int16")
            buffer = BytesIO()
            write_wav(buffer, sampling_rate, audio_int16)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to encode audio for reward model: %s", exc)
            return None

    def _build_payload(
        self,
        question_text: str,
        completion_text: str,
        answer_audio_tensor: Optional[torch.Tensor],
    ) -> dict:
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Question:\n{question_text}\n\n"
                    f"Transcript of answer:\n{completion_text or '[empty]'}"
                ),
            }
        ]
        audio_b64 = self._encode_audio_tensor_base64(answer_audio_tensor)
        if audio_b64:
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            )

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.evaluation_prompt},
                {"role": "user", "content": user_content},
            ],
        }

    def _request_score(
        self,
        question_text: str,
        completion_text: str,
        answer_audio_tensor: Optional[torch.Tensor],
    ) -> float:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(question_text, completion_text, answer_audio_tensor)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                match = re.search(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", content)
                if match:
                    return float(match.group(1))
                logger.warning("Reward response missing <score>: %s", content[:200])
            except Exception as exc:  # pragma: no cover - external service
                logger.warning("Reward request failed on attempt %d/%d: %s", attempt + 1, self.max_retries, exc)
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_delay)

        return self.default_score

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        audios: Optional[List[Optional[torch.Tensor]]] = None,
        **_: object,
    ) -> List[float]:
        audios = audios or [None] * len(prompts)
        scores: List[float] = []
        for question_text, completion_text, audio_tensor in zip(prompts, completions, audios):
            scores.append(
                self._request_score(
                    question_text=question_text or "",
                    completion_text=completion_text or "",
                    answer_audio_tensor=audio_tensor,
                )
            )
        return scores


def create_vita_audio_reward_functions(use_single_reward: bool = True) -> List[GPT4oRewardFunction]:
    if use_single_reward:
        return [GPT4oRewardFunction(evaluation_type="holistic")]
    return [
        GPT4oRewardFunction(evaluation_type="semantic"),
        GPT4oRewardFunction(evaluation_type="acoustic"),
    ]
