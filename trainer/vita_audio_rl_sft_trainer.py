"""Compatibility shim for VITA-Audio RL+SFT trainer.

This module now re-exports the canonical implementation from
`trainer.vita_audio_rl_sft_trainer_clean` to guarantee we always run
with the original VitaAudio loss computation (text+audio handling,
SFT preprocessing, etc.).  The previous lightweight copy diverged
from upstream and could skip SFT loss; keeping a single source of truth
prevents regressions.
"""

from .vita_audio_rl_sft_trainer_clean import VitaAudioRLSFTTrainer

__all__ = ["VitaAudioRLSFTTrainer"]
