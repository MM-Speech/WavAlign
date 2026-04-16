# WavAlign

<p align="center">
  <img src="assets/method.png" alt="WavAlign method overview" width="88%">
</p>

<p align="center">
  <a href="https://speechrl.github.io/">Project Page</a> |
  <a href="#code-release">Code Release</a> |
  <a href="#citation">Citation</a>
</p>

WavAlign is a post-training recipe for end-to-end spoken dialogue models that improves semantic intelligence while preserving speech naturalness and expressiveness.

Our paper **"WavAlign: Enhancing Intelligence and Expressiveness in Spoken Dialogue Models via Adaptive Hybrid Post-Training"** has been accepted to **Findings of ACL 2026**.

Project page: https://speechrl.github.io/

## News

- `2026-04` Initial code release for the WavAlign training pipeline.
- `2026-04` Project homepage is live at https://speechrl.github.io/
- `Coming soon` Model checkpoints.
- `Coming soon` Training and evaluation datasets.

## Overview

WavAlign is built around a simple principle from the paper:

- Use preference optimization where the signal is most reliable: the semantic text channel.
- Keep speech generation anchored with supervised targets to avoid acoustic drift.
- Support both online RL-style optimization and offline DPO-style optimization under the same mixed text-speech setup.

This repository currently releases the **core post-training code** used for:

- masked RL + SFT training for spoken dialogue models
- text-token / speech-token masking controls
- offline and online DPO training
- DPO pair construction from scored multi-sample generations

The release intentionally excludes:

- model weights
- full training datasets
- private data preprocessing pipelines tied to internal storage layouts
- experiment logs, caches, and analysis artifacts

## Code Release

```text
WavAlign/
├── assets/                        # README assets
├── config/                        # DeepSpeed templates
├── dataset/                       # Generic JSONL/HF dataset loader
├── dpo/                           # DPO trainer
├── examples/                      # Minimal schema examples
├── scripts/                       # Launch scripts
├── trainer/                       # RL+SFT trainers
├── utils/                         # Reward model + DPO pair builder
├── train_vita_audio_rl_sft_masked.py
└── train_vita_audio_dpo.py
```

## Installation

This code depends on a local checkout of the upstream `VITA-Audio` codebase.

```bash
git clone <this-repo>
cd WavAlign
pip install -r requirements.txt
export VITA_AUDIO_ROOT=/path/to/VITA-Audio
```

For RL training with API-based reward scoring, also set:

```bash
export WAVALIGN_REWARD_API_KEY=...
export WAVALIGN_REWARD_API_BASE=...
export WAVALIGN_REWARD_MODEL=...
```

The reward client expects an OpenAI-compatible multimodal chat endpoint.

## Data Format

The public release uses a simple JSONL schema. See `examples/sample_rl_sft.jsonl` and `examples/sample_dpo.jsonl`.

RL + SFT training sample:

```json
{
  "messages": [
    {"role": "system", "content": "You are Luke, the voice AI assistant. You can speak and listen."},
    {"role": "user", "content": "...\n\n<|audio|>"}
  ],
  "audios": ["audio/example_question.wav"],
  "sft_target_text": "text target",
  "sft_target_audio": "audio/example_answer.wav",
  "task_type": "s2s",
  "question_text": "plain-text prompt for reward evaluation"
}
```

DPO training adds:

```json
{
  "rejected_text": "worse response",
  "rejected_audio": "audio/example_bad.wav"
}
```

## Quick Start

Masked RL + SFT:

```bash
export VITA_AUDIO_ROOT=/path/to/VITA-Audio
export WAVALIGN_REWARD_API_KEY=...
bash scripts/train_rl_sft_masked.sh plus-vanilla
```

Offline DPO:

```bash
export VITA_AUDIO_ROOT=/path/to/VITA-Audio
bash scripts/train_dpo.sh plus-vanilla
```

Build DPO pairs from scored candidates:

```bash
python utils/dpo_pair_builder.py \
  --input_path scored_generations.json \
  --output_path dpo_pairs.jsonl \
  --chosen_source best_output \
  --rejected_source worst_output \
  --score_mode sum
```

## Notes

- The current release focuses on the training recipe and trainer implementation.
- Project page, paper metadata, and future artifact updates will be maintained at https://speechrl.github.io/
- Checkpoints and datasets will be added in a later release.

