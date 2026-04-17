#!/usr/bin/env python3
"""VITA-Audio RL+SFT training with token-type masking controls."""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from transformers import HfArgumentParser, set_seed
from trl import GRPOConfig

from dataset.vita_audio_rl_sft_dataset import VitaAudioRLSFTDataset, collate_fn_simple
from trainer.vita_audio_rl_sft_trainer_masked import VitaAudioRLSFTTrainerMasked
from utils.import_utils import ensure_vita_audio_importable

ensure_vita_audio_importable()
from vita_audio.tokenizer import get_audio_tokenizer

logger = logging.getLogger(__name__)

try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
except ImportError:  # pragma: no cover - optional dependency
    LossScaler = None


if hasattr(torch.serialization, "add_safe_globals") and LossScaler is not None:
    torch.serialization.add_safe_globals([LossScaler])
    print(
        "Registered deepspeed.runtime.fp16.loss_scaler.LossScaler as a safe global for torch serialization."
    )


@dataclass
class VitaAudioRLSFTMaskedArguments:
    """CLI arguments mirroring the mixed RL script plus masking toggles."""

    model_name_or_path: str = field(metadata={"help": "Checkpoint to finetune."})
    model_variant: str = field(metadata={"help": "VITA-Audio variant: balance|boost|plus-vanilla."})
    audio_tokenizer_path: str = field(metadata={"help": "Path to audio tokenizer."})
    audio_tokenizer_type: str = field(metadata={"help": "Audio tokenizer type: glm4voice|sensevoice_glm4voice."})
    flow_path: str = field(metadata={"help": "Path to flow decoder for audio generation."})
    dataset_path: str = field(metadata={"help": "Path to RL+SFT dataset (JSONL or HF dataset)."})
    output_dir: str = field(metadata={"help": "Directory for checkpoints and logs."})

    num_train_epochs: int = 1
    max_steps: int = 2000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    warmup_steps: int = 50
    lr_scheduler_type: str = "cosine"
    num_generations_per_prompt: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0
    speech_beta: float = 0.001
    text_beta: float = 0.001
    reward_weight_semantic: float = 0.7
    reward_weight_acoustic: float = 0.3
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    attn_implementation: str = "flash_attention_2"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 1
    evaluation_steps: int = 0
    text_audio_interval_ratio: Optional[str] = None
    audio_model_freeze: bool = True
    task_types: str = "s2s"
    force_s2s: bool = False
    max_samples: Optional[int] = None
    use_wandb: bool = False
    deepspeed_config: Optional[str] = None
    tokenizer_name: Optional[str] = None
    model_max_length: Optional[int] = None
    reward_model_type: str = "api"
    use_single_reward: bool = True
    run_name: Optional[str] = None
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    skip_steps: int = 0

    rl_loss_weight: float = 0.7
    sft_loss_weight: float = 0.3
    adaptive_mixing: bool = True
    adaptive_lambda_max: float = 0.8
    adaptive_ema_alpha: float = 0.9
    adaptive_gate_slope: float = 5.0
    adaptive_reward_threshold: float = 3.0
    sft_learning_rate: Optional[float] = None
    convert_data: bool = False
    input_data_path: Optional[str] = None
    audio_base_path: Optional[str] = None
    validate_audio: bool = False

    rl_token_type: str = "all"
    sft_token_type: str = "all"
    include_audio_boundaries: bool = True
    rl_speech_weight: float = 1.0
    rl_text_weight: float = 1.0
    sft_speech_weight: float = 1.0
    sft_text_weight: float = 1.0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_text_audio_interval_ratio(ratio: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not ratio:
        return None
    parts = ratio.replace(",", " ").split()
    values = tuple(int(piece) for piece in parts)
    return values or None


def get_variant_defaults(variant: str) -> dict:
    return {
        "balance": {"text_audio_interval_ratio": (1, 4, 3, 8, 4, 10)},
        "boost": {"text_audio_interval_ratio": (1, 10, 4, 10)},
        "plus-vanilla": {"text_audio_interval_ratio": (1, 10, 4, 10)},
    }.get(variant, {"text_audio_interval_ratio": (1, 4, 3, 8, 4, 10)})


def create_training_config(args: VitaAudioRLSFTMaskedArguments) -> Tuple[GRPOConfig, Tuple[int, ...]]:
    defaults = get_variant_defaults(args.model_variant)
    text_audio_ratio = parse_text_audio_interval_ratio(args.text_audio_interval_ratio)
    if text_audio_ratio is None:
        text_audio_ratio = defaults["text_audio_interval_ratio"]

    if args.bf16 and args.fp16:
        raise ValueError("Only one of bf16 or fp16 can be enabled at a time.")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    max_prompt_length = args.model_max_length or 2048
    run_name = args.run_name or f"vita-audio-{args.model_variant}-rl-sft-masked"

    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        max_prompt_length=max_prompt_length,
        max_completion_length=1024,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        beta=args.beta,
        num_generations=args.num_generations_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.evaluation_steps,
        save_strategy="steps",
        evaluation_strategy="steps" if args.evaluation_steps > 0 else "no",
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        report_to=["wandb"] if args.use_wandb else ["tensorboard"],
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=1.2,
        model_init_kwargs={
            "torch_dtype": torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
    )

    training_args.reward_weights = (
        [1.0]
        if args.use_single_reward
        else [args.reward_weight_semantic, args.reward_weight_acoustic]
    )
    training_args.speech_beta = args.speech_beta
    training_args.text_beta = args.text_beta
    if args.sft_learning_rate is not None:
        training_args.sft_learning_rate = args.sft_learning_rate

    training_args.skip_steps = args.skip_steps
    return training_args, text_audio_ratio


def setup_reward_functions(reward_model_type: str, use_single_reward: bool) -> List:
    reward_model_type = (reward_model_type or "api").lower()
    if reward_model_type in {"api", "gpt4o"}:
        from utils.vita_audio_rewards import GPT4oRewardFunction

        if use_single_reward:
            return [GPT4oRewardFunction(evaluation_type="holistic")]
        return [
            GPT4oRewardFunction(evaluation_type="semantic"),
            GPT4oRewardFunction(evaluation_type="acoustic"),
        ]
    raise ValueError(f"Unsupported reward_model_type: {reward_model_type}")


def maybe_convert_dataset(args: VitaAudioRLSFTMaskedArguments) -> None:
    if not args.convert_data:
        return
    raise ValueError(
        "Dataset conversion utilities are not included in this repository. "
        "Please prepare a JSONL file following the schema documented in README.md."
    )


def build_dataset(args: VitaAudioRLSFTMaskedArguments, task_types: List[str]) -> VitaAudioRLSFTDataset:
    dataset = VitaAudioRLSFTDataset(
        dataset_path=args.dataset_path,
        task_types=task_types,
        use_luke_system=True,
        force_s2s=args.force_s2s,
        audio_base_path=args.audio_base_path
        or os.path.dirname(os.path.abspath(args.dataset_path)),
        validate_audio=args.validate_audio,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    logger.info("Dataset ready with %d samples", len(dataset))
    return dataset


def main() -> None:
    parser = HfArgumentParser(VitaAudioRLSFTMaskedArguments)
    args = parser.parse_args_into_dataclasses()[0]

    setup_logging()
    logger.info("================= Launch VITA-Audio RL+SFT (Masked) =================")
    logger.info("Args: %s", args)

    set_seed(args.seed)
    maybe_convert_dataset(args)

    task_types = [t.strip() for t in args.task_types.split(",") if t.strip()]
    training_args, text_audio_ratio = create_training_config(args)
    logger.info("Resolved text/audio interval ratio: %s", text_audio_ratio)

    audio_tokenizer = get_audio_tokenizer(
        args.audio_tokenizer_path,
        args.audio_tokenizer_type,
        flow_path=args.flow_path,
    )

    dataset = build_dataset(args, task_types)
    reward_funcs = setup_reward_functions(args.reward_model_type, args.use_single_reward)

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = VitaAudioRLSFTTrainerMasked(
        model=args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn_simple,
        audio_tokenizer=audio_tokenizer,
        audio_tokenizer_type=args.audio_tokenizer_type,
        flow_path=args.flow_path,
        model_variant=args.model_variant,
        text_audio_interval_ratio=text_audio_ratio,
        freeze_audio_components=args.audio_model_freeze,
        rl_loss_weight=args.rl_loss_weight,
        sft_loss_weight=args.sft_loss_weight,
        skip_steps=args.skip_steps,
        adaptive_mixing=args.adaptive_mixing,
        adaptive_lambda_max=args.adaptive_lambda_max,
        adaptive_ema_alpha=args.adaptive_ema_alpha,
        adaptive_gate_slope=args.adaptive_gate_slope,
        adaptive_reward_threshold=args.adaptive_reward_threshold,
        rl_token_type=args.rl_token_type,
        sft_token_type=args.sft_token_type,
        include_audio_boundaries=args.include_audio_boundaries,
        rl_speech_weight=args.rl_speech_weight,
        rl_text_weight=args.rl_text_weight,
        sft_speech_weight=args.sft_speech_weight,
        sft_text_weight=args.sft_text_weight,
    )

    logger.info(
        "Starting training (masked): variant=%s, adaptive=%s, rl_weight=%.3f, sft_weight=%.3f, rl_tokens=%s, sft_tokens=%s",
        args.model_variant,
        args.adaptive_mixing,
        args.rl_loss_weight,
        args.sft_loss_weight,
        args.rl_token_type,
        args.sft_token_type,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    trainer.create_model_card(f"vita-audio-{args.model_variant}-rl-sft-masked")
    logger.info("Training complete. Artifacts stored in %s", args.output_dir)


if __name__ == "__main__":
    main()
