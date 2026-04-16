#!/usr/bin/env python3
"""VITA-Audio Direct Preference Optimization entry point."""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from transformers import HfArgumentParser, set_seed
from trl import GRPOConfig

from dataset.vita_audio_rl_sft_dataset import VitaAudioRLSFTDataset, collate_fn_simple
from dpo import VitaAudioDPOTrainer
from utils.import_utils import ensure_vita_audio_importable

ensure_vita_audio_importable()
from vita_audio.tokenizer import get_audio_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class VitaAudioDPOArguments:
    """CLI arguments for DPO training."""

    model_name_or_path: str = field(metadata={"help": "Checkpoint to finetune."})
    model_variant: str = field(metadata={"help": "VITA-Audio variant: balance|boost|plus-vanilla."})
    audio_tokenizer_path: str = field(metadata={"help": "Path to audio tokenizer."})
    audio_tokenizer_type: str = field(metadata={"help": "Audio tokenizer type."})
    flow_path: str = field(metadata={"help": "Path to flow decoder for audio generation."})
    dataset_path: str = field(metadata={"help": "Path to RL+SFT dataset (JSONL or HF dataset)."})
    output_dir: str = field(metadata={"help": "Directory for checkpoints and logs."})

    num_train_epochs: int = 1
    max_steps: int = 2000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    warmup_steps: int = 50
    lr_scheduler_type: str = "cosine"
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
    run_name: Optional[str] = None
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    convert_data: bool = False
    input_data_path: Optional[str] = None
    audio_base_path: Optional[str] = None
    validate_audio: bool = False

    dpo_beta: float = 0.1
    num_negative_samples: int = 1
    negative_max_new_tokens: int = 256
    negative_temperature: float = 0.7
    negative_top_p: float = 0.9
    dpo_token_type: str = field(
        default="text",
        metadata={"help": "Which token types contribute to DPO log-prob: all|text|speech."},
    )
    include_audio_boundaries: bool = field(
        default=True,
        metadata={
            "help": "When dpo_token_type='speech', treat <|begin_of_audio|>/<|end_of_audio|> as speech tokens."
        },
    )
    dpo_data_mode: str = field(
        default="offline",
        metadata={"help": "DPO data source: auto|offline (use rejected_* fields)|online (sample negatives)."},
    )


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


def create_training_config(args: VitaAudioDPOArguments) -> Tuple[GRPOConfig, Tuple[int, ...]]:
    defaults = get_variant_defaults(args.model_variant)
    text_audio_ratio = parse_text_audio_interval_ratio(args.text_audio_interval_ratio)
    if text_audio_ratio is None:
        text_audio_ratio = defaults["text_audio_interval_ratio"]

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.bf16 and args.fp16:
        raise ValueError("Only one of bf16 or fp16 can be enabled at a time.")

    max_prompt_length = args.model_max_length or 2048
    run_name = args.run_name or f"vita-audio-{args.model_variant}-dpo"
    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        max_prompt_length=max_prompt_length,
        max_completion_length=args.negative_max_new_tokens,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        beta=1.0,
        num_generations=max(1, args.num_negative_samples),
        temperature=args.negative_temperature,
        top_p=args.negative_top_p,
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
        max_grad_norm=1.0,
        model_init_kwargs={
            "torch_dtype": torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
    )

    return training_args, text_audio_ratio


def maybe_convert_dataset(args: VitaAudioDPOArguments) -> None:
    if not args.convert_data:
        return
    raise ValueError(
        "Dataset conversion utilities are not part of the public release. "
        "Please prepare a JSONL file following the schema documented in README.md."
    )


def build_dataset(args: VitaAudioDPOArguments, task_types: List[str]) -> VitaAudioRLSFTDataset:
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
    parser = HfArgumentParser(VitaAudioDPOArguments)
    args = parser.parse_args_into_dataclasses()[0]

    setup_logging()
    logger.info("================= Launch VITA-Audio DPO =================")
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
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = VitaAudioDPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=[],  # DPO uses only SFT targets
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn_simple,
        audio_tokenizer=audio_tokenizer,
        audio_tokenizer_type=args.audio_tokenizer_type,
        flow_path=args.flow_path,
        model_variant=args.model_variant,
        text_audio_interval_ratio=text_audio_ratio,
        freeze_audio_components=args.audio_model_freeze,
        rl_loss_weight=0.0,
        sft_loss_weight=0.0,
        dpo_beta=args.dpo_beta,
        num_negative_samples=args.num_negative_samples,
        negative_generation_kwargs={
            "max_new_tokens": args.negative_max_new_tokens,
            "temperature": args.negative_temperature,
            "top_p": args.negative_top_p,
        },
        dpo_token_type=args.dpo_token_type,
        include_audio_boundaries=args.include_audio_boundaries,
        dpo_data_mode=args.dpo_data_mode,
    )

    logger.info(
        "Starting DPO training: variant=%s, beta=%.3f, negatives=%d",
        args.model_variant,
        args.dpo_beta,
        args.num_negative_samples,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    trainer.create_model_card(f"vita-audio-{args.model_variant}-dpo")
    logger.info("Training complete. Artifacts stored in %s", args.output_dir)


if __name__ == "__main__":
    main()
