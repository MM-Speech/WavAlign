import copy
import logging
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sized, Union

import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
from trl.trainer.callbacks import SyncRefModelCallback

from utils.import_utils import ensure_vita_audio_importable

ensure_vita_audio_importable()
from vita_audio.tokenizer import get_audio_tokenizer
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.constants import (
    AUD_TAG_TOKEN,
    AUD_CONTEXT_TOKEN,
    AUD_START_TOKEN,
    AUD_END_TOKEN,
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

logger = logging.getLogger(__name__)


class VitaAudioRLSFTTrainer(Trainer):
    """
    Clean RL+SFT Trainer for VITA-Audio models
    Maintains exact consistency with original GRPO logic, adds SFT capability
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        data_collator:  Callable = None,
        # VITA-Audio specific parameters
        audio_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        audio_tokenizer_type: str = "glm4voice",  # or "sensevoice_glm4voice"
        flow_path: Optional[str] = None,
        model_variant: str = "balance",  # balance, boost, plus-vanilla
        text_audio_interval_ratio: Optional[tuple] = None,
        freeze_audio_components: bool = True,
        # RL+SFT specific parameters
        rl_loss_weight: float = 0.7,
        sft_loss_weight: float = 0.3,
        skip_steps: int = 0,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-VITA-Audio-RL-SFT")

        # Store RL+SFT configuration
        self.rl_loss_weight = rl_loss_weight
        self.sft_loss_weight = sft_loss_weight
        self._skip_batches_remaining = max(0, skip_steps)

        # Set variant-specific parameters
        self.model_variant = model_variant
        self.text_audio_interval_ratio = text_audio_interval_ratio or self._get_default_interval_ratio(model_variant)
        self.freeze_audio_components = freeze_audio_components
        
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype", torch.bfloat16)
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(f"Invalid `torch_dtype` ... got {torch_dtype}.")
            
            # Load VITA-Audio model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                **model_init_kwargs
            )
            
            # Freeze audio components if specified
            if self.freeze_audio_components and hasattr(model, 'audio_encoder'):
                logger.info("Freezing audio encoder parameters...")
                for param in model.audio_encoder.parameters():
                    param.requires_grad = False
                    
            if self.freeze_audio_components and hasattr(model, 'audio_decoder'):
                logger.info("Freezing audio decoder parameters...")
                for param in model.audio_decoder.parameters():
                    param.requires_grad = False
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("You passed `model_init_kwargs` but your model is already instantiated.")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Setup reference model for KL regularization (COPIED FROM ORIGINAL)
        self.ref_model = None
        if args.beta > 0:
            if is_deepspeed_zero3_enabled():
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    **model_init_kwargs
                )
            elif not is_peft_model(model):
                self.ref_model = create_reference_model(model)
            else:
                self.ref_model = None

        # Store flow path first
        self.flow_path = flow_path
        
        # Setup audio tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_tokenizer_type = audio_tokenizer_type

        # Setup text tokenizer/processor with correct chat template (COPIED FROM ORIGINAL)
        if processing_class is None:
            model_path = model_id if isinstance(model, str) else model.config._name_or_path
            
            # Load config to determine chat template
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # Set chat template based on model type
            chat_template = None
            if "qwen2" in config.model_type.lower():
                try:
                    from evaluation.get_chat_template import qwen2_chat_template as chat_template
                except ImportError:
                    chat_template = None
                self.add_generation_prompt = True
                self.default_system_message = []
            elif "hunyuan" in config.model_type.lower():
                try:
                    from evaluation.get_chat_template import hunyuan_chat_template as chat_template
                except ImportError:
                    chat_template = None
                self.add_generation_prompt = False
                self.default_system_message = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant.",
                    }
                ]
            else:
                self.add_generation_prompt = True
                self.default_system_message = []
            
            # Luke system message (consistent with original)
            self.luke_system_message = [
                {
                    "role": "system",
                    "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
                },
            ]
            
            processing_class = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                chat_template=chat_template if chat_template else None
            )
            
            # Set pad token if not set
            if processing_class.pad_token is None:
                processing_class.pad_token = processing_class.eos_token

        # Initialize reward functions - following reference pattern (COPIED FROM ORIGINAL)
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        for i, rf in enumerate(self.reward_funcs):
            if isinstance(rf, str):
                from transformers import AutoModelForSequenceClassification
                self.reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(rf, num_labels=1)

        # Setup reward weights like reference (COPIED FROM ORIGINAL)
        self.reward_weights = torch.tensor(
            args.reward_weights if hasattr(args, 'reward_weights') and args.reward_weights else [1.0] * len(self.reward_funcs),
            dtype=torch.float32,
        )

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(self.reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        self.reward_processing_classes = reward_processing_classes
        for i, (rf, tok) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(rf, PreTrainedModel):
                if tok is None:
                    tok = AutoTokenizer.from_pretrained(rf.config._name_or_path, padding_side="right")
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                rf.config.pad_token_id = tok.pad_token_id
                self.reward_processing_classes[i] = tok

        # Store config and setup generation config (COPIED FROM ORIGINAL)
        self.args = args
        self.processing_class = processing_class
        
        # ---------- Metrics (consistent with grpo_trainer_text) ----------
        self._metrics = defaultdict(list)
        
        # Setup generation config with MTP support
        self.generation_config = self._setup_generation_config(model)
        
        # Initialize parent class
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            data_collator=data_collator
        )
        
        # Setup sample logging (consistent with grpo_trainer_text)
        if self.accelerator.is_main_process:
            import datetime
            from pathlib import Path
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.sample_log_path = out_dir / f"train_samples_{ts}.jsonl"
            self.sample_log_path.write_text("")  # Clear/create empty file
        else:
            self.sample_log_path = None

    def get_batch_samples(self, epoch_iterator, num_batches):
        """
        Consume and drop the first N batches once per training run to avoid OOM, then
        fall back to the default batching logic (which also handles resume skipping).
        """
        if self._skip_batches_remaining > 0:
            skipped = 0
            for _ in range(min(num_batches, self._skip_batches_remaining)):
                try:
                    next(epoch_iterator)
                    skipped += 1
                except StopIteration:
                    break
            self._skip_batches_remaining -= skipped
            if skipped > 0:
                logger.info("Skipped %d batch(es) via skip_steps warmup.", skipped)
            if skipped == num_batches:
                return [], None
            # adjust the requested batch count for remaining fetch
            num_batches = num_batches - skipped
        return super().get_batch_samples(epoch_iterator, num_batches)
    
    def _get_default_interval_ratio(self, variant: str) -> tuple:
        """Get default text-audio interval ratio for different variants (COPIED FROM ORIGINAL)"""
        ratios = {
            "boost": [1, 10, 4, 10],
            "balance": [1, 4, 3, 8, 4, 10],
            "plus-vanilla":[1, 10, 4, 10],  # Plus-vanilla uses same as boost
        }
        return ratios.get(variant, [1, 4, 3, 8, 4, 10])
    
    def _setup_generation_config(self, model):
        """Setup generation config with VITA-Audio specific settings (COPIED FROM ORIGINAL)"""
        try:
            # Try to load original generation config
            if hasattr(model, 'generation_config') and model.generation_config is not None:
                generation_config = copy.deepcopy(model.generation_config)
                logger.info("Using model's original generation config")
            else:
                # Fallback: create new generation config
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.max_completion_length,
                    do_sample=True,
                    temperature=self.args.temperature,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                )
                logger.info("Created fallback generation config")
            
            # Override with training-specific settings
            generation_config.max_new_tokens = self.args.max_completion_length
            generation_config.do_sample = True
            generation_config.temperature = self.args.temperature
            generation_config.use_cache = True
            
            # Handle EOS token for different model types
            if hasattr(model.config, 'model_type') and model.config.model_type == "hunyuan":
                generation_config.eos_token_id = getattr(self.processing_class, 'eos_id', self.processing_class.eos_token_id)
            else:
                generation_config.eos_token_id = self.processing_class.eos_token_id
            
            # Set MTP mode based on model architecture and variant
            has_mtp = getattr(model.config, 'num_nextn_predict_layers', 0) > 0
            
            if has_mtp:
                # Model has MTP layers, use variant-specific mode
                mtp_mode = list(self.text_audio_interval_ratio)
                generation_config.mtp_inference_mode = mtp_mode
                logger.info(f"Set MTP mode for {self.model_variant}: {mtp_mode}")
            else:
                # Vanilla model without MTP layers, use standard generation
                if hasattr(generation_config, 'mtp_inference_mode'):
                    generation_config.mtp_inference_mode = [8192, 0]  # Pure standard mode
                logger.info(f"Model has no MTP layers, using standard generation mode")
            
            logger.info(f"Final generation config: {generation_config}")
            return generation_config
            
        except Exception as e:
            logger.warning(f"Failed to setup generation config: {e}")
            # Minimal fallback
            return GenerationConfig(
                max_new_tokens=self.args.max_completion_length,
                do_sample=True,
                temperature=self.args.temperature,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )
    
    def _supports_token_type_separation(self):
        """Check if model supports separate speech/text token KL calculation. (COPIED FROM ORIGINAL)"""
        return True

    def _generate_sequences_for_variant(
        self,
        generation_model: PreTrainedModel,
        prompt_inputs: Dict[str, torch.Tensor],
        generation_config: GenerationConfig,
        num_generations: int,
    ) -> torch.Tensor:
        if num_generations < 1:
            raise ValueError("num_generations must be >= 1")

        base_config = copy.deepcopy(generation_config)

        if self.model_variant == "plus-vanilla":
            sequences: List[torch.Tensor] = []
            max_length = 0
            for _ in range(num_generations):
                per_call_config = copy.deepcopy(base_config)
                per_call_config.num_return_sequences = 1
                result = generation_model.generate(
                    **prompt_inputs,
                    generation_config=per_call_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                seq = result.sequences  # (1, L)
                sequences.append(seq)
                max_length = max(max_length, seq.size(1))

            pad_token_id = self.processing_class.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processing_class.eos_token_id if self.processing_class.eos_token_id is not None else 0

            padded = []
            for seq in sequences:
                if seq.size(1) < max_length:
                    pad_width = max_length - seq.size(1)
                    seq = F.pad(seq, (0, pad_width), value=pad_token_id)
                padded.append(seq)
            return torch.cat(padded, dim=0)

        multi_config = copy.deepcopy(base_config)
        multi_config.num_return_sequences = num_generations
        result = generation_model.generate(
            **prompt_inputs,
            generation_config=multi_config,
            return_dict_in_generate=True,
            output_scores=False,
        )
        return result.sequences

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined RL+SFT loss - EACH SAMPLE contributes to BOTH losses
        """
        if return_outputs:
            raise ValueError("The VitaAudioRLSFTTrainer does not support returning outputs")
        
        # Step 1: Compute GRPO loss (exactly as original) 
        grpo_loss = self._compute_grpo_loss(model, inputs)
        
        # Step 2: Compute SFT loss for samples that have answer targets
        sft_loss = self._compute_sft_loss(model, inputs)
        
        # Step 3: Combine losses
        total_loss = self.rl_loss_weight * grpo_loss + self.sft_loss_weight * sft_loss
        
        # Log metrics (aggregate globally across devices)
        rl_loss_g = self.accelerator.gather_for_metrics(grpo_loss.detach()).mean().item()
        sft_loss_g = self.accelerator.gather_for_metrics(sft_loss.detach()).mean().item()
        total_loss_g = self.accelerator.gather_for_metrics(total_loss.detach()).mean().item()
        num_samples_t = torch.tensor(len(inputs), device=grpo_loss.device, dtype=torch.float32)
        num_samples_g = self.accelerator.gather_for_metrics(num_samples_t).mean().item()
        
        self._metrics["rl_loss"].append(rl_loss_g)
        self._metrics["sft_loss"].append(sft_loss_g) 
        self._metrics["combined_loss"].append(total_loss_g)
        self._metrics["num_samples"].append(num_samples_g)
        
        logger.info(f"GRPO loss: {grpo_loss.item():.4f}, SFT loss: {sft_loss.item():.4f}, Combined: {total_loss.item():.4f}")
        
        return total_loss
    
    def _compute_grpo_loss(self, model, rl_inputs):
        """
        Compute pure GRPO loss - EXACTLY COPIED FROM ORIGINAL TRAINER
        """
        # Following qwenomnithinker_rl structure exactly
        batch_losses = []
        batch_rewards_stats = defaultdict(list)
        batch_kl_stats = []
        batch_completion_lengths = []

        # Collect all generated sequences and metadata for batch processing later
        all_generated_sequences = []  # Will store all B*G sequences
        all_prompt_lengths = []
        all_completions_text = []
        all_messages_for_reward = []
        all_audios_for_reward = []

        # Track audio features/indices per RL row (aligned with all_generated_sequences order)
        rl_audios_per_row: list[list[torch.Tensor]] = []
        rl_audio_indices_per_row: list[list[torch.Tensor]] = []
        
        # Process each sample individually (required for audio handling)
        for sample_idx in range(len(rl_inputs)):
            current_sample = rl_inputs[sample_idx]
            messages = current_sample["messages"]
            audios = current_sample.get("audios", [])
            
            # Process S2S messages with proper system message
            processed_messages = self._prepare_s2s_messages(messages, audios)
            
            # Handle VITA-Audio specific processing
            prompt_input_ids = self.processing_class.apply_chat_template(
                processed_messages,
                tokenize=True,
                add_generation_prompt=self.add_generation_prompt,
            )
            
            # Handle contiguous audio codec if needed
            audios_processed = None
            audio_indices = None
            
            if audios and self.audio_tokenizer and self.audio_tokenizer.apply_to_role(
                "user", is_contiguous=True
            ):
                # Contiguous codec
                prompt_input_ids, audios_processed, audio_indices = add_audio_input_contiguous(
                    prompt_input_ids, audios, self.processing_class, self.audio_tokenizer
                )
                logger.debug(f"Processed contiguous audio: {len(audios)} files")
            elif audios and self.audio_tokenizer and self.audio_tokenizer.apply_to_role(
                "user", is_discrete=True
            ):
                # Discrete codec - process audio tokens into text
                processed_messages = self._process_discrete_audio_tokens(processed_messages, audios)
                # Re-tokenize with audio tokens embedded
                prompt_input_ids = self.processing_class.apply_chat_template(
                    processed_messages,
                    tokenize=True,
                    add_generation_prompt=self.add_generation_prompt,
                )
                logger.debug(f"Using discrete audio tokens in text")
            
            prompt_inputs_dict = {
                "input_ids": torch.tensor([prompt_input_ids], dtype=torch.long),
            }
            
            if audios_processed is not None:
                prompt_inputs_dict["audios"] = audios_processed
                prompt_inputs_dict["audio_indices"] = audio_indices
            
            # Move to device
            for k, v in prompt_inputs_dict.items():
                if torch.is_tensor(v):
                    prompt_inputs_dict[k] = v.to(model.device)
                    
            prompt_length = prompt_inputs_dict["input_ids"].size(1)
                
            # Generate G completions using num_return_sequences (much more efficient!)
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped, torch.no_grad():
                
                generation_config = copy.deepcopy(self.generation_config)
                generation_config.max_new_tokens = min(512, self.args.max_completion_length)
                generation_config.temperature = self.args.temperature
                generation_config.num_return_sequences = self.args.num_generations  # Generate G sequences at once
                generation_config.pad_token_id = self.processing_class.pad_token_id
                generation_config.eos_token_id = self.processing_class.eos_token_id
                
                generated_sequences = self._generate_sequences_for_variant(
                    generation_model=unwrapped,
                    prompt_inputs=prompt_inputs_dict,
                    generation_config=generation_config,
                    num_generations=self.args.num_generations,
                )
                             
            # Store for batch processing
            for g_idx in range(self.args.num_generations):
                seq = generated_sequences[g_idx:g_idx+1]  # Keep dimension (1, P+C)
                all_generated_sequences.append(seq)
                all_prompt_lengths.append(prompt_length)
                
                # Decode completion text (filter out speech tokens)
                completion_ids = seq[:, prompt_length:]
                speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
                text_only_ids = [tid for tid in completion_ids[0].tolist() if tid < speech_token_offset]
                completion_text = self.processing_class.decode(text_only_ids, skip_special_tokens=True)
                all_completions_text.append(completion_text)
                
                # Store original prompt text for reward computation (use question_text field)
                original_prompt_text = current_sample.get("question_text")
                if not original_prompt_text:
                    user_segments = []
                    for msg in processed_messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if content:
                                user_segments.append(content.replace("<|audio|>", "").strip())
                    original_prompt_text = "\n\n".join(seg for seg in user_segments if seg) or "[audio-only prompt]"
                all_messages_for_reward.append(original_prompt_text)

                # Record audio features/indices for this RL row
                if audios_processed is not None and audio_indices is not None:
                    if isinstance(audios_processed, torch.Tensor):
                        rl_audios_per_row.append([audios_processed])
                    else:
                        rl_audios_per_row.append(audios_processed)
                    if isinstance(audio_indices, torch.Tensor):
                        rl_audio_indices_per_row.append([audio_indices])
                    else:
                        rl_audio_indices_per_row.append(audio_indices)
                else:
                    rl_audios_per_row.append([])
                    rl_audio_indices_per_row.append([])

        
        # Now we have all B*G sequences, do batch forward pass
        # Step 1: Efficiently batch and pad all sequences to same length
        max_seq_length = max(seq.size(1) for seq in all_generated_sequences)
        
        # Pad all sequences to max length first, then stack
        padded_sequences = []
        for seq in all_generated_sequences:
            if seq.size(1) < max_seq_length:
                pad_size = max_seq_length - seq.size(1)
                padded_seq = F.pad(seq, (0, pad_size), value=self.processing_class.pad_token_id)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        # Now we can safely concatenate
        batch_sequences = torch.cat(padded_sequences, dim=0)  # (B*G, max_seq_length)
        
        # Vectorized creation of attention masks
        # Simply check for non-padding tokens (including EOS)
        batch_attention_masks = (batch_sequences != self.processing_class.pad_token_id).float()
        
        # Create completion masks efficiently
        batch_completion_masks = []
        prompt_lens = torch.tensor(all_prompt_lengths, device=batch_sequences.device)
        
        for i in range(batch_sequences.size(0)):
            prompt_len = prompt_lens[i].item()
            
            # Create completion mask
            comp_mask = torch.zeros(max_seq_length - prompt_len, device=batch_sequences.device)
            
            # Find EOS position in completion part
            completion_part = batch_sequences[i, prompt_len:]
            eos_mask = (completion_part == self.processing_class.eos_token_id)
            
            if eos_mask.any():
                # Mask up to and including first EOS
                eos_idx = eos_mask.nonzero(as_tuple=True)[0][0].item()
                comp_mask[:eos_idx + 1] = 1
            else:
                # No EOS found, mask all non-padding tokens in completion
                non_pad_mask = (completion_part != self.processing_class.pad_token_id)
                if non_pad_mask.any():
                    last_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][-1].item()
                    comp_mask[:last_non_pad + 1] = 1
            
            batch_completion_masks.append(comp_mask.unsqueeze(0))
        
        batch_completion_masks = torch.cat(batch_completion_masks, dim=0)  # (B*G, max_comp_length)
        
        # Step 2: Single batch forward pass for all sequences
        # Attach audio features/indices for RL rows if present
        policy_inputs_audio = {}
        flat_audios: list[torch.Tensor] = []
        flat_audio_indices: list[torch.Tensor] = []
        for row_idx in range(len(rl_audios_per_row)):
            for a in rl_audios_per_row[row_idx]:
                flat_audios.append(a)
            for idx_t in rl_audio_indices_per_row[row_idx]:
                t = idx_t.clone()
                # set batch index in indices to RL row index in the concatenated batch
                t[0, :, :] = row_idx
                flat_audio_indices.append(t)
        if len(flat_audios) > 0 and len(flat_audio_indices) > 0:
            policy_inputs_audio = {"audios": flat_audios, "audio_indices": flat_audio_indices}

        batch_logps, batch_logits = self._get_per_token_logps(
            model, batch_sequences, batch_attention_masks, policy_inputs_audio, return_logits=True
        )
        
        # Step 3: Single batch forward pass for reference model
        with torch.no_grad():
            if self.ref_model is not None:
                self.ref_model.train()
                # Ensure ref_model is on correct device
                target_device = batch_sequences.device
                try:
                    ref_device = next(self.ref_model.parameters()).device
                    if ref_device != target_device:
                        self.ref_model = self.ref_model.to(target_device)
                except StopIteration:
                    self.ref_model = self.ref_model.to(target_device)
                
                # Reference forward under the same multi-modal condition
                ref_inputs_audio = {}
                if len(flat_audios) > 0 and len(flat_audio_indices) > 0:
                    ref_inputs_audio = {"audios": flat_audios, "audio_indices": flat_audio_indices}
                ref_batch_logps = self._get_per_token_logps(
                    self.ref_model, batch_sequences, batch_attention_masks, ref_inputs_audio, return_logits=False
                )
            else:
                ref_batch_logps = batch_logps.detach()
        
        # Step 4: Process results for each sample
        batch_size = len(rl_inputs)
        num_generations = self.args.num_generations
        
        # Step 5: Extract audio from generated sequences and decode
        all_decoded_audios = []
        if self.audio_tokenizer:
            # Get speech token offset from model config (consistent with other parts)
            speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
            
            for seq in all_generated_sequences:
                # Extract audio tokens from the sequence
                audio_tokens = []
                for token_id in seq[0].tolist():  # seq is (1, length)
                    if token_id >= speech_token_offset:
                        # Audio tokens are offset by speech_token_offset
                        audio_tokens.append(token_id - speech_token_offset)
                
                # Decode audio tokens to waveform
                if len(audio_tokens) > 0:
                    try:
                        decoded_audio = self.audio_tokenizer.decode(audio_tokens)
                        all_decoded_audios.append(decoded_audio)
                    except Exception as e:
                        logger.warning(f"Failed to decode audio: {e}")
                        all_decoded_audios.append(None)
                else:
                    all_decoded_audios.append(None)
        
        # Step 6: Compute GRPO loss for each sample
        total_loss = 0
        for sample_idx in range(batch_size):
            # Get indices for this sample's generations
            start_idx = sample_idx * num_generations
            end_idx = start_idx + num_generations
            
            # Extract logps for this sample's generations
            sample_model_logps = batch_logps[start_idx:end_idx]  # (G, max_length)
            sample_ref_logps = ref_batch_logps[start_idx:end_idx]  # (G, max_length)
            sample_completion_masks = batch_completion_masks[start_idx:end_idx]  # (G, max_comp_length)
            
            # Get prompt length for this sample
            prompt_len = all_prompt_lengths[start_idx]
            
            # Extract completion logps
            completion_model_logps = sample_model_logps[:, prompt_len - 1:]  # (G, max_C)
            completion_ref_logps = sample_ref_logps[:, prompt_len - 1:]  # (G, max_C)
            
            # Align masks with logps
            max_logps_len = completion_model_logps.size(1)
            if sample_completion_masks.size(1) != max_logps_len:
                # Pad or truncate masks to match logps
                if sample_completion_masks.size(1) < max_logps_len:
                    pad_size = max_logps_len - sample_completion_masks.size(1)
                    sample_completion_masks = F.pad(sample_completion_masks, (0, pad_size), value=0)
                else:
                    sample_completion_masks = sample_completion_masks[:, :max_logps_len]
            
            # Compute rewards for this sample's generations - following reference logic
            sample_completions = all_completions_text[start_idx:end_idx]
            sample_audios = all_decoded_audios[start_idx:end_idx] if all_decoded_audios else [None] * num_generations
            sample_messages = all_messages_for_reward[start_idx:end_idx]
            
            # Create rewards tensor following reference pattern
            rewards_f = torch.zeros(num_generations, len(self.reward_funcs), device=model.device)
            for idx, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    # Model-based reward (if any)
                    # Handle similar to reference implementation
                    texts = [sample_messages[g] + sample_completions[g] for g in range(num_generations)]
                    inp = self.reward_processing_classes[idx](texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
                    inp = {k: v.to(model.device) for k, v in inp.items()}
                    with torch.inference_mode():
                        rewards_f[:, idx] = reward_func(**inp).logits[:, 0]
                else:
                    # Callable reward function
                    reward_vals = reward_func(
                        prompts=sample_messages,
                        completions=sample_completions,
                        audios=sample_audios
                    )
                    rewards_f[:, idx] = torch.tensor(reward_vals, dtype=torch.float32, device=model.device)
            
            # Apply reward weights and sum - exactly like reference  
            rewards_f *= self.reward_weights.to(model.device).unsqueeze(0)
            rewards = rewards_f.sum(dim=1)  # (G,)
            
            # GRPO loss computation
            grouped_mean = rewards.mean()
            grouped_std = rewards.std() + 1e-4
            adv = (rewards - grouped_mean) / grouped_std
            
            # Log rewards like reference
            print(f"Rewards: {rewards.tolist()}, Adv: {adv.tolist()}")
            
            # Separate KL divergence for text and speech tokens
            per_token_kl = torch.exp(completion_ref_logps - completion_model_logps) - \
                          (completion_ref_logps - completion_model_logps) - 1  # (G, max_C)
            
            # Get token type masks if model supports it
            if self._supports_token_type_separation():
                # Get completion IDs for mask creation
                completion_ids_for_mask = batch_sequences[start_idx:end_idx, prompt_len:]
                speech_token_mask, text_token_mask = self._get_token_type_masks(
                    completion_ids_for_mask, sample_completion_masks
                )
                
                # Apply different beta weights for text and speech
                text_beta = getattr(self.args, 'text_beta', self.args.beta)
                speech_beta = getattr(self.args, 'speech_beta', self.args.beta * 0.5)  # Lower beta for speech
                
                weighted_kl = text_beta * per_token_kl * text_token_mask + \
                             speech_beta * per_token_kl * speech_token_mask
            else:
                # Apply uniform beta weighting
                weighted_kl = self.args.beta * per_token_kl
            
            # GRPO per-token loss - following reference exactly
            per_tok_loss = torch.exp(completion_model_logps - completion_model_logps.detach()) * adv.unsqueeze(1)
            per_tok_loss = -(per_tok_loss - weighted_kl)
            
            # Average over completions
            sample_loss = ((per_tok_loss * sample_completion_masks).sum(dim=1) / \
                          sample_completion_masks.sum(dim=1).clamp(min=1)).mean()
            
            total_loss += sample_loss
            
            # Collect metrics following reference pattern
            comp_len = sample_completion_masks.sum(1).float().mean()
            self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(comp_len).mean().item())
            
            # Gather rewards for metrics - following reference exactly
            rw_gather = self.accelerator.gather_for_metrics(rewards_f).mean(0)
            for i, rf in enumerate(self.reward_funcs):
                nm = rf.config._name_or_path.split("/")[-1] if isinstance(rf, PreTrainedModel) else rf.__name__
                self._metrics[f"rewards/{nm}"].append(rw_gather[i].item())
            self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
            self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(grouped_std).mean().item())
            
            if self._supports_token_type_separation():
                # Separate KL metrics for text and speech
                text_kl = ((per_token_kl * text_token_mask).sum(1) / \
                          text_token_mask.sum(1).clamp(min=1)).mean()
                speech_kl = ((per_token_kl * speech_token_mask).sum(1) / \
                            speech_token_mask.sum(1).clamp(min=1)).mean()
                kl_overall = ((per_token_kl * sample_completion_masks).sum(1) / \
                              sample_completion_masks.sum(1).clamp(min=1)).mean()
                print(f"speech kl:{speech_kl.item()} text kl:{text_kl.item()}")
                self._metrics["text_kl"].append(self.accelerator.gather_for_metrics(text_kl).mean().item())
                self._metrics["speech_kl"].append(self.accelerator.gather_for_metrics(speech_kl).mean().item())
                self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl_overall).mean().item())
                
                # Also compute entropy by token type
                with torch.no_grad():
                    # Get logits for entropy computation
                    completion_logits = batch_logits[start_idx:end_idx, prompt_len - 1:-1, :]  # (G, C, V)
                    entropy_stats = self._compute_entropy_by_token_type(
                        completion_logits, completion_ids_for_mask, sample_completion_masks
                    )
                    print(f"text enr:{entropy_stats['text_entropy']} speech_entropy:{entropy_stats['speech_entropy']}")
                    text_ent_tensor = torch.tensor(entropy_stats["text_entropy"], device=model.device)
                    speech_ent_tensor = torch.tensor(entropy_stats["speech_entropy"], device=model.device)
                    total_ent_tensor = torch.tensor(entropy_stats["total_entropy"], device=model.device)
                    self._metrics["text_entropy"].append(self.accelerator.gather_for_metrics(text_ent_tensor).mean().item())
                    self._metrics["speech_entropy"].append(self.accelerator.gather_for_metrics(speech_ent_tensor).mean().item())
                    self._metrics["total_entropy"].append(self.accelerator.gather_for_metrics(total_ent_tensor).mean().item())
            else:
                kl_overall = ((per_token_kl * sample_completion_masks).sum(1) / \
                              sample_completion_masks.sum(1).clamp(min=1)).mean()
                self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl_overall).mean().item())
        
        # Average loss over all samples
        final_loss = total_loss / batch_size
        return final_loss
    
    def _compute_sft_loss(self, model, inputs):
        """
        Compute SFT loss using integrated VITA-Audio preprocess logic
        Each sample contributes to SFT loss if it has answer targets
        """
        batch_loss = 0.0
        valid_samples = 0
        
        # Get token IDs
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
        
        IGNORE_TOKEN_ID = -100
        
        for sample in inputs:
            # Check if sample has SFT targets
            sft_target_text = sample.get("sft_target_text", "")
            sft_target_audio_path = sample.get("sft_target_audio")
            
            if not (sft_target_text or sft_target_audio_path):
                continue
            
            try:
                # Construct messages with SFT target
                messages = sample["messages"].copy()
                audios = sample.get("audios", []).copy()
                
                # Add target assistant response
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
                
                # Track which audios are used for different processing modes
                contiguous_audio_idxs = []
                
                # Process discrete audio tokens if needed
                if audios and self.audio_tokenizer and self.audio_tokenizer.is_discrete:
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
                            # This role uses contiguous, record audio indices for later processing
                            audio_count = content.count(AUD_TAG_TOKEN)
                            contiguous_audio_idxs.extend(range(audio_idx, audio_idx + audio_count))
                            audio_idx += audio_count
                        sentence["content"] = content
                
                # Build input_ids and targets following VITA-Audio logic
                input_ids, targets = [], []
                audios_processed = []
                audio_indices_processed = []
                
                # Prepare messages with system message (following VITA-Audio logic)
                processed_messages = self._prepare_s2s_messages(messages, audios)
                
                # Process each message
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
                            # Apply text-audio interval
                            content_input_id = self._apply_text_audio_interval(
                                content_input_id, AUD_START_ID, AUD_END_ID
                            )
                        
                        _input_id = (
                            IM_START_IDS + ASSISTANT_IDS + nl_tokens + 
                            content_input_id + IM_END_IDS + nl_tokens
                        )
                        _target = (
                            [IGNORE_TOKEN_ID] * len(IM_START_IDS) +
                            [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS) +
                            [IGNORE_TOKEN_ID] * len(nl_tokens) +
                            content_input_id + IM_END_IDS + nl_tokens
                        )
                    
                    elif role == "system":
                        _input_id = (
                            IM_START_IDS + SYSTEM_IDS + nl_tokens +
                            self.processing_class(content, add_special_tokens=False).input_ids +
                            IM_END_IDS + nl_tokens
                        )
                        _target = [IGNORE_TOKEN_ID] * len(_input_id)
                    
                    input_ids += _input_id
                    targets += _target
                
                # Process contiguous audio (only for remaining AUD_TAG_TOKEN)
                if contiguous_audio_idxs and self.audio_tokenizer and self.audio_tokenizer.is_contiguous:
                    aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
                    if len(aud_positions) > 0:
                        new_input_ids = []
                        new_targets = []
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
                                
                                # Create audio indices
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
                
                # Convert to tensors and forward pass
                inputs_dict = {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long, device=model.device),
                    "labels": torch.tensor([targets], dtype=torch.long, device=model.device)
                }
                
                if audios_processed:
                    inputs_dict["audios"] = audios_processed
                if audio_indices_processed:
                    inputs_dict["audio_indices"] = audio_indices_processed
                
                outputs = model(**inputs_dict)
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    batch_loss += outputs.loss
                    valid_samples += 1
                    
            except Exception as e:
                logger.warning(f"SFT processing failed for sample: {e}")
                continue
        
        if valid_samples > 0:
            return batch_loss / valid_samples
        else:
            return torch.tensor(0.0, device=model.device, requires_grad=True)
    
    def _apply_text_audio_interval(self, content_input_id, AUD_START_ID, AUD_END_ID):
        """Apply text-audio interval following VITA-Audio logic exactly"""
        if not self.text_audio_interval_ratio:
            return content_input_id
        
        # Use the exact same logic as the original text_audio_interval function
        text_nums = list(self.text_audio_interval_ratio[::2])  # [1, 3, 4]
        audio_nums = list(self.text_audio_interval_ratio[1::2])  # [4, 8, 10]
        
        # exclude AUD_START and AUD_END
        audio_nums = [x - 2 for x in audio_nums]  # [2, 6, 8]
        
        st = [i for i, x in enumerate(content_input_id) if x == AUD_START_ID]
        ed = [i for i, x in enumerate(content_input_id) if x == AUD_END_ID]
        
        # only text - no audio blocks found
        if len(st) == 0 and len(ed) == 0:
            return content_input_id
        
        assert len(st) == 1, f"Expected 1 AUD_START, found {len(st)}"
        assert len(ed) == 1, f"Expected 1 AUD_END, found {len(ed)}"
        
        st = st[0]
        ed = ed[0]
        
        assert st < ed, f"AUD_START {st} should be before AUD_END {ed}"
        
        # only audio - entire content is audio
        if st == 0 and ed == len(content_input_id) - 1:
            return content_input_id
        
        # Extract audio and text tokens
        audio_tokens = content_input_id[st + 1 : ed]  # Tokens between AUD_START and AUD_END
        text_tokens = content_input_id[:st] + content_input_id[ed + 1 :]  # Everything else
        
        # Chunk audio tokens according to audio_nums
        audio_tokens_chunks = []
        audio_nums_copy = audio_nums.copy()  # Don't modify original
        while len(audio_tokens) > 0:
            if len(audio_nums_copy) > 1:
                audio_num = audio_nums_copy.pop(0)
            else:
                audio_num = audio_nums_copy[0]
            
            audio_tokens_chunks.append(audio_tokens[:audio_num])
            audio_tokens = audio_tokens[audio_num:]
        
        # Chunk text tokens according to text_nums
        text_tokens_chunks = []
        text_nums_copy = text_nums.copy()  # Don't modify original
        while len(text_tokens) > 0:
            if len(text_nums_copy) > 1:
                text_num = text_nums_copy.pop(0)
            else:
                text_num = text_nums_copy[0]
            
            text_tokens_chunks.append(text_tokens[:text_num])
            text_tokens = text_tokens[text_num:]
        
        # Balance chunks and merge remainder
        chunk_num = min(len(audio_tokens_chunks), len(text_tokens_chunks))
        audio_tokens_chunks = audio_tokens_chunks[: chunk_num - 1] + [
            sum(audio_tokens_chunks[chunk_num - 1 :], [])
        ]
        text_tokens_chunks = text_tokens_chunks[: chunk_num - 1] + [
            sum(text_tokens_chunks[chunk_num - 1 :], [])
        ]
        
        # Reconstruct in alternating pattern
        interval_input_ids = []
        for text_tokens, audio_tokens in zip(text_tokens_chunks, audio_tokens_chunks):
            interval_input_ids += text_tokens + [AUD_START_ID] + audio_tokens + [AUD_END_ID]
        
        return interval_input_ids

    # ALL METHODS BELOW ARE COPIED EXACTLY FROM ORIGINAL TRAINER
    def _get_per_token_logps(self, model, input_ids, attention_mask, prompt_inputs_dict, return_logits=False):
        """
        Compute per-token log probabilities from VITA-Audio model (COPIED FROM ORIGINAL)
        """
        # For VITA-Audio models, we need to handle both text and audio inputs
        target_ids = input_ids[:, 1:]  # Shift for next token prediction

        
        # Debug: Check model training mode
        logger.debug(f"Model training mode: {model.training}")
        
        # Initialize num_prefill_tokens if model has it but not initialized
        if not model.training and not hasattr(model, 'num_prefill_tokens'):
            model.num_prefill_tokens = -1
            logger.warning("Initialized num_prefill_tokens for model in eval mode")
        
        # Forward pass through model with proper VITA-Audio handling
        try:
            if prompt_inputs_dict.get("audios") is not None and prompt_inputs_dict.get("audio_indices") is not None:
                # Handle contiguous audio inputs
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audios=prompt_inputs_dict["audios"],
                    audio_indices=prompt_inputs_dict["audio_indices"],
                    return_dict=True,
                    use_cache=False
                )
            else:
                # Standard forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False
                )
        except Exception as e:
            logger.warning(f"Forward pass failed with audio inputs, falling back to text-only: {e}")
            # Fallback: use input_ids only
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False
            )
        
        logits = outputs.logits[:, :-1, :]  # Remove last token logits
        
        # Use selective_log_softmax similar to qwenomnithinker_rl
        from trl.trainer.utils import selective_log_softmax
        per_token_logps = selective_log_softmax(logits, target_ids)
        
        if return_logits:
            return per_token_logps, outputs.logits
        else:
            return per_token_logps
    
    def _get_token_type_masks(self, completion_ids, completion_masks):
        """
        Get masks for speech tokens vs text tokens in VITA-Audio model. (COPIED FROM ORIGINAL)
        """
        # Get speech token offset from model config
        speech_token_offset = self.processing_class.convert_tokens_to_ids("<|audio_0|>")
        vocab_size = getattr(self.model.config, 'vocab_size', 168072)
        
        # Create masks for token types
        # Speech tokens: [speech_token_offset, vocab_size)
        # Text tokens: [0, speech_token_offset)
        speech_token_mask = ((completion_ids >= speech_token_offset) & 
                           (completion_ids < vocab_size)).float()
        text_token_mask = (completion_ids < speech_token_offset).float()
        
        # Apply completion mask to ensure we only consider valid tokens
        speech_token_mask = speech_token_mask * completion_masks.float()
        text_token_mask = text_token_mask * completion_masks.float()
        
        return speech_token_mask, text_token_mask
    
    def _compute_entropy_by_token_type(self, logits, completion_ids, completion_masks):
        """
        Compute entropy separately for speech and text tokens. (COPIED FROM ORIGINAL)
        NOTE: This should be called within torch.no_grad() context.
        """
        speech_token_mask, text_token_mask = self._get_token_type_masks(completion_ids, completion_masks)
        
        # Compute per-token entropy (detach to ensure no gradients)
        logits = logits.detach()
        probs = F.softmax(logits, dim=-1)  # (G, C, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (G, C, V)
        entropy = -(probs * log_probs).sum(dim=-1)  # (G, C)
        
        # Separate entropy for speech and text tokens
        speech_entropy = (entropy * speech_token_mask).sum() / speech_token_mask.sum().clamp(min=1)
        text_entropy = (entropy * text_token_mask).sum() / text_token_mask.sum().clamp(min=1)
        total_entropy = (entropy * completion_masks.float()).sum() / completion_masks.sum().clamp(min=1)
        
        return {
            "speech_entropy": speech_entropy.item(),
            "text_entropy": text_entropy.item(), 
            "total_entropy": total_entropy.item(),
            "speech_token_count": speech_token_mask.sum().item(),
            "text_token_count": text_token_mask.sum().item()
        }

    def _prepare_s2s_messages(self, messages: list, audios: list) -> list:
        """
        Prepare messages for S2S tasks and prepend a single system message when needed.
        Avoids double system injection if the first message is already a system.
        """
        # Determine if this is an S2S task based on message content
        has_input_audio = any("<|audio|>" in msg.get("content", "") for msg in messages if msg.get("role") == "user")
        has_output_audio = any("<|audio|>" in msg.get("content", "") for msg in messages if msg.get("role") == "assistant")

        # Choose system message depending on modality
        if has_input_audio or has_output_audio:
            system_message = self.luke_system_message
        else:
            system_message = self.default_system_message

        # If the first message is already a system, do not add another
        if messages and messages[0].get("role") == "system":
            return messages

        # If no system message configured, return as-is
        if not system_message:
            return messages

        # Prepend exactly one system message
        return system_message + messages
    
    def _process_discrete_audio_tokens(self, messages: list, audios: list) -> list:
        """
        Process discrete audio tokens into message content. (COPIED FROM ORIGINAL)
        Following the official inference_sts.py format.
        """
        if not audios or not self.audio_tokenizer:
            return messages
        
        processed_messages = copy.deepcopy(messages)
        audio_idx = 0
        
        for msg in processed_messages:
            content = msg.get("content", "")
            # Replace audio placeholders with encoded tokens
            while "<|audio|>" in content and audio_idx < len(audios):
                audio_path = audios[audio_idx]
                
                # Ensure audio tokenizer model is in float32 mode for DeepSpeed Zero3 compatibility
                if hasattr(self.audio_tokenizer, 'whisper_model') and self.audio_tokenizer.whisper_model is not None:
                    current_dtype = next(self.audio_tokenizer.whisper_model.parameters()).dtype
                    if current_dtype != torch.float32:
                        logger.info(f"Converting audio tokenizer from {current_dtype} to float32 for Zero3 compatibility")
                        self.audio_tokenizer.whisper_model = self.audio_tokenizer.whisper_model.float()
                with torch.autocast(device_type="cuda", enabled=False):
                    audio_tokens = self.audio_tokenizer.encode(
                        audio_path, is_discrete=True, is_contiguous=False
                    )
                audio_tokens_str = "".join(f"<|audio_{i}|>" for i in audio_tokens)
                audio_replacement = f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>"
                content = content.replace("<|audio|>", audio_replacement, 1)
                audio_idx += 1
            msg["content"] = content
        
        return processed_messages

    # ---------------- logging override (COPIED FROM ORIGINAL) -----------------
    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        avg = {k: sum(v) / max(len(v), 1) for k, v in self._metrics.items()}
        logs.update(avg)
        super().log(logs, start_time)
        self._metrics.clear()
