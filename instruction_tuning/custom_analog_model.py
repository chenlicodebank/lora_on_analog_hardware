#!/usr/bin/env python3
"""
HFLM-based custom analog model for lm-evaluation-harness.
Inherits from HFLM to get efficient batched evaluation for free.
Only overrides model loading to add analog conversion, LoRA, and PCM noise support.
"""

import os
import sys
import torch
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

try:
    from unsloth import FastModel
    print("✓ Unsloth imported successfully")
except ImportError as e:
    print(f"✗ Failed to import unsloth: {e}")
    raise

try:
    from aihwkit_lightning.nn import AnalogLinear
    from aihwkit_lightning.nn.conversion import convert_to_analog
    from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
    from aihwkit_lightning.simulator.parameters.enums import (
        WeightNoiseInjectionType,
        WeightClipType,
        WeightQuantizationType,
    )
    ANALOG_AVAILABLE = True
    print("✓ aihwkit_lightning imported successfully")
except ImportError as e:
    print(f"✗ Failed to import aihwkit_lightning: {e}")
    ANALOG_AVAILABLE = False

try:
    from apply_noise_model import NoiseConfig, apply_noise
    PCM_AVAILABLE = True
    print("✓ PCM noise model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PCM noise model: {e}")
    PCM_AVAILABLE = False

eval_logger = logging.getLogger(__name__)


def gen_rpu_config(output_noise_level=0, weight_noise_level=0.067,
                   eval_with_noise=False):
    """Generate RPU config for aihwkit-lightning."""
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    rpu_config.modifier.std_dev = weight_noise_level
    rpu_config.modifier.noise_type = WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL

    # No quantization
    rpu_config.modifier.quantization_type = WeightQuantizationType.NONE
    rpu_config.modifier.res = 0.0

    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False
    rpu_config.clip.type = WeightClipType.NONE
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1

    # Enable input range only when output noise > 0
    if output_noise_level > 0:
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.learn_input_range = True
        rpu_config.pre_post.input_range.init_value = 3.0
        rpu_config.pre_post.input_range.decay = 0.001
        rpu_config.pre_post.input_range.input_min_percentage = 0.95
        rpu_config.pre_post.input_range.manage_output_clipping = False

    rpu_config.eval_with_noise = eval_with_noise
    return rpu_config


def _apply_eval_noise_patches(model, rpu_config_template):
    """Patch AnalogLinear layers to apply noise during eval mode.

    Since enable_during_test is deprecated in aihwkit-lightning 2.x, we
    temporarily set self.training = True so the library's own forward path
    applies noise (modify_weight) in the correct order.
    """
    original_forwards = {}

    def patched_forward(self, x):
        should_apply = (
            not self.training and
            hasattr(self, 'rpu_config') and
            self.rpu_config is not None and
            getattr(self.rpu_config, 'eval_with_noise', False)
        )
        if should_apply:
            ir_cfg = self.rpu_config.pre_post.input_range
            orig_ir_enable = ir_cfg.enable
            ir_cfg.enable = False

            self.training = True
            try:
                result = original_forwards[self](x)
            finally:
                self.training = False
                ir_cfg.enable = orig_ir_enable
            return result
        else:
            return original_forwards[self](x)

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and module not in original_forwards:
            original_forwards[module] = module.forward
            module.forward = patched_forward.__get__(module, AnalogLinear)
            count += 1
    print(f"Applied eval noise patches to {count} analog layers")


def _build_analog_model(model_name, use_analog, lora_path, output_noise_level,
                        weight_noise_level, eval_with_noise,
                        evaluate_with_pcm_model=False, pcm_max_inp_size=512,
                        max_seq_length=2048):
    """Build model with unsloth, LoRA, analog conversion, and optional PCM noise.
    Returns (model, tokenizer)."""

    print(f"Loading base model: {model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float32,
        load_in_4bit=False,
    )

    # Add LoRA structure (needed even for "before LoRA" to match architecture)
    if lora_path is None:
        print("Adding random-initialized LoRA adapters (before-training baseline)")
    else:
        print("Adding LoRA structure before loading trained weights")

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load trained LoRA weights if provided
    if lora_path and os.path.exists(lora_path):
        print(f"Loading trained LoRA adapter from {lora_path}")
        model.load_adapter(lora_path, adapter_name="trained_adapter")
        model.set_adapter("trained_adapter")
        print(f"Active adapter: {model.active_adapter}")

        model_dtype = next(model.parameters()).dtype
        for name, param in model.named_parameters():
            if 'lora_' in name and param.dtype != model_dtype:
                param.data = param.data.to(model_dtype)

    # Convert to analog if requested
    if use_analog and ANALOG_AVAILABLE:
        print("Converting to analog hardware simulation...")
        exclude_modules = []
        for name, module in model.named_modules():
            if 'lora_A' in name or 'lora_B' in name or name.endswith('lm_head'):
                exclude_modules.append(name)

        rpu_config = gen_rpu_config(
            output_noise_level=output_noise_level,
            weight_noise_level=weight_noise_level,
            eval_with_noise=eval_with_noise,
        )

        model = convert_to_analog(
            model, rpu_config,
            exclude_modules=exclude_modules,
            inplace=True,
        )
        print("Model converted to analog")

        if eval_with_noise:
            _apply_eval_noise_patches(model, rpu_config)

    # Apply PCM noise if requested (mutually exclusive with analog noise —
    # PCM directly modifies weights on a digital model)
    if evaluate_with_pcm_model:
        if not PCM_AVAILABLE:
            print("WARNING: PCM evaluation requested but apply_noise_model not available. Skipping.")
        else:
            print(f"Applying PCM noise (max_inp_size={pcm_max_inp_size})...")
            pcm_noise_config = NoiseConfig(
                noise_type="pcm",
                max_inp_size=pcm_max_inp_size,
            )
            apply_noise(model, pcm_noise_config)
            print("PCM noise applied to model weights")

    model.eval()
    return model, tokenizer


@register_model("custom_analog")
class CustomAnalogHFLM(HFLM):
    """
    Analog hardware simulation model that inherits HFLM's batched evaluation.
    Supports analog noise, PCM noise, and LoRA adapters.
    """

    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B",
        use_analog: bool = False,
        lora_path: str = None,
        output_noise_level: float = 0,
        weight_noise_level: float = 0.067,
        eval_with_noise: bool = False,
        evaluate_with_pcm_model: bool = False,
        pcm_max_inp_size: int = 512,
        **kwargs,
    ):
        # Parse string types from CLI model_args
        if isinstance(use_analog, str):
            use_analog = use_analog.lower() in ('true', '1', 'yes')
        if isinstance(eval_with_noise, str):
            eval_with_noise = eval_with_noise.lower() in ('true', '1', 'yes')
        if isinstance(evaluate_with_pcm_model, str):
            evaluate_with_pcm_model = evaluate_with_pcm_model.lower() in ('true', '1', 'yes')
        if isinstance(output_noise_level, str):
            output_noise_level = float(output_noise_level)
        if isinstance(weight_noise_level, str):
            weight_noise_level = float(weight_noise_level)
        if isinstance(pcm_max_inp_size, str):
            pcm_max_inp_size = int(pcm_max_inp_size)

        print(f"CustomAnalogHFLM config:")
        print(f"  model_name={model_name}, use_analog={use_analog}")
        print(f"  lora_path={lora_path}")
        print(f"  weight_noise={weight_noise_level}, output_noise={output_noise_level}")
        print(f"  eval_with_noise={eval_with_noise}")
        print(f"  evaluate_with_pcm_model={evaluate_with_pcm_model}, pcm_max_inp_size={pcm_max_inp_size}")

        # Build the model with analog conversion and/or PCM noise
        model, tokenizer = _build_analog_model(
            model_name=model_name,
            use_analog=use_analog,
            lora_path=lora_path,
            output_noise_level=output_noise_level,
            weight_noise_level=weight_noise_level,
            eval_with_noise=eval_with_noise,
            evaluate_with_pcm_model=evaluate_with_pcm_model,
            pcm_max_inp_size=pcm_max_inp_size,
        )

        # Pass pre-built model and tokenizer to HFLM
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            dtype=torch.float32,
            batch_size=kwargs.get('batch_size', 1),
            max_length=2048,
            trust_remote_code=True,
            **{k: v for k, v in kwargs.items() if k != 'batch_size'},
        )

        print(f"CustomAnalogHFLM initialized. Device: {self.device}")

    def _model_call(self, inps, attn_mask=None, labels=None, **kwargs):
        """Override to avoid transformers.AutoModelForVision2Seq check."""
        with torch.no_grad():
            call_kwargs = {}
            if attn_mask is not None:
                call_kwargs["attention_mask"] = attn_mask
            return self.model(inps, **call_kwargs).logits

    def _detect_batch_size(self, *args, **kwargs):
        """Skip auto-detection, return fixed batch size."""
        return self._batch_size if isinstance(self._batch_size, int) else 4
