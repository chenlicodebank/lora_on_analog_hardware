#!/usr/bin/env python3


import os
import sys
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
import logging

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import lm_eval dependencies
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    pad_and_concat,
    clear_torch_cache,
)

# Import unsloth and analog hardware dependencies
try:
    from unsloth import FastModel
    print("✓ Unsloth imported successfully")
except ImportError as e:
    print(f"✗ Failed to import unsloth: {e}")
    raise

try:
    import aihwkit_lightning
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

# Import PCM noise model dependencies
try:
    from apply_noise_model import NoiseConfig, apply_noise, add_PCM_noise
    PCM_AVAILABLE = True
    print("✓ PCM noise model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PCM noise model: {e}")
    PCM_AVAILABLE = False

eval_logger = logging.getLogger(__name__)


def show_adapter_weights(model, adapter_name=None):
    """Show some LoRA weights to verify different adapters have different weights."""
    if adapter_name:
        try:
            model.set_adapter(adapter_name)
        except:
            print(f"Could not set adapter to {adapter_name}")
            return
    
    # Find first LoRA layer and show some weights
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if hasattr(module.lora_A, model.active_adapter) and hasattr(module.lora_B, model.active_adapter):
                lora_a = getattr(module.lora_A, model.active_adapter)
                lora_b = getattr(module.lora_B, model.active_adapter)
                print(f"Adapter '{model.active_adapter}' - {name} - LoRA A weights sample: {lora_a.weight.data[:2, :2]}")
                print(f"Adapter '{model.active_adapter}' - {name} - LoRA B weights sample: {lora_b.weight.data[:2, :2]}")
                return
    print(f"No LoRA weights found for adapter '{model.active_adapter}'")


def gen_lightning_rpu_config(output_noise_level=0, weight_noise_level=0.067, eval_with_noise=False):
    """
    Generate RPU config for aihwkit-lightning
    
    Args:
        output_noise_level (float): Standard deviation of output noise (default: 0)
        weight_noise_level (float): Standard deviation of weight noise (default: 0.067)
        eval_with_noise (bool): Whether to apply noise during evaluation (default: False)
    
    Returns:
        TorchInferenceRPUConfig: Configured RPU config for analog inference
    """
    if not ANALOG_AVAILABLE:
        raise RuntimeError("aihwkit_lightning not available")
    
    eval_logger.info(f"Generating RPU config with:")
    eval_logger.info(f"  Weight noise std_dev: {weight_noise_level}")
    eval_logger.info(f"  Output noise level: {output_noise_level}")
    eval_logger.info(f"  Eval with noise: {eval_with_noise}")
        
    rpu_config = TorchInferenceRPUConfig()

    # Mapping configuration
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    # Weight modifier
    rpu_config.modifier.std_dev = weight_noise_level
    rpu_config.modifier.noise_type = WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL

    # Weight quantization
    rpu_config.modifier.quantization_type = WeightQuantizationType.NONE
    rpu_config.modifier.res = 0.0

    # Forward pass parameters
    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False

    # Weight clipping
    rpu_config.clip.type = WeightClipType.NONE

    # Input/Output quantization
    rpu_config.forward.inp_res = -1  # no input quantization
    rpu_config.forward.out_res = -1  # no output quantization

    # Add custom flag for evaluation noise
    rpu_config.eval_with_noise = eval_with_noise
    
    eval_logger.info("RPU config generated successfully")
    
    return rpu_config


@register_model("custom_analog")
class CustomAnalogModel(TemplateLM):
    """
    Custom model class that integrates with lm-evaluation-harness
    and supports analog hardware simulation and LoRA adapters.
    """
    
    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B",
        use_analog: bool = False,
        lora_path: Optional[str] = None,
        output_noise_level: float = 0,
        weight_noise_level: float = 0.067,
        eval_with_noise: bool = False,
        evaluate_with_pcm_model: bool = False,
        pcm_max_inp_size: int = 512,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
        **kwargs
    ):
        super().__init__()
        
        # Validate noise level parameters
        if weight_noise_level < 0:
            raise ValueError(f"weight_noise_level must be non-negative, got {weight_noise_level}")
        if weight_noise_level > 1.0:
            eval_logger.warning(f"Weight noise level {weight_noise_level} is unusually high (>1.0)")
        
        if output_noise_level < 0:
            raise ValueError(f"output_noise_level must be non-negative, got {output_noise_level}")
        
        # Validate PCM parameters
        if evaluate_with_pcm_model and not PCM_AVAILABLE:
            eval_logger.warning("PCM model evaluation requested but PCM modules not available. Disabling PCM evaluation.")
            evaluate_with_pcm_model = False
        
        if pcm_max_inp_size <= 0:
            raise ValueError(f"pcm_max_inp_size must be positive, got {pcm_max_inp_size}")
        
        self.model_name = model_name
        self.use_analog = use_analog and ANALOG_AVAILABLE
        self.lora_path = lora_path
        self.output_noise_level = output_noise_level
        self.weight_noise_level = weight_noise_level
        self.eval_with_noise = eval_with_noise
        self.evaluate_with_pcm_model = evaluate_with_pcm_model
        self.pcm_max_inp_size = pcm_max_inp_size
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size_per_gpu = batch_size
        self._max_length = max_length
        
        # PCM noise tracking
        self._pcm_original_weights = None
        self._pcm_noise_applied = False
        
        eval_logger.info(f"Initializing CustomAnalogModel:")
        eval_logger.info(f"  Model: {model_name}")
        eval_logger.info(f"  Use analog: {self.use_analog}")
        eval_logger.info(f"  LoRA path: {lora_path}")
        eval_logger.info(f"  Weight noise level: {weight_noise_level}")
        eval_logger.info(f"  Output noise level: {output_noise_level}")
        eval_logger.info(f"  Eval with noise: {eval_with_noise}")
        eval_logger.info(f"  Evaluate with PCM model: {self.evaluate_with_pcm_model}")
        eval_logger.info(f"  PCM max input size: {pcm_max_inp_size}")
        eval_logger.info(f"  Device: {self._device}")
        
        if self.use_analog and not ANALOG_AVAILABLE:
            eval_logger.warning("Analog hardware requested but aihwkit_lightning not available. Using digital model.")
            self.use_analog = False
        
        self._load_model()
        
    def _load_model(self):
        """Load the model with optional analog conversion and LoRA"""
        eval_logger.info("Loading base model...")
        
        # Load base model using unsloth
        # Use float32 to avoid mixed precision dtype conflicts
        self._model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self._max_length,
            dtype=torch.float32,  # Use float32 for consistency
            load_in_4bit=False,
        )
        
        # Get the model's native dtype for consistency
        model_dtype = next(self._model.parameters()).dtype
        eval_logger.info(f"Model loaded with dtype: {model_dtype}")
        
        # Add LoRA adapters if not loading from path
        if self.lora_path is None:
            eval_logger.info("Adding random initialized LoRA adapters...")
            self._model = FastModel.get_peft_model(
                self._model,
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
        
        # Load LoRA adapters if specified
        if self.lora_path and os.path.exists(self.lora_path):
            eval_logger.info(f"Loading LoRA adapters from {self.lora_path}...")
            
            # First add PEFT if not already done
            if not hasattr(self._model, 'peft_config'):
                self._model = FastModel.get_peft_model(
                    self._model,
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
            
            # Load the LoRA adapter
            self._model.load_adapter(self.lora_path, adapter_name="trained_adapter")
            self._model.set_adapter("trained_adapter")
            eval_logger.info(f"Activated LoRA adapter: {self._model.active_adapter}")
            
            # Ensure LoRA parameters have consistent dtype
            for name, param in self._model.named_parameters():
                if 'lora_' in name and param.dtype != model_dtype:
                    param.data = param.data.to(model_dtype)
                    eval_logger.info(f"Converted LoRA parameter {name} to {model_dtype}")
        
        # Convert to analog if requested
        if self.use_analog:
            eval_logger.info("Converting model to analog hardware...")
            
            # Generate exclusion list for LoRA and LM head
            exclude_modules = []
            for name, module in self._model.named_modules():
                if ('lora_A' in name or 'lora_B' in name or name.endswith('lm_head')):
                    exclude_modules.append(name)
            
            self._model = convert_to_analog(
                self._model,
                gen_lightning_rpu_config(
                    output_noise_level=self.output_noise_level,
                    weight_noise_level=self.weight_noise_level,
                    eval_with_noise=self.eval_with_noise
                ),
                exclude_modules=exclude_modules,
                inplace=True
            )
            eval_logger.info("Model converted to analog hardware")
            
            # Apply noise patching for evaluation if needed
            if self.eval_with_noise:
                self._apply_eval_noise_patches()
        
        # Set model to evaluation mode and move to device
        self._model.eval()
        self._model.to(self._device)
        
        eval_logger.info(f"Model loaded successfully on {self._device}")
        
        # Cache some properties
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = self.batch_size_per_gpu
        
    def _apply_eval_noise_patches(self):
        """Apply patches to enable noise during evaluation for analog layers"""
        if not self.use_analog:
            return
            
        original_forwards = {}
        
        def patched_forward(self, x):
            """Modified forward method that applies noise during evaluation"""
            should_apply_noise = (
                not self.training and  # In evaluation mode
                hasattr(self, 'rpu_config') and 
                self.rpu_config is not None and
                hasattr(self.rpu_config, 'modifier') and
                self.rpu_config.modifier.std_dev > 0 and
                getattr(self.rpu_config, 'eval_with_noise', False)
            )
            
            if should_apply_noise:
                original_weight = self.weight.data.clone()
                try:
                    modifier = self.rpu_config.modifier
                    assumed_wmax = self.weight.data.abs().amax(dim=1, keepdim=True)
                    
                    if modifier.noise_type.name == 'ADD_NORMAL_PER_CHANNEL':
                        noise = modifier.std_dev * assumed_wmax * torch.randn_like(self.weight.data)
                    else:
                        noise = modifier.std_dev * assumed_wmax.max() * torch.randn_like(self.weight.data)
                    
                    self.weight.data = original_weight + noise
                    result = original_forwards[self](x)
                    self.weight.data = original_weight
                    return result
                except Exception as e:
                    self.weight.data = original_weight
                    return original_forwards[self](x)
            else:
                return original_forwards[self](x)
        
        # Apply patches to all AnalogLinear layers
        for name, module in self._model.named_modules():
            if isinstance(module, AnalogLinear) and module not in original_forwards:
                original_forwards[module] = module.forward
                module.forward = patched_forward.__get__(module, AnalogLinear)
                
        eval_logger.info(f"Applied noise patches to {len(original_forwards)} analog layers")
    
    def update_weight_noise_level(self, new_weight_noise_level: float):
        """
        Update the weight noise level for analog layers after model initialization.
        
        Args:
            new_weight_noise_level (float): New weight noise standard deviation
        
        Returns:
            int: Number of analog layers updated
        """
        if not self.use_analog:
            eval_logger.warning("Cannot update weight noise level: model is not using analog hardware")
            return 0
        
        if new_weight_noise_level < 0:
            raise ValueError(f"weight_noise_level must be non-negative, got {new_weight_noise_level}")
        
        if new_weight_noise_level > 1.0:
            eval_logger.warning(f"Weight noise level {new_weight_noise_level} is unusually high (>1.0)")
        
        old_level = self.weight_noise_level
        self.weight_noise_level = new_weight_noise_level
        
        # Update RPU config for all analog layers
        updated_count = 0
        for name, module in self._model.named_modules():
            if isinstance(module, AnalogLinear) and hasattr(module, 'rpu_config'):
                module.rpu_config.modifier.std_dev = new_weight_noise_level
                updated_count += 1
        
        eval_logger.info(f"Updated weight noise level from {old_level} to {new_weight_noise_level} for {updated_count} analog layers")
        return updated_count
    
    def get_current_noise_config(self) -> dict:
        """
        Get the current noise configuration.
        
        Returns:
            dict: Current noise configuration parameters
        """
        config = {
            "weight_noise_level": self.weight_noise_level,
            "output_noise_level": self.output_noise_level,
            "eval_with_noise": self.eval_with_noise,
            "use_analog": self.use_analog,
            "evaluate_with_pcm_model": self.evaluate_with_pcm_model,
            "pcm_max_inp_size": self.pcm_max_inp_size
        }
        
        if self.use_analog:
            # Count analog layers
            analog_layer_count = sum(1 for _, module in self._model.named_modules() 
                                   if isinstance(module, AnalogLinear))
            config["analog_layer_count"] = analog_layer_count
        
        return config
    
    def load_and_inspect_lora_adapter(self, lora_path: str, adapter_name: str = "adapter_model", show_weights: bool = True):
        """
        Load a LoRA adapter with detailed weight inspection for debugging.
        
        Args:
            lora_path (str): Path to the LoRA adapter
            adapter_name (str): Name for the loaded adapter
            show_weights (bool): Whether to show weight samples before/after loading
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        if not os.path.exists(lora_path):
            eval_logger.error(f"LoRA path does not exist: {lora_path}")
            return False
        
        try:
            # Show default adapter weights before loading
            if show_weights:
                eval_logger.info("=== BEFORE LOADING ADAPTER ===")
                show_adapter_weights(self._model)
            
            # Load the adapter
            eval_logger.info(f"Loading LoRA adapter from {lora_path}...")
            self._model.load_adapter(lora_path, adapter_name=adapter_name)
            self._model.set_adapter(adapter_name)
            
            eval_logger.info(f"Loaded and activated LoRA adapter from: {lora_path}")
            eval_logger.info(f"Active adapter: {self._model.active_adapter}")
            
            # Show loaded adapter weights
            if show_weights:
                eval_logger.info("=== AFTER LOADING AND ACTIVATING ADAPTER ===")
                show_adapter_weights(self._model)
                
                # Compare with default adapter if it exists
                try:
                    eval_logger.info("=== DEFAULT ADAPTER WEIGHTS FOR COMPARISON ===")
                    show_adapter_weights(self._model, "default")
                    # Switch back to loaded adapter
                    self._model.set_adapter(adapter_name)
                except:
                    eval_logger.info("No default adapter available for comparison")
            
            # Ensure LoRA parameters have consistent dtype
            model_dtype = next(self._model.parameters()).dtype
            for name, param in self._model.named_parameters():
                if 'lora_' in name and param.dtype != model_dtype:
                    param.data = param.data.to(model_dtype)
                    eval_logger.info(f"Converted LoRA parameter {name} to {model_dtype}")
            
            # Update stored lora_path
            self.lora_path = lora_path
            
            eval_logger.info("LoRA adapter loaded successfully with weight inspection")
            return True
            
        except Exception as e:
            eval_logger.error(f"Failed to load LoRA adapter: {e}")
            return False
    
    def show_adapter_weights(self, adapter_name=None):
        """
        Show LoRA weights for the current or specified adapter.
        
        Args:
            adapter_name (str, optional): Name of adapter to inspect. Uses current if None.
        """
        show_adapter_weights(self._model, adapter_name)
    
    def apply_pcm_noise(self):
        """
        Apply PCM noise to the model for evaluation.
        
        Returns:
            bool: True if PCM noise was successfully applied, False otherwise
        """
        if not self.evaluate_with_pcm_model:
            eval_logger.warning("PCM model evaluation not enabled")
            return False
        
        if not PCM_AVAILABLE:
            eval_logger.error("PCM noise modules not available")
            return False
        
        if self._pcm_noise_applied:
            eval_logger.warning("PCM noise already applied")
            return True
        
        try:
            eval_logger.info(f"Applying PCM noise to model...")
            eval_logger.info(f"  PCM max input size: {self.pcm_max_inp_size}")
            
            pcm_noise_config = NoiseConfig(
                noise_type="pcm",
                max_inp_size=self.pcm_max_inp_size
            )
            
            self._pcm_original_weights = apply_noise(self._model, pcm_noise_config)
            self._pcm_noise_applied = True
            
            eval_logger.info(f"PCM noise applied to {len(self._pcm_original_weights)} base layers (LoRA adapters excluded)")
            return True
            
        except Exception as e:
            eval_logger.error(f"Failed to apply PCM noise: {e}")
            self._pcm_original_weights = None
            self._pcm_noise_applied = False
            return False
    
    def restore_original_weights(self):
        """
        Restore original weights after PCM noise evaluation.
        
        Returns:
            bool: True if weights were successfully restored, False otherwise
        """
        if not self._pcm_noise_applied:
            eval_logger.debug("No PCM noise applied, nothing to restore")
            return True
        
        if self._pcm_original_weights is None:
            eval_logger.error("No original weights stored for restoration")
            return False
        
        try:
            eval_logger.info("Restoring original weights after PCM evaluation...")
            
            for module, original_weight in self._pcm_original_weights.items():
                module.weight.data = original_weight
            
            eval_logger.info("Original weights restored")
            self._pcm_original_weights = None
            self._pcm_noise_applied = False
            return True
            
        except Exception as e:
            eval_logger.error(f"Failed to restore original weights: {e}")
            return False
    
    @property
    def dtype(self):
        """Get the model's dtype"""
        # Get dtype from a non-embedding parameter to avoid issues with mixed precision
        for name, param in self.model.named_parameters():
            if 'embed' not in name.lower() and param.requires_grad:
                return param.dtype
        # Fallback to first parameter if no suitable parameter found
        return next(self.model.parameters()).dtype
    
    @property
    def device(self):
        return self._device
    
    @property
    def eot_token_id(self):
        """End of text token ID required by TemplateLM"""
        return self.tokenizer.eos_token_id
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer_name(self) -> str:
        return self.model_name
    
    @property
    def max_length(self):
        return self._max_length
    
    def tok_encode(self, string: str, **kwargs):
        """Tokenize a string"""
        encoding = self.tokenizer.encode(string, add_special_tokens=False)
        return encoding
    
    def tok_decode(self, tokens, **kwargs):
        """Decode tokens to string"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def _model_call(self, inps: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Run forward pass on the model
        """
        with torch.no_grad():
            # Token indices should always remain as integers (long)
            # Only move to device, don't change dtype for input tokens
            inps = inps.to(device=self.device)
            
            # Ensure input tokens are long integers (required by embedding layers)
            if inps.dtype != torch.long:
                inps = inps.long()
            
            kwargs = {}
            if attn_mask is not None:
                # Use float32 for attention mask (standard for attention computations)
                kwargs["attention_mask"] = attn_mask.to(device=self.device, dtype=torch.float32)
            
            try:
                outputs = self.model(inps, **kwargs)
                return outputs.logits
            except RuntimeError as e:
                if "dtype" in str(e):
                    eval_logger.error(f"Dtype error in forward pass: {e}")
                    eval_logger.error(f"Input dtype: {inps.dtype}")
                    if attn_mask is not None:
                        eval_logger.error(f"Attention mask dtype: {kwargs['attention_mask'].dtype}")
                    eval_logger.error(f"Model dtype: {next(self.model.parameters()).dtype}")
                raise e
    
    def _model_generate(self, context: torch.Tensor, max_length: int, stop: List[str] = None, **kwargs):
        """
        Generate text from the model
        """
        # Token indices should always remain as integers (long)
        # Only move to device, don't change dtype for context tokens
        context = context.to(device=self.device)
        
        # Ensure context tokens are long integers (required by embedding layers)
        if context.dtype != torch.long:
            context = context.long()
        
        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": max_length,
            "do_sample": False,  # Greedy decoding for consistency
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs.update(kwargs)
        
        with torch.no_grad():
            outputs = self.model.generate(context, **generation_kwargs)
        
        # Extract only the generated part
        generated = outputs[:, context.shape[1]:]
        return generated
    
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of generating continuations from contexts
        """
        # Apply PCM noise if requested for evaluation
        pcm_applied = False
        if self.evaluate_with_pcm_model:
            pcm_applied = self.apply_pcm_noise()
            if not pcm_applied:
                eval_logger.warning("Failed to apply PCM noise, proceeding without PCM")
        
        try:
            new_reqs = []
            for context, continuation in [req.args for req in requests]:
                if context == "":
                    # BOS or EOS as context
                    context_enc, continuation_enc = (
                        [self.tokenizer.bos_token_id],
                        self.tok_encode(continuation),
                    )
                else:
                    context_enc, continuation_enc = (
                        self.tok_encode(context),
                        self.tok_encode(continuation),
                    )
                
                new_reqs.append(((context, continuation), context_enc, continuation_enc))
            
            return self._loglikelihood_tokens(new_reqs)
        
        finally:
            # Restore original weights if PCM noise was applied
            if pcm_applied:
                self.restore_original_weights()
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        Compute rolling log-likelihood for perplexity computation
        """
        # Apply PCM noise if requested for evaluation
        pcm_applied = False
        if self.evaluate_with_pcm_model:
            pcm_applied = self.apply_pcm_noise()
            if not pcm_applied:
                eval_logger.warning("Failed to apply PCM noise, proceeding without PCM")
        
        try:
            loglikelihoods = []
            
            for request in requests:
                rolling_token_windows = []
                string = request.args[0]
                
                # Encode the full string
                encoding = self.tok_encode(string)
                
                # Process in chunks that fit within max_length
                for i in range(len(encoding)):
                    # Create context up to position i
                    if i == 0:
                        # First token gets BOS context
                        inp = [self.tokenizer.bos_token_id] + encoding[:1]
                        target = encoding[0]
                    else:
                        # Subsequent tokens get previous context
                        start_idx = max(0, i - self.max_length + 1)
                        inp = encoding[start_idx:i+1]
                        target = encoding[i]
                    
                    rolling_token_windows.append((inp, target))
                
                # Compute log-likelihood for each position
                rolling_loglikelihoods = []
                for inp, target in rolling_token_windows:
                    inp_tensor = torch.tensor([inp], device=self.device, dtype=torch.long)
                    
                    with torch.no_grad():
                        logits = self._model_call(inp_tensor)
                        
                    # Get logits for the target position
                    target_logits = logits[0, -2, :]  # -2 because we want logits for predicting the last token
                    
                    # Convert to float32 for numerical stability and consistent dtypes
                    target_logits = target_logits.float()
                    
                    log_probs = F.log_softmax(target_logits, dim=-1)
                    
                    rolling_loglikelihoods.append(log_probs[target].item())
                
                loglikelihoods.append(sum(rolling_loglikelihoods))
            
            return loglikelihoods
        
        finally:
            # Restore original weights if PCM noise was applied
            if pcm_applied:
                self.restore_original_weights()
    
    def _loglikelihood_tokens(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for token sequences
        """
        results = []
        
        for (context, continuation), context_enc, continuation_enc in requests:
            # Prepare input sequence
            full_enc = context_enc + continuation_enc
            
            # Handle sequences longer than max_length
            if len(full_enc) > self.max_length:
                # Truncate from the left, keeping the continuation
                start_idx = len(full_enc) - self.max_length
                if start_idx > len(context_enc):
                    # If we have to truncate into the continuation, skip this request
                    results.append((float('-inf'), False))
                    continue
                full_enc = full_enc[start_idx:]
                context_enc = context_enc[max(0, len(context_enc) - (self.max_length - len(continuation_enc))):]
            
            # Convert to tensor
            inp = torch.tensor([full_enc], device=self.device, dtype=torch.long)
            
            # Get logits
            with torch.no_grad():
                logits = self._model_call(inp)
            
            # Compute log-likelihood for the continuation part
            logits = logits[0]  # Remove batch dimension
            
            # Get logits for predicting continuation tokens
            continuation_logits = logits[len(context_enc)-1:len(context_enc)-1+len(continuation_enc)]
            continuation_tokens = torch.tensor(continuation_enc, device=self.device, dtype=torch.long)
            
            # Convert logits to float32 for numerical stability and consistent dtypes
            continuation_logits = continuation_logits.float()
            
            # Compute log probabilities
            log_probs = F.log_softmax(continuation_logits, dim=-1)
            
            # Sum log probabilities for the continuation
            loglikelihood = 0.0
            for i, token in enumerate(continuation_tokens):
                if i < log_probs.shape[0]:
                    loglikelihood += log_probs[i, token].item()
            
            # Check if this would be the greedy choice
            greedy_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = torch.equal(greedy_tokens[:len(continuation_tokens)], continuation_tokens)
            
            results.append((loglikelihood, is_greedy))
        
        return results
    
    def generate_until(self, requests) -> List[str]:
        """
        Generate text until stopping criteria are met
        """
        # Apply PCM noise if requested for evaluation
        pcm_applied = False
        if self.evaluate_with_pcm_model:
            pcm_applied = self.apply_pcm_noise()
            if not pcm_applied:
                eval_logger.warning("Failed to apply PCM noise, proceeding without PCM")
        
        try:
            results = []
            
            for request in requests:
                context, gen_kwargs = request.args
                
                # Encode context
                context_enc = self.tok_encode(context)
                context_tensor = torch.tensor([context_enc], device=self.device, dtype=torch.long)
                
                # Extract generation parameters
                max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
                until = gen_kwargs.get("until", [self.tokenizer.eos_token])
                
                # Generate
                generated_tokens = self._model_generate(
                    context_tensor, 
                    max_length=max_gen_toks,
                    stop=until
                )
                
                # Decode generated text
                generated_text = self.tok_decode(generated_tokens[0].tolist())
                
                # Apply stopping criteria
                if until:
                    for stop_str in until:
                        if stop_str in generated_text:
                            generated_text = generated_text.split(stop_str)[0]
                            break
                
                results.append(generated_text)
            
            return results
        
        finally:
            # Restore original weights if PCM noise was applied
            if pcm_applied:
                self.restore_original_weights()


if __name__ == "__main__":
    # Test the model loading
    print("Testing CustomAnalogModel...")
    
    # model = CustomAnalogModel(
    #     model_name="unsloth/Meta-Llama-3.1-8B",
    #     use_analog=False,  # Start with digital for testing
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    
    # print("✓ Model loaded successfully!")
    # print(f"Device: {model.device}")
    # print(f"Tokenizer vocab size: {model.vocab_size}")


    # Test with different weight noise levels
    print("\n=== Testing Digital Model ===")
    model_digital = CustomAnalogModel(
        model_name="unsloth/Meta-Llama-3.1-8B",
        use_analog=False,  # Digital model
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("✓ Digital model loaded successfully!")
    print(f"Device: {model_digital.device}")
    print(f"Tokenizer vocab size: {model_digital.vocab_size}")
    print(f"Noise config: {model_digital.get_current_noise_config()}")
    
    if ANALOG_AVAILABLE:
        print("\n=== Testing Analog Model with Different Noise Levels ===")
        
        # Test with default noise level
        model_analog_default = CustomAnalogModel(
            model_name="unsloth/Meta-Llama-3.1-8B",
            use_analog=True,
            weight_noise_level=0.067,  # Default
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✓ Analog model with default noise (0.067) loaded!")
        print(f"Noise config: {model_analog_default.get_current_noise_config()}")
        
        # Test with custom noise level
        model_analog_custom = CustomAnalogModel(
            model_name="unsloth/Meta-Llama-3.1-8B",
            use_analog=True,
            weight_noise_level=0.1,  # Custom higher noise
            output_noise_level=0.01,
            eval_with_noise=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✓ Analog model with custom noise (0.1) loaded!")
        print(f"Noise config: {model_analog_custom.get_current_noise_config()}")
        
        # Test dynamic noise level update
        print("\n=== Testing Dynamic Noise Level Update ===")
        print("Updating weight noise level to 0.05...")
        model_analog_custom.update_weight_noise_level(0.05)
        print(f"Updated noise config: {model_analog_custom.get_current_noise_config()}")
        
        print("\nWeight noise control features:")
        print("✓ Configurable via --model_args weight_noise_level=X.X")
        print("✓ Runtime validation and warnings")
        print("✓ Detailed logging of noise configuration")
        print("✓ Dynamic update capability")
        print("✓ Configuration inspection via get_current_noise_config()")
    else:
        print("\n⚠️  aihwkit_lightning not available - analog features not tested") 
    
    if PCM_AVAILABLE:
        print("\n=== Testing PCM Model Evaluation ===")
        
        # Test with PCM model evaluation enabled
        model_pcm = CustomAnalogModel(
            model_name="unsloth/Meta-Llama-3.1-8B",
            use_analog=False,  # Can use PCM with digital models too
            evaluate_with_pcm_model=True,
            pcm_max_inp_size=512,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✓ Model with PCM evaluation enabled loaded!")
        print(f"Noise config: {model_pcm.get_current_noise_config()}")
        
        # Test PCM noise application and restoration
        print("\n=== Testing PCM Noise Application ===")
        print("Applying PCM noise...")
        pcm_success = model_pcm.apply_pcm_noise()
        print(f"PCM noise application success: {pcm_success}")
        print(f"PCM noise applied: {model_pcm._pcm_noise_applied}")
        
        print("Restoring original weights...")
        restore_success = model_pcm.restore_original_weights()
        print(f"Weight restoration success: {restore_success}")
        print(f"PCM noise applied after restore: {model_pcm._pcm_noise_applied}")
        
        # Test combined analog + PCM
        if ANALOG_AVAILABLE:
            print("\n=== Testing Combined Analog + PCM Model ===")
            model_combined = CustomAnalogModel(
                model_name="unsloth/Meta-Llama-3.1-8B",
                use_analog=True,
                weight_noise_level=0.05,
                eval_with_noise=True,
                evaluate_with_pcm_model=True,
                pcm_max_inp_size=256,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("✓ Combined analog + PCM model loaded!")
            print(f"Combined config: {model_combined.get_current_noise_config()}")
        
        print("\nPCM model evaluation features:")
        print("✓ Configurable via --model_args evaluate_with_pcm_model=True")
        print("✓ Customizable PCM input size via pcm_max_inp_size")
        print("✓ Automatic application during evaluation methods")
        print("✓ Automatic weight restoration after evaluation")
        print("✓ Compatible with both digital and analog models")
        print("✓ Detailed logging of PCM operations")
    else:
        print("\n⚠️  PCM noise model modules not available - PCM features not tested")
    
    print("\n=== Testing LoRA Adapter Weight Inspection ===")
    # Test showing adapter weights for digital model (which has LoRA)
    print("Testing adapter weight inspection on digital model...")
    model_digital.show_adapter_weights()
    
    print("\nLoRA adapter inspection features:")
    print("✓ Show LoRA weights via show_adapter_weights() method")
    print("✓ Enhanced LoRA loading with load_and_inspect_lora_adapter()")
    print("✓ Before/after weight comparison during adapter loading")
    print("✓ Automatic dtype conversion for LoRA parameters")
    print("✓ Detailed logging of adapter operations") 