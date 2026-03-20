#!/usr/bin/env python3
"""
GRPO Training with Analog Hardware Support

This script implements Group Relative Policy Optimization (GRPO) training
for language models on GSM8K dataset with support for analog hardware simulation/evaluation using aihwkit-lightning.
"""

# - wandb related imports
import wandb

# - general imports
import os
import glob
import argparse
import sys
import copy

# - torch related imports
import torch    
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import Optimizer, AdamW

# - unsloth related imports
from unsloth import FastModel

# - PCM noise model imports
from apply_noise_model import NoiseConfig, apply_noise, add_PCM_noise

# - aihwkit_lightning related imports
import aihwkit_lightning
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.nn import AnalogSequential
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.simulator.configs import (
    WeightModifierParameter,
    MappingParameter,
    IOParameters,
)
from aihwkit_lightning.simulator.parameters.enums import (
    WeightNoiseInjectionType,
    WeightClipType,
    WeightQuantizationType,
    WeightModifierType,
)

# - get arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument('--use_analog', action="store_true", default=False,
                        help="Whether to use analog hardware model")
    parser.add_argument('--pcm_model', type=str, default="PCM_Gmax25", help="PCM model to use")
    parser.add_argument('--output_noise_level', type=float, default=0, help="Noise level to use in RPU config")
    parser.add_argument('--weight_noise_level', type=float, default=0, 
                        help="Weight noise standard deviation (default: 0)")
    parser.add_argument('--analog_optimizer', type=str, default=None,
                        help="Analog optimizer to use (options: 'AnalogSGD', 'AnalogAdam'). If not provided, no analog optimizer is used.")
    parser.add_argument('--analog_lr', type=float, default=0.000005,
                        help="Learning rate for the analog optimizer")
    parser.add_argument('--debug', action="store_true", default=False,
                        help="Debug mode")
    # Evaluation mode arguments
    parser.add_argument('--mode', type=str, default='train_only', 
                        choices=['train_only', 'eval_original', 'train_and_eval', 'eval_lora'],
                        help="Execution mode: train_only, eval_original, train_and_eval, eval_lora")
    parser.add_argument('--lora_path', type=str, default="saved_lora_models",
                        help="Path to the LoRA adapter for eval_lora mode")
    
    # Noise control during evaluation
    parser.add_argument('--eval_with_noise', action="store_true", default=False,
                        help="Whether to apply noise during evaluation mode (normally noise is disabled during eval)")
    
    # PCM model evaluation
    parser.add_argument('--evaluate_with_pcm_model', action="store_true", default=False,
                        help="Whether to evaluate with PCM noise model applied (default: False)")
    parser.add_argument('--pcm_max_inp_size', type=int, default=512,
                        help="Max input size for PCM noise model tiling (default: 512)")
    parser.add_argument('--discretised_state', type=int, default=0,
        help="Number of discrete weight states (0=disabled, 12=Runze's config)")
    parser.add_argument('--weight_clip_type', type=str, default='none',
        choices=['none', 'layer_gaussian'])
    parser.add_argument('--weight_clip_sigma', type=float, default=2.0)

    # Evaluation configuration
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument('--eval_samples', type=int, default=100,
                        help="Number of samples to evaluate (use -1 for full test set)")
    return parser.parse_args()


args = parse_args()  # parse command line arguments

# Derive run name from WANDB_NAME for organizing outputs
run_name = os.environ.get("WANDB_NAME", "default")
output_dir = os.path.join("outputs", run_name)
lora_save_dir = os.path.join("saved_lora_models", run_name)


print(f"Running in mode: {args.mode}")
print(f"Configuration:")
print(f"  Use analog: {args.use_analog}")
if args.use_analog:
    print(f"  Weight noise level: {args.weight_noise_level}")
    print(f"  Output noise level: {args.output_noise_level}")
    print(f"  Eval with noise: {args.eval_with_noise}")
    print(f"  Analog optimizer: {args.analog_optimizer}")
    print(f"  Analog learning rate: {args.analog_lr}")
print(f"  Evaluate with PCM model: {args.evaluate_with_pcm_model}")
if args.evaluate_with_pcm_model:
    print(f"  PCM max input size: {args.pcm_max_inp_size}")
print(f"  Debug mode: {args.debug}")
print(f"  Run name: {run_name}")
print(f"  Output dir: {output_dir}")
print(f"  LoRA save dir: {lora_save_dir}")

# Use lora_save_dir as default lora_path if not explicitly provided
if args.lora_path == "saved_lora_models":
    args.lora_path = lora_save_dir

# Validate requirements for eval_lora mode
if args.mode == 'eval_lora' and not os.path.exists(args.lora_path):
    print(f"Error: LoRA path '{args.lora_path}' not found for eval_lora mode")
    sys.exit(1)

# - wandb related functions
def get_last_run_id():
    """
    Retrieve the last recorded run ID from the WandB cache.
    """
    print(f"Retrieving last run ID from WandB cache...")
    # Look in the local wandb directory first
    local_wandb_dir = os.path.join(os.getcwd(), "wandb")
    run_id = None

    if os.path.exists(local_wandb_dir):
        # Get all run directories
        run_dirs = [d for d in os.listdir(local_wandb_dir) if d.startswith("run-")]
        if run_dirs:
            # Sort by creation time (newest first)
            run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(local_wandb_dir, x)), reverse=True)
            # Extract run ID from the most recent directory name
            latest_run = run_dirs[0]
            run_id = latest_run.split("-")[-1]
            print(f"Found run ID from local directory: {run_id}")

    # If no run found locally, try the global wandb directory
    if not run_id:
        wandb_dir = os.path.expanduser("~/.wandb/wandb")
        if os.path.exists(wandb_dir):
            run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
            if run_dirs:
                run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(wandb_dir, x)), reverse=True)
                latest_run = run_dirs[0]
                run_id = latest_run.split("-")[-1]
                print(f"Found run ID from global directory: {run_id}")

    if not run_id:
        print("No previous WandB run found")
    return run_id


def init_wandb(args):
    wandb.login()

    # Determine if we are in a distributed setting
    is_rank_0 = True  # Single-node training only

    if is_rank_0:
        # Retrieve the last run ID if --resume is set
        wandb_run_id = get_last_run_id() if args.resume else None

        if args.resume and wandb_run_id:
            print(f"Resuming previous WandB run: {wandb_run_id}")
            # Initialize WandB with resumption
            wandb.init(
                project="analora",
                name=os.environ.get("WANDB_NAME", os.path.basename(os.getcwd())),
                resume="allow",
                id=wandb_run_id,
            )
        else:
            print("Starting a new WandB run.")
            # Initialize WandB as new run
            wandb.init(
                project="analora",
                name=os.environ.get("WANDB_NAME", os.path.basename(os.getcwd())),
            )

        # Store the run ID if resumed
        if wandb_run_id:
            os.environ["WANDB_RUN_ID"] = wandb_run_id

    else:
        # Disable WandB for non-primary processes to prevent conflicts
        wandb.init(mode="disabled")


init_wandb(args)


def gen_lightning_rpu_config(output_noise_level=0, weight_noise_level=0.03,
                            eval_with_noise=False, discretised_state=0,
                            weight_clip_type='none', weight_clip_sigma=2.0):
    """
    Generate RPU config for aihwkit-lightning

    Args:
        output_noise_level (float): Standard deviation of output noise (default: 0)
        weight_noise_level (float): Standard deviation of weight noise (default: 0.03)
        eval_with_noise (bool): Whether to apply noise during evaluation (default: False)
        discretised_state (int): Number of discrete weight states (0=disabled, 12=Runze's config)
        weight_clip_type (str): Weight clipping type ('none' or 'layer_gaussian')
        weight_clip_sigma (float): Sigma for layer_gaussian clipping (default: 2.0)

    Returns:
        TorchInferenceRPUConfig: Configured RPU config for analog inference
    """
    print(f"Generating RPU config with:")
    print(f"  Weight noise std_dev: {weight_noise_level}")
    print(f"  Output noise level: {output_noise_level}")
    print(f"  Eval with noise: {eval_with_noise}")
    print(f"  Discretised state: {discretised_state}")
    print(f"  Weight clip type: {weight_clip_type}")
    print(f"  Weight clip sigma: {weight_clip_sigma}")

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

    # CRITICAL: Enable noise during evaluation/test time when eval_with_noise=True
    rpu_config.modifier.enable_during_test = eval_with_noise

    # Weight quantization / discretization
    if discretised_state > 0:
        # DISCRETIZE_PER_CHANNEL scales each output channel by its own max weight,
        # giving all n_states levels to every channel.
        rpu_config.modifier.quantization_type = WeightQuantizationType.DISCRETIZE_PER_CHANNEL
        rpu_config.modifier.res = 1.0 / discretised_state
    else:
        rpu_config.modifier.quantization_type = WeightQuantizationType.NONE
        rpu_config.modifier.res = 0.0

    # Forward pass parameters
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False

    # Weight clipping
    if weight_clip_type == 'layer_gaussian':
        rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
        rpu_config.clip.sigma = weight_clip_sigma
    else:
        rpu_config.clip.type = WeightClipType.NONE

    # Input/Output quantization
    rpu_config.forward.inp_res = -1  # no input quantization
    rpu_config.forward.out_res = -1  # no output quantization

    # Input range clipping: only enable when output noise is used.
    # When enabled, input_range is initialized to init_value and calibrated from data
    # during forward passes where training=True. However, model.generate() internally
    # sets training=False, so input_range never gets calibrated during generation.
    # The uncalibrated input_range clips activations and destroys model output
    # when combined with weight discretization.
    needs_input_range = output_noise_level > 0
    rpu_config.pre_post.input_range.enable = needs_input_range
    rpu_config.pre_post.input_range.learn_input_range = True
    rpu_config.pre_post.input_range.init_value = 3.0
    rpu_config.pre_post.input_range.init_from_data = 100
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.decay = 0.001
    rpu_config.pre_post.input_range.input_min_percentage = 0.95

    # Add our custom flag for evaluation noise
    rpu_config.eval_with_noise = eval_with_noise

    print("RPU config generated successfully")

    return rpu_config


def patch_aihwkit_lightning_for_eval_noise():
    """
    Monkey-patch AnalogLinear forward method to apply noise and discretization
    during evaluation. Since enable_during_test is deprecated in
    aihwkit-lightning 2.x, we temporarily set self.training = True so the
    library's own forward path applies both discretization (clip_and_quantize)
    and noise (modify_weight) in the correct order: discretize -> noise.
    """
    original_forwards = {}

    def patched_forward(self, x):
        """Forward that forces the library's training-mode weight processing during eval."""
        should_apply = (
            not self.training and
            hasattr(self, 'rpu_config') and
            self.rpu_config is not None and
            getattr(self.rpu_config, 'eval_with_noise', False)
        )

        if should_apply:
            # Temporarily pretend we are in training mode so the library's
            # TorchLinear.linear applies clip_and_quantize AND modify_weight.
            # We disable input_range updates to avoid side-effects.
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

    def apply_patches(model):
        """Apply forward patches to all AnalogLinear layers"""
        for name, module in model.named_modules():
            if isinstance(module, AnalogLinear) and module not in original_forwards:
                original_forwards[module] = module.forward
                module.forward = patched_forward.__get__(module, AnalogLinear)

                if args.debug:
                    print(f"Patched AnalogLinear forward method for {name}")

    return apply_patches


max_seq_length = 1024

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    # model_name = "unsloth/gemma-3-1b-it",
    # model_name= "unsloth/Llama-3.1-8B",
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    # model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

"""We now add LoRA adapters so we only need to update a small amount of parameters!"""

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# Debug: Show initial adapter state
print(f"After get_peft_model - Active adapter: {model.active_adapter}")
print(f"Available adapters: {list(model.peft_config.keys())}")
if model.active_adapter == "default":
    print("Default adapter does not change output, so model is still the original model, unless trained/loaded with trainedLoRA")


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

if args.mode == 'eval_lora':
    # Show default adapter weights before loading
    print("=== BEFORE LOADING ADAPTER ===")
    show_adapter_weights(model)
    
    model.load_adapter(args.lora_path, adapter_name="adapter_model")
    model.set_adapter("adapter_model")  # Set the loaded adapter as active
    print(f"Loaded and activated LoRA adapter from: {args.lora_path}")
    print(f"Active adapter: {model.active_adapter}")
    
    # Show loaded adapter weights
    print("=== AFTER LOADING AND ACTIVATING ADAPTER ===")
    show_adapter_weights(model)
    
    # Compare with default adapter
    print("=== DEFAULT ADAPTER WEIGHTS FOR COMPARISON ===")
    show_adapter_weights(model, "default")
    # Switch back to loaded adapter
    model.set_adapter("adapter_model")
    
    model.to("cuda")




def get_current_noise_config(model, args):
    """
    Get the current noise configuration.
    
    Args:
        model: The model to inspect
        args: Command line arguments
    
    Returns:
        dict: Current noise configuration parameters
    """
    config = {
        "weight_noise_level": args.weight_noise_level,
        "output_noise_level": args.output_noise_level,
        "eval_with_noise": args.eval_with_noise,
        "use_analog": args.use_analog,
        "evaluate_with_pcm_model": args.evaluate_with_pcm_model
    }
    
    if args.use_analog:
        # Count analog layers
        analog_layer_count = sum(1 for _, module in model.named_modules() 
                               if isinstance(module, AnalogLinear))
        config["analog_layer_count"] = analog_layer_count
    
    if args.evaluate_with_pcm_model:
        config["pcm_max_inp_size"] = args.pcm_max_inp_size
    
    return config





# Convert model to analog (only if use_analog is True)
if args.use_analog:
    # Automatically generate exclusion list
    exclude_modules = []

    # Get all module names and filter for lora_A, lora_B, and lm_head
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name or name.endswith('lm_head')):
            exclude_modules.append(name)

    rpu_config = gen_lightning_rpu_config(
        output_noise_level=args.output_noise_level,
        weight_noise_level=args.weight_noise_level,
        eval_with_noise=args.eval_with_noise,
        discretised_state=args.discretised_state,
        weight_clip_type=args.weight_clip_type,
        weight_clip_sigma=args.weight_clip_sigma,
    )
    model = convert_to_analog(
        model,
        rpu_config,
        exclude_modules=exclude_modules,
        inplace=True  # This avoids the deepcopy that's causing OOM
    )
    print("Model converted to analog hardware")
    print("rpu config:")
    print(rpu_config)
    
    # Configure noise during evaluation if eval_with_noise is enabled
    if args.eval_with_noise:
        # Get the patching function and apply it to the model
        apply_patches_func = patch_aihwkit_lightning_for_eval_noise()
        apply_patches_func(model)
        print(f"Patched AnalogLinear layers for noise during evaluation (eval_with_noise={args.eval_with_noise})")
    else:
        print("Noise will be disabled during evaluation mode")
        
    # Log current noise configuration
    print("\n" + "="*50)
    print("NOISE CONFIGURATION")
    print("="*50)
    noise_config = get_current_noise_config(model, args)
    for key, value in noise_config.items():
        print(f"  {key}: {value}")
    print("="*50)
else:
    print("Using digital model (no analog conversion)")


if args.use_analog:
    print("Analog model with LORA (After selective conversion)")
else:
    print("Digital model with LORA")
print(model)

model.print_trainable_parameters()
model.to("cuda")

# Get analog layers after conversion to analog to be used for analog optimizer
if args.use_analog:
    analog_layers_after_conversion = []
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            analog_layers_after_conversion.append(module)

    print(f"Found {len(analog_layers_after_conversion)} analog layers after conversion")

    if args.analog_optimizer == "AnalogSGD":
        if len(analog_layers_after_conversion) > 0:
            # optimizer = AnalogOptimizer(SGD, analog_layers_after_conversion, model.parameters(), lr=args.analog_lr, momentum=0.9)
            # Create a callable function that returns the analog layers
            analog_layers_func = lambda: analog_layers_after_conversion
            optimizer = AnalogOptimizer(SGD, analog_layers_func, model.parameters(), lr=args.analog_lr, momentum=0.9)
            print(f"Created AnalogSGD optimizer with {len(analog_layers_after_conversion)} analog layers")
        else:
            print("No analog layers found, using regular SGD")
            optimizer = SGD(model.parameters(), lr=args.analog_lr, momentum=0.9)
    elif args.analog_optimizer == "AnalogAdam":
        if len(analog_layers_after_conversion) > 0:
            # optimizer = AnalogOptimizer(AdamW, analog_layers_after_conversion, model.parameters(), lr=args.analog_lr)
            # optimizer = AnalogOptimizer(AdamW, analog_layers_after_conversion, model.parameters(),
            #                             lr=args.analog_lr, betas=(0.9, 0.99), weight_decay=0.1)
            # Create a callable function that returns the analog layers
            analog_layers_func = lambda: analog_layers_after_conversion
            optimizer = AnalogOptimizer(AdamW, analog_layers_func, model.parameters(),
                                        lr=args.analog_lr, betas=(0.9, 0.99), weight_decay=0.1)
            print(f"Created AnalogAdam optimizer with {len(analog_layers_after_conversion)} analog layers")
        else:
            print("No analog layers found, using regular AdamW")
            optimizer = AdamW(model.parameters(), lr=args.analog_lr)
        if args.debug:
            # Create debug optimizer with zero learning rate for testing
            analog_layers_func = lambda: analog_layers_after_conversion
            optimizer = AnalogOptimizer(AdamW, analog_layers_func, model.parameters(), lr=0, weight_decay=0)
            print(f"Created debug AnalogAdam optimizer with {len(analog_layers_after_conversion)} analog layers, lr=0, weight_decay=0")
    else:
        optimizer = None
else:
    optimizer = None

# convert_to_analog will make analog layers trainable, so we need to freeze them
if args.use_analog and args.analog_optimizer in ["AnalogSGD", "AnalogAdam"]:
    # Freeze analog linear layers (except LoRA parameters)
    frozen_count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            # Only freeze the base analog layer, not LoRA adapters
            if hasattr(module, 'weight') and 'lora' not in name.lower():
                module.weight.requires_grad = False
                frozen_count += 1
    print(f"Frozen {frozen_count} analog layer weights (LoRA adapters remain trainable)")
    model.print_trainable_parameters()
else:
    if args.use_analog:
        print("No analog optimizer specified, keeping all analog parameters trainable")
    else:
        print("Using digital model, keeping all parameters trainable")




"""### Data Prep
<a name="Data"></a>

We're using OpenAI's famous GSM8K dataset!
"""

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split = "train")
dataset

"""Let's look at the first row:"""

dataset[0]["question"]

dataset[0]["answer"]

"""We notice all answers like about have a ####, so we extract it:"""

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()
extract_hash_answer(dataset[0]["answer"])

"""We now create a system prompt which can be customized. We add 4 extra symbols for working out or thinking / reasoning sections and a final answer:"""

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# system_prompt = \
# f"""You are given a problem.
# Think about the problem and provide your working out.
# Place it between {reasoning_start} and {reasoning_end}.
# Then, provide your solution between {solution_start}{solution_end}"""


# system_prompt = \
# f"""You are given a problem.
# Think about the problem and provide your working out.
# Place it between {reasoning_start} and {reasoning_end}.
# Then, provide your solution (number only) between {solution_start}{solution_end}, and then end the response."""


# system_prompt = f"""
# Respond in the following format:
# {reasoning_start}
# ...
# {reasoning_end}
# {solution_start}
# ...
# {solution_end}
# """


system_prompt = f"""You will receive a problem to solve.

First, think step by step and show all your reasoning and calculations.
Write your working out between {reasoning_start} and {reasoning_end}.

After finishing your reasoning, give your final answer (number only)
between {solution_start} and {solution_end}.
Do not include any extra text after your answer.

Example format:
{reasoning_start}
<your detailed reasoning, calculations, or derivations here>
{reasoning_end}
{solution_start}
<your final answer here>
{solution_end}
"""




print(system_prompt)

"""Let's map the dataset! and see the first row:"""

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})
dataset[0]

"""We create a regex format to match the reasoning sections and answers:"""

import re

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

"""We verify it works:"""

print("test match_format:")
print(match_format.search(
    "<start_working_out>Let me think!<end_working_out>"\
    "<SOLUTION>2</SOLUTION>",
))

"""We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:"""

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(solution_start)  == 1 else -0.5
        score += 0.5 if response.count(solution_end)    == 1 else -0.5
        scores.append(score)
    return scores

"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                else: score -= 1.0 # Penalize wrong answers
            except:
                score -= 0.5 # Penalize
        scores.append(score)
    return scores

"""Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20."""

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)

print("test match_numbers:")
print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess       = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores


def evaluate_on_gsm8k(model, tokenizer, args, mode='eval_lora'):
    """
    Evaluate the model on GSM8K test set
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        args: Command line arguments
        mode: Evaluation mode ('eval_original', 'eval_lora', 'train_and_eval')
    """
    print(f"Starting evaluation on GSM8K test set (mode: {mode})...")
    print(f"Current active adapter: {model.active_adapter}")
    print(f"Available adapters: {list(model.peft_config.keys())}")
    print(f"Batch size: {args.eval_batch_size}")
    
    # Log noise settings for analog models
    if args.use_analog:
        if args.eval_with_noise:
            print(f"🔊 EVALUATION WITH NOISE ENABLED - Custom noise patching active")
            config = gen_lightning_rpu_config(output_noise_level=args.output_noise_level, weight_noise_level=args.weight_noise_level, eval_with_noise=args.eval_with_noise, discretised_state=args.discretised_state, weight_clip_type=args.weight_clip_type, weight_clip_sigma=args.weight_clip_sigma)
            print(f"   Weight noise std_dev: {config.modifier.std_dev}")
            print(f"   Output noise level: {config.forward.out_noise}")
            print(f"   Noise type: {config.modifier.noise_type.name}")
            print(f"   Noise will be scaled by max weights per channel")
        else:
            print(f"🔇 EVALUATION WITHOUT NOISE - No noise patching applied")
            print(f"   Weight noise std_dev configured: {args.weight_noise_level} (disabled during eval)")
            print(f"   Output noise level: {args.output_noise_level}")
    else:
        print(f"📟 DIGITAL MODEL - No analog noise effects")
    
    # Log PCM model evaluation settings
    if args.evaluate_with_pcm_model:
        print(f"🔧 PCM MODEL EVALUATION ENABLED")
        print(f"   PCM max input size: {args.pcm_max_inp_size}")
        print(f"   PCM noise will be applied to all Linear layers during evaluation")
    else:
        print(f"📟 PCM MODEL EVALUATION DISABLED")
    
    # Set model to evaluation mode
    model.eval()
    
    # Apply PCM noise if requested
    original_weights = None
    if args.evaluate_with_pcm_model:
        print(f"Applying PCM noise to model before evaluation...")
        pcm_noise_config = NoiseConfig(
            noise_type="pcm",
            max_inp_size=args.pcm_max_inp_size
        )
        original_weights = apply_noise(model, pcm_noise_config)
        print(f"PCM noise applied to {len(original_weights)} base layers (LoRA adapters excluded)")

    try:
        # Load test dataset using the same approach as training data
        from datasets import load_dataset
        test_dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        # Apply the same mapping as training data
        test_dataset = test_dataset.map(lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        })
        
        # Limit samples if specified
        if args.eval_samples > 0 and args.eval_samples < len(test_dataset):
            test_dataset = test_dataset.select(range(args.eval_samples))
        
        print(f"Evaluating on {len(test_dataset)} samples")
        
        # Initialize metrics
        total_samples = len(test_dataset)
        correct_answers = 0
        format_correct = 0
        valid_numbers = 0
        processed_samples = 0
        batch_size = args.eval_batch_size
        
        # Process samples in batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = test_dataset[batch_start:batch_end]
            
            try:
                # Prepare batch prompts
                batch_texts = []
                batch_answers = []
                
                # Handle batch_samples which is a dict with lists of values
                batch_prompts = batch_samples['prompt']
                batch_answer_values = batch_samples['answer']
                
                for prompt, answer in zip(batch_prompts, batch_answer_values):
                    text = tokenizer.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    batch_texts.append(text)
                    batch_answers.append(answer)
                
                # Tokenize batch inputs
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(model.device)
                
                # Generate responses for the batch
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.95,
                        # top_k=64,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Extract responses for each sample in the batch
                input_lengths = inputs['input_ids'].shape[1]
                batch_responses = []
                
                for i in range(len(batch_answers)):  # Use batch_answers length to be safe
                    try:
                        # Extract only the newly generated tokens for this sample
                        new_tokens = generated_ids[i][input_lengths:]
                        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        batch_responses.append(response)
                    except Exception as e:
                        print(f"ERROR: Decoding response for sample {batch_start + i + 1}: {e}")
                        batch_responses.append("")  # Add empty response to maintain alignment
                
                # Process each response in the batch
                for i, (response, answer) in enumerate(zip(batch_responses, batch_answers)):
                    sample_idx = batch_start + i
                    try:
                        # Extract answer from response using the same format as training
                        extracted_answer = None
                        match = match_format.search(response)
                        if match:
                            extracted_answer = match.group(1).strip()
                        
                        # Check correctness using same logic as training check_answer function
                        is_correct = False
                        if extracted_answer and answer:
                            # First try exact match
                            if extracted_answer == answer:
                                correct_answers += 1
                                is_correct = True
                            # Then try with stripped whitespace
                            elif extracted_answer.strip() == answer.strip():
                                correct_answers += 1
                                is_correct = True
                            else:
                                # Try numerical comparison as in training
                                try:
                                    ratio = float(extracted_answer) / float(answer)
                                    if ratio >= 0.9 and ratio <= 1.1:
                                        correct_answers += 1
                                        is_correct = True
                                except:
                                    pass
                        
                        # Check format correctness
                        if match_format.search(response) is not None:
                            format_correct += 1
                        
                        # Check if extracted answer is a valid number using numbers regex
                        number_match = match_numbers.search(response)
                        if number_match:
                            valid_numbers += 1
                        
                        # ALWAYS increment processed_samples for every sample that we attempt to process
                        processed_samples += 1

                        # Log first few examples
                        if sample_idx < 3:
                            question = batch_prompts[i][-1]['content']
                            print(f"\nExample {sample_idx + 1}:")
                            print(f"Question: {question}")
                            print(f"Expected: {answer}")
                            print(f"Generated: {response}")
                            print(f"Extracted: {extracted_answer}")
                            print(f"Correct: {is_correct}")
                    
                    except Exception as e:
                        print(f"ERROR: Processing individual sample {sample_idx + 1}: {e}")
                        # Still count it as processed since we attempted to process it
                        processed_samples += 1
                        # Continue processing next sample in batch
                        continue
                
                # Progress update
                print(f"Processed batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}: {processed_samples}/{total_samples} samples...")
            
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Skip this batch but continue with next one
                continue
        
        # Calculate final metrics
        if processed_samples == 0:
            print("No samples were processed successfully!")
            return None
        
        accuracy = correct_answers / processed_samples
        format_accuracy = format_correct / processed_samples
        number_accuracy = valid_numbers / processed_samples
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Total samples requested: {total_samples}")
        print(f"Successfully processed: {processed_samples}")
        if processed_samples < total_samples:
            print(f"Failed to process: {total_samples - processed_samples} samples")
        print(f"Correct answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Format accuracy: {format_accuracy:.4f} ({format_accuracy*100:.2f}%)")
        print(f"Valid number accuracy: {number_accuracy:.4f} ({number_accuracy*100:.2f}%)")
        print(f"{'='*50}")
        
        # Log to wandb if available
        try:
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/format_accuracy": format_accuracy,
                "eval/number_accuracy": number_accuracy,
                "eval/correct_answers": correct_answers,
                "eval/total_samples": total_samples,
                "eval/processed_samples": processed_samples
            })
        except:
            print("Could not log to wandb")
        
        return {
            "accuracy": accuracy,
            "format_accuracy": format_accuracy,
            "number_accuracy": number_accuracy,
            "correct_answers": correct_answers,
            "total_samples": total_samples,
            "processed_samples": processed_samples
        }

    except Exception as e:
        print(f"Error in evaluate_on_gsm8k: {e}")
        return None



# Only perform training if mode requires it
if args.mode in ['train_only', 'train_and_eval']:
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print("="*60)

    """<a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations!
    """

    max_prompt_length = 256

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        learning_rate = 5e-6, # this setting is not used for analog model
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch_fused",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Must be 1 to avoid Unsloth shape mismatch bug when multiple prompts with different lengths are in the same mini-batch
        num_generations = 16, # Effective batch size = num_generations. Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 500 if not args.debug else 1,
        save_steps = 50,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = output_dir,
    )

    """And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

    You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

    | Step | Training Loss | reward    | reward_std | completion_length | kl       |
    |------|---------------|-----------|------------|-------------------|----------|
    | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
    | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
    | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |

    """

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = dataset,
    )

    # Set the analog optimizer after trainer creation
    if optimizer is not None:
        trainer.optimizer = optimizer
        print(f"Set analog optimizer: {type(optimizer).__name__}")
        print(trainer.optimizer)

    trainer.train()
else:
    print(f"\n{'='*60}")
    print("SKIPPING TRAINING (evaluation-only mode)")
    print("="*60)

# Run evaluation based on mode
if args.mode in ['eval_original', 'train_and_eval', 'eval_lora']:
    print(f"\n{'='*60}")
    print("RUNNING EVALUATION ON GSM8K TEST SET")
    print("="*60)
    
    # Determine evaluation mode
    eval_mode = args.mode
    
    # Run evaluation
    eval_results = evaluate_on_gsm8k(model, tokenizer, args, mode=eval_mode)
    
    if eval_results is None:
        print("Evaluation failed!")
else:
    print(f"\n{'='*60}")
    print("SKIPPING EVALUATION (training-only mode)")
    print("="*60)


"""<a name="Inference"></a>
### Inference
Now let's try the model we just trained!
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "What is the sqrt of 101?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    tokenize = False,
)

print("test running one sample")
print("------------prompt------------")
print(text) 
print("------------response------------")

if args.mode == 'train_only' or args.mode == 'train_and_eval':
    model.save_pretrained(lora_save_dir)  # Local saving
    tokenizer.save_pretrained(lora_save_dir)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

print("test running one sample done")

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""



#following codes are not used in this exp
"""### Saving to float16 for VLLM

We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3-finetune`. Set `if False` to `if True` to let it run!
"""

if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3-finetune", tokenizer)

"""If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!"""

if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3-finetune", tokenizer,
        token = "hf_..."
    )

"""### GGUF / llama.cpp Conversion
To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!
"""

if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

"""Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!"""

if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-finetune-gguf",
        token = "hf_...",
    )

"""Now, use the `gemma-3-finetune.gguf` file or `gemma-3-finetune-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>

"""