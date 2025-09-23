#!/bin/bash

# Evaluation script for analog model with LoRA adapters
# This evaluates the analog version of unsloth/Meta-Llama-3.1-8B model with trained LoRA adapters
# Evaluates 12 tasks individually with wandb logging

echo "=========================================="
echo "EVALUATING: Analog Model + LoRA"
echo "=========================================="

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if LoRA path exists
LORA_PATH="lora_model"
if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA path '$LORA_PATH' not found!"
    echo "Please make sure you have trained the model using instruction_tuning.py first."
    echo "The training script should create a 'lora_model' directory with the saved adapters."
    exit 1
fi

echo "Using LoRA adapters from: $LORA_PATH"

# Configuration for analog hardware
OUTPUT_NOISE_LEVEL=0
WEIGHT_NOISE_LEVEL=0.067
EVAL_WITH_NOISE=True  # Set to True to enable noise during evaluation

echo "Analog configuration:"
echo "  Output noise level: $OUTPUT_NOISE_LEVEL"
echo "  Weight noise level: $WEIGHT_NOISE_LEVEL"
echo "  Eval with noise: $EVAL_WITH_NOISE"

# Get script name without .sh extension for wandb run name
SCRIPT_NAME=$(basename "$0" .sh)
echo "WandB run name: $SCRIPT_NAME"

# Run evaluation on all 12 tasks with wandb logging
echo "Starting evaluation of 12 tasks: wikitext, lambada_openai, piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq, sciq, copa, openbookqa"

python lm_eval_wrapper_all_tasks.py \
    --run_name "$SCRIPT_NAME" \
    --model custom_analog \
    --model_args "model_name=unsloth/Meta-Llama-3.1-8B,use_analog=True,lora_path=$LORA_PATH,output_noise_level=$OUTPUT_NOISE_LEVEL,eval_with_noise=$EVAL_WITH_NOISE" \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "./results/analog_lora_0shot" \
    --trust_remote_code \
    --show_config

echo "Analog model + LoRA evaluation completed!"
echo "Results saved in ./results/analog_lora_0shot_*/ directories"
echo "WandB run: $SCRIPT_NAME" 