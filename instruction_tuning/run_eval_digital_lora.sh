#!/bin/bash

# Evaluation script for digital model with LoRA adapters
# This evaluates the base unsloth/Meta-Llama-3.1-8B model with trained LoRA adapters
# Evaluates 12 tasks individually with wandb logging

echo "=========================================="
echo "EVALUATING: Digital Model + LoRA"
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

# Get script name without .sh extension for wandb run name
SCRIPT_NAME=$(basename "$0" .sh)
echo "WandB run name: $SCRIPT_NAME"

# Run evaluation on all 12 tasks with wandb logging
echo "Starting evaluation of 12 tasks: wikitext, lambada_openai, piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq, sciq, copa, openbookqa"

python lm_eval_wrapper_all_tasks.py \
    --run_name "$SCRIPT_NAME" \
    --model custom_analog \
    --model_args "model_name=unsloth/Meta-Llama-3.1-8B,use_analog=False,lora_path=$LORA_PATH" \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "./results/digital_lora_0shot" \
    --trust_remote_code \
    --show_config


echo "Digital model + LoRA evaluation completed!"
echo "Results saved in ./results/digital_lora_0shot_*/ directories"
echo "WandB run: $SCRIPT_NAME" 