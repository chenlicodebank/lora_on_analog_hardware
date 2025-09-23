#!/bin/bash

# Evaluation script for original digital model (no LoRA, no analog)
# This evaluates the base unsloth/Meta-Llama-3.1-8B model
# Evaluates 12 tasks individually with wandb logging

echo "=========================================="
echo "EVALUATING: Digital Original Model"
echo "=========================================="

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Get script name without .sh extension for wandb run name
SCRIPT_NAME=$(basename "$0" .sh)
echo "WandB run name: $SCRIPT_NAME"

# Run evaluation on all 12 tasks with wandb logging
echo "Starting evaluation of 12 tasks: wikitext, lambada_openai, piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq, sciq, copa, openbookqa"

    python lm_eval_wrapper_all_tasks.py \
    --run_name "$SCRIPT_NAME" \
    --model custom_analog \
    --model_args "model_name=unsloth/Meta-Llama-3.1-8B,use_analog=False" \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "./results/digital_original_0shot" \
    --trust_remote_code \
    --show_config

echo "Digital original model evaluation completed!"
echo "Results saved in ./results/digital_original_0shot_*/ directories"
echo "WandB run: $SCRIPT_NAME" 