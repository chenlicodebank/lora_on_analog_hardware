#!/bin/bash

# Evaluation script for analog original model (no LoRA)
# This evaluates the analog version of unsloth/Meta-Llama-3.1-8B model without LoRA
# Evaluates 1 selected task

echo "=========================================="
echo "EVALUATING: Analog Original Model"
echo "=========================================="

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration for analog hardware
OUTPUT_NOISE_LEVEL=0
WEIGHT_NOISE_LEVEL=0.01
EVAL_WITH_NOISE=True  # Set to True to enable noise during evaluation

echo "Analog configuration:"
echo "  Output noise level: $OUTPUT_NOISE_LEVEL"
echo "  Weight noise level: $WEIGHT_NOISE_LEVEL"
echo "  Eval with noise: $EVAL_WITH_NOISE"

# Get script name without .sh extension for wandb run name
SCRIPT_NAME=$(basename "$0" .sh)
echo "WandB run name: $SCRIPT_NAME"

# Run evaluation on all 1 selected task
echo "Starting evaluation of 1 selected task: hellaswag"

python lm_eval_wrapper_single_task.py --model custom_analog \
    --model_args model_name=unsloth/Meta-Llama-3.1-8B,use_analog=True,output_noise_level=$OUTPUT_NOISE_LEVEL,weight_noise_level=0.067,eval_with_noise=$EVAL_WITH_NOISE \
    --tasks hellaswag \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path ./results/analog_original_0shot \
    --trust_remote_code \
    --show_config

