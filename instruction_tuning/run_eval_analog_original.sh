#!/bin/bash

# Evaluation script for analog original model (no LoRA)
# This evaluates the analog version of unsloth/Meta-Llama-3.1-8B model without LoRA
# Evaluates 12 tasks individually with wandb logging

echo "=========================================="
echo "EVALUATING: Analog Original Model"
echo "=========================================="

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

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
    --model_args "model_name=unsloth/Meta-Llama-3.1-8B,use_analog=True,output_noise_level=$OUTPUT_NOISE_LEVEL,weight_noise_level=$WEIGHT_NOISE_LEVEL,eval_with_noise=$EVAL_WITH_NOISE" \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "./results/analog_original_0shot" \
    --trust_remote_code \
    --show_config



echo "Analog original model evaluation completed!"
echo "Results saved in ./results/analog_original_0shot_*/ directories"
echo "WandB run: $SCRIPT_NAME" 