#!/bin/bash

# Evaluation script for analog original model with PCM noise model
# This evaluates the analog version of unsloth/Meta-Llama-3.1-8B model without LoRA
# Uses PCM noise model instead of Gaussian noise during evaluation

echo "=========================================="
echo "EVALUATING: Analog Original Model (PCM)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration for PCM noise model
EVALUATE_WITH_PCM_MODEL=True
PCM_MAX_INP_SIZE=512  # Controls whether per column or per tile application
OUTPUT_NOISE_LEVEL=0  # Keep output noise at 0 for PCM evaluation
WEIGHT_NOISE_LEVEL=0  # Disable Gaussian weight noise when using PCM

echo "PCM noise model configuration:"
echo "  Evaluate with PCM model: $EVALUATE_WITH_PCM_MODEL"
echo "  PCM max input size: $PCM_MAX_INP_SIZE"
echo "  Output noise level: $OUTPUT_NOISE_LEVEL (disabled for PCM)"
echo "  Weight noise level: $WEIGHT_NOISE_LEVEL (disabled for PCM)"

# Get script name without .sh extension for wandb run name
SCRIPT_NAME=$(basename "$0" .sh)
echo "WandB run name: $SCRIPT_NAME"

# Run evaluation on all 12 tasks with PCM noise model
echo "Starting evaluation of 12 tasks with PCM noise model: wikitext, lambada_openai, piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq, sciq, copa, openbookqa"

python lm_eval_wrapper_all_tasks.py --model custom_analog \
    --model_args model_name=unsloth/Meta-Llama-3.1-8B,use_analog=False,evaluate_with_pcm_model=$EVALUATE_WITH_PCM_MODEL,pcm_max_inp_size=$PCM_MAX_INP_SIZE,output_noise_level=$OUTPUT_NOISE_LEVEL,weight_noise_level=$WEIGHT_NOISE_LEVEL \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path ./results/analog_original_pcm_0shot \
    --trust_remote_code \
    --show_config

echo "=========================================="
echo "PCM EVALUATION COMPLETED"
echo "Results saved to: ./results/analog_original_pcm_0shot"
echo "=========================================="
