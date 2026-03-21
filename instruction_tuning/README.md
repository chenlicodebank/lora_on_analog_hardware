# Instruction tuning

Instruction tuning and evaluations (both qualitative and quantitative).
## Installation


```bash
conda create -n analora-llm python=3.11
conda activate analora-llm
pip install unsloth
pip install wandb
pip install git+https://github.com/IBM/aihwkit-lightning.git
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
# If unsloth pulls CPU-only torch, must force reinstall with CUDA
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# login wandb, for experiment tracking (optional)
huggingface-cli login # for using llama3.1
```


## Usage

### Training Modes

#### 1. Digital Training (Baseline)
```bash
# Train only
WANDB_PROJECT="instruction_tuning" WANDB_NAME="IT_digital" \
    python instruction_tuning.py --mode train_only
```

#### 2. Analog Training (Hardware Simulation)
```bash
# Analog training with per-channel noise
WANDB_PROJECT="instruction_tuning" WANDB_NAME="IT_analog_noise_0.067" \
    python instruction_tuning.py \
    --use_analog \
    --weight_noise_level 0.067 \
    --output_noise_level 0 \
    --mode train_only \
    --analog_optimizer AnalogAdam \
    --analog_lr 0.0002
```

### Qualitative Evaluation

These modes run the model on a sample prompt (e.g., "Continue the Fibonacci sequence") and print the generated output for qualitative inspection.

```bash
# Digital model before LoRA training (base model with random-initialized LoRA)
WANDB_PROJECT="instruction_tuning" WANDB_NAME="IT_digital_eval_original" \
    python instruction_tuning.py --mode eval_original

# Digital model after LoRA training (base model with trained LoRA weights)
WANDB_PROJECT="instruction_tuning" WANDB_NAME="IT_digital_eval_lora" \
    python instruction_tuning.py --mode eval_lora

# Analog model before LoRA training
WANDB_PROJECT="instruction_tuning" WANDB_NAME="IT_analog_eval_original" \
    python instruction_tuning.py \
    --use_analog \
    --weight_noise_level 0.067 \
    --output_noise_level 0 \
    --mode eval_original

# Analog model after LoRA training
WANDB_PROJECT="instruction_tuning" WANDB_NAME="IT_analog_eval_lora" \
    python instruction_tuning.py \
    --use_analog \
    --weight_noise_level 0.067 \
    --output_noise_level 0 \
    --mode eval_lora
```

### Quantitative Evaluation

The following evaluations use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework to run benchmark tasks and report numerical metrics (accuracy).

#### Comprehensive Evaluation (All Tasks, Before and After LoRA)
```bash
# Recommended: runs all 7 tasks, before and after LoRA
python run_eval_full.py
```

#### PCM Noise Model Evaluation

> PCM evaluation uses `custom_analog_model.py` (registered as `custom_analog`) with `evaluate_with_pcm_model=True`.

```bash
# Evaluate analog model with PCM noise (before LoRA)
python lm_eval_wrapper_all_tasks.py --run_name "pcm_eval" --model custom_analog \
    --model_args "model_name=unsloth/Meta-Llama-3.1-8B,use_analog=False,evaluate_with_pcm_model=True,pcm_max_inp_size=512,output_noise_level=0,weight_noise_level=0" \
    --tasks sciq,piqa,arc_easy,arc_challenge,boolq,winogrande,hellaswag \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path ./results/analog_original_pcm_0shot
```


### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--use_analog` | Enable analog hardware simulation | `False` | `True/False` |
| `--mode` | Operation mode | `train_only` | `train_only`, `train_and_eval`, `eval_original`, `eval_lora` |
| `--weight_noise_level` | Weight noise level for analog training | `0.067` | e.g. `0.03`, `0.067` |
| `--output_noise_level` | Noise level for analog outputs | `0.0` | `0.0-1.0` |
| `--analog_optimizer` | Optimizer for analog training | `None` | `AnalogSGD`, `AnalogAdam` |
| `--analog_lr` | Learning rate for analog training | `0.000005` | e.g. `2e-4` |
| `--lora_path` | Path to LoRA weights for eval_lora mode | `saved_lora_models` | any path |

## Evaluation Benchmarks
| Category | Task | Description | Metric |
|----------|------|-------------|--------|
| **Physical Reasoning** | `piqa` | Physical interaction QA | Accuracy |
| **Commonsense Reasoning** | `hellaswag` | Sentence completion | Accuracy |
| **Coreference Resolution** | `winogrande` | Pronoun resolution | Accuracy |
| **Science QA** | `arc_easy` | Elementary science questions | Accuracy |
| **Science QA** | `arc_challenge` | Advanced science questions | Accuracy |
| **Boolean QA** | `boolq` | Yes/no questions | Accuracy |
| **Science QA** | `sciq` | Multiple choice science | Accuracy |

> **Note**: Analog model evaluation can be significantly slower than digital evaluation due to hardware simulation overhead.

## Performance

### Results (single task evaluation on hellaswag)

1. **Digital Baseline**: 78.91%
1. **Analog Baseline**: 29.48%
3. **Analog + lora training**: 67.71%

### Pretrained LoRA Model
You can download the trained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1ysoBDs6ljeN0fcrmQi-Xb80c5ujFeRe9?usp=sharing).


## References & Acknowledgments

This project builds upon several outstanding open-source projects:

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**: Standardized evaluation framework for language models
- **[Unsloth](https://github.com/unslothai/unsloth)**: Ultra-fast and memory-efficient LLM fine-tuning
- **[AIHWKit-Lightning](https://github.com/IBM/aihwkit-lightning)**: IBM's analog hardware simulation toolkit
- **[LoRA](https://arxiv.org/abs/2106.09685)**: Low-Rank Adaptation technique for efficient fine-tuning
- **[Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)**: Instruction-following dataset
