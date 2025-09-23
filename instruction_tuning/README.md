# 🔬 Instruction tuning

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
# login wandb, for experiment tracking (optional)
huggingface-cli login # for using llama3.1 
```


## 🎯 Usage

### Training Modes

#### 1. Digital Training (Baseline)
```bash
# Train and evaluate
python instruction_tuning.py --mode train_and_eval

# Evaluate only
python instruction_tuning.py --mode eval_original
```

#### 2. Analog Training (Hardware Simulation)
```bash
# Full analog training pipeline
python instruction_tuning.py \
    --use_analog \
    --mode train_and_eval \
    --output_noise_level 0.1 \
    --analog_optimizer AnalogAdam \
    --analog_lr 0.0002 \
    --pcm_noise_std 0.05

# Analog evaluation only
python instruction_tuning.py \
    --use_analog \
    --mode eval_original
```

### Evaluation Options

#### Comprehensive Evaluation (All Tasks)
```bash
# Digital model evaluation
./run_eval_digital_original.sh      # Base model
./run_eval_digital_lora.sh          # With LoRA adapters

# Analog model evaluation
./run_eval_analog_original.sh       # Base analog model
./run_eval_analog_lora.sh           # Analog + LoRA
./run_eval_analog_original_pcm.sh   # With PCM noise
```

#### Single Task Evaluation (Faster)
```bash
# Recommended for analog models (faster)
./run_eval_analog_original_singletask.sh
```

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--use_analog` | Enable analog hardware simulation | `False` | `True/False` |
| `--mode` | Operation mode | `train_and_eval` | `train_and_eval`, `eval_original` |
| `--output_noise_level` | Noise level for analog outputs | `0.0` | `0.0-1.0` |
| `--analog_optimizer` | Optimizer for analog training | `AnalogSGD` | `AnalogSGD`, `AnalogAdam` |
| `--analog_lr` | Learning rate for analog training | `0.0001` | `2e-4` |

## 📊 Evaluation Benchmarks
| Category | Task | Description | Metric |
|----------|------|-------------|--------|
| **Physical Reasoning** | `piqa` | Physical interaction QA | Accuracy ↑ |
| **Commonsense Reasoning** | `hellaswag` | Sentence completion | Accuracy ↑ |
| **Coreference Resolution** | `winogrande` | Pronoun resolution | Accuracy ↑ |
| **Science QA** | `arc_easy` | Elementary science questions | Accuracy ↑ |
| **Science QA** | `arc_challenge` | Advanced science questions | Accuracy ↑ |
| **Boolean QA** | `boolq` | Yes/no questions | Accuracy ↑ |
| **Science QA** | `sciq` | Multiple choice science | Accuracy ↑ |
| **Causal Reasoning** | `copa` | Cause and effect | Accuracy ↑ |
| **Open Domain QA** | `openbookqa` | Elementary science reasoning | Accuracy ↑ |

> **📝 Note**: Analog model evaluation can be significantly slower than digital evaluation due to hardware simulation overhead.

## 📈 Performance

### Results (single task evaluation on hellaswag)

1. **Digital Baseline**: 78.91%
1. **Analog Baseline**: 29.48%
3. **Analog + lora training**: 67.71%

### Pretrained LoRA Model
You can download the trained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1ysoBDs6ljeN0fcrmQi-Xb80c5ujFeRe9?usp=sharing).


## 📄 References & Acknowledgments

This project builds upon several outstanding open-source projects:

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**: Standardized evaluation framework for language models
- **[Unsloth](https://github.com/unslothai/unsloth)**: Ultra-fast and memory-efficient LLM fine-tuning
- **[AIHWKit-Lightning](https://github.com/IBM/aihwkit-lightning)**: IBM's analog hardware simulation toolkit
- **[LoRA](https://arxiv.org/abs/2106.09685)**: Low-Rank Adaptation technique for efficient fine-tuning
- **[Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)**: Instruction-following dataset


