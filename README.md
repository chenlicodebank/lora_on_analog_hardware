
# LoRA on analog hardware

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F2634--4386%2Fae405e-blue)](https://doi.org/10.1088/2634-4386/ae405e)
[![arXiv](https://img.shields.io/badge/arXiv-2411.17367-b31b1b.svg)](https://arxiv.org/abs/2411.17367)

Official code for [**Efficient transformer adaptation for analog in-memory computing via low-rank adapters**](https://doi.org/10.1088/2634-4386/ae405e), published in *Neuromorphic Computing and Engineering* **6**, 014011 (2026, open access). [[IOP]](https://iopscience.iop.org/article/10.1088/2634-4386/ae405e) [[arXiv]](https://arxiv.org/abs/2411.17367)

**AHWA-LoRA training** adapts transformers to analog in-memory computing (AIMC) / processing-in-memory (PIM) hardware without retraining or reprogramming the base model: the pre-trained weights stay frozen on the analog tiles as weight-stationary meta-weights, and lightweight external LoRA adapters — trained hardware-aware — absorb both device non-idealities and downstream-task shifts.

## Results at a glance

| Setting | Model | Headline result |
|---|---|---|
| SQuAD v1.1, PCM hardware model | MobileBERT | F1 **89.06** at t=0 / **85.36** after 10-year drift — on par with full analog hardware-aware retraining (89.47 / 85.14) while training **1.63M vs 24.67M** parameters (15× fewer) |
| GLUE (one frozen base + per-task LoRA) | MobileBERT | Average **78.8** at t=0, **76.9** after 10-year drift |
| Scaling | BERT-Base / BERT-Large | 10-year-drift F1 drop of only **0.63 / 0.48** points (LoRA = 1.3M / 3.5M weights) |
| Instruction tuning (zero-shot, 9 benchmarks) | LLaMA 3.1 8B | Recovers analog accuracy, e.g. HellaSwag **29.5 → 67.7** (digital baseline 78.9) |
| Reinforcement learning (GRPO, GSM8K CoT) | LLaMA 3.1 8B | Analog accuracy **37.98 → 70.74**, cutting the analog–digital gap from ~30 to ~15 points (~2.5 under reduced evaluation noise) |
| Hybrid deployment (AIMC tiles + RISC-V PMCA pipeline) | MobileBERT | As low as **4%** per-layer latency overhead vs a fully-AIMC baseline |

---

## Table of Contents
- [Installation](#installation)
- [Hardware-Aware Training on SQuAD](#hardware-aware-training-on-squad)
- [Hardware-Aware LoRA Training on SQuAD](#hardware-aware-lora-training-on-squad)
- [Hardware-Aware LoRA Training on GLUE](#hardware-aware-lora-training-on-glue)
- [Instruction Tuning with Analog Hardware](#instruction-tuning-with-analog-hardware)
- [GRPO Training with Analog Hardware](#grpo-training-with-analog-hardware)
- [Scaling](#scaling)
- [Better LoRA and AIHWKIT Settings](#better-lora-and-aihwkit-settings)
- [Citation](#citation)

---


## Installation

1. **Clone the Repository:**
   ```
   conda create -n analog_lora python=3.9
   conda activate analog_lora
   git clone https://github.com/chenlicodebank/lora_on_analog_hardware.git
   cd lora_on_analog_hardware
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Install AIHWKIT:**
   
   If aihwkit doesn't work, use [advanced installation](https://aihwkit.readthedocs.io/en/latest/advanced_install.html).

---

## Hardware-Aware Training on SQuAD

This section provides a straightforward application of Hardware-Aware Training with MobileBERT on the SQuAD v1.1 dataset. 

### Example:
```
cd hardware_aware_training
```
```
export SQUAD_DIR=$(pwd)/data
python run_qa.py \
  --model_name_or_path csarron/mobilebert-uncased-squad-v1 --dataset_name squad \
  --do_train \
  --report_to wandb \
  --logging_steps 100 \
  --do_eval \
  --save_strategy epoch \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --weight_decay 0.0001 \
  --num_train_epochs 15 \
  --max_seq_length 320 \
  --evaluation_strategy epoch \
  --doc_stride 128 \
  --warmup_steps 0 \
  --output_dir ./squad_models_train/ \
  --pcm_model PCM_Gmax25 \
  --output_noise_level 0.04 \
  --analog_optimizer AnalogAdam \
  --analog_lr 0.00005 \
  --num_evaluation_drift_values 7 \
  --num_evaluation_repetition 10
```

---

## Hardware-Aware LoRA Training on SQuAD

This section provides an application of the Hardware-Aware LoRA Training with MobileBERT on the SQuAD v1.1 dataset. In traditional Hardware-Aware Training, the process typically involves two steps: fine-tuning the model in full precision, and then using the fine-tuned model for hardware-aware training. However, in Hardware-Aware LoRA Training, we skip the full precision fine-tuning step and directly use the pretrained model. Specifically, we use `--model_name_or_path google/mobilebert-uncased` instead of `--model_name_or_path csarron/mobilebert-uncased-squad-v1`. This approach retains the flexibility for post-deployment adaptations to new tasks (e.g., GLUE) and hardware configurations (e.g., ADC bit settings), see [paper](https://arxiv.org/abs/2411.17367) for details.

### Example:
```
cd lora_training
```
```
export SQUAD_DIR=$(pwd)/data
python run_qa.py \
   --model_name_or_path google/mobilebert-uncased --dataset_name squad \
   --do_train \
   --report_to wandb \
   --logging_steps 100 \
   --do_eval \
   --save_strategy epoch \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --weight_decay 0.0001 \
   --num_train_epochs 15 \
   --max_seq_length 320 \
   --evaluation_strategy epoch \
   --doc_stride 128 \
   --warmup_steps 0 \
   --output_dir ./squad_models_train/ \
   --pcm_model PCM_Gmax25 \
   --output_noise_level 0.04 \
   --analog_optimizer AnalogAdam \
   --analog_lr 0.0002 \
   --num_evaluation_drift_values 7 \
   --num_evaluation_repetition 10
```

---

## Hardware-Aware LoRA Training on GLUE

This section provides an application of the Hardware-Aware LoRA Training with MobileBERT on GLUE. The example is on CoLA, check shell scripts in [lora_training_glue](https://github.com/chenlicodebank/lora_on_analog_hardware/tree/main/lora_training_glue) for other GLUE subtasks.

### Example:
```
cd lora_training_glue
```
```
export TASK_NAME=cola
export EXP_INDEX=1
python run_glue.py \
  --model_name_or_path google/mobilebert-uncased \
  --task_name $TASK_NAME \
  --ignore_mismatched_sizes \
  --report_to wandb \
  --logging_steps 100 \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 15 \
  --output_dir ./results/$TASK_NAME/$EXP_INDEX \
  --pcm_model PCM_Gmax25 \
  --output_noise_level 0.04 \
  --analog_optimizer AnalogAdam \
  --analog_lr 0.0002 \
  --num_evaluation_drift_values 7 \
  --num_evaluation_repetition 10
```

---

## Instruction Tuning with Analog Hardware

This directory contains tools for instruction tuning and evaluation of language models with analog hardware simulation, using the lm-evaluation-harness framework. It supports both digital and analog models with LoRA adapters.

### Key Features:
- **Custom Analog Model Integration**: Seamless integration with lm-evaluation-harness
- **Multiple Evaluation Tasks**: 7 benchmark tasks including BoolQ, PIQA, HellaSwag, etc.
- **WandB Integration**: Automatic experiment tracking and logging
- **Noise Model Support**: Configurable analog hardware noise simulation
- **PCM Models**: Phase Change Memory noise characteristics

### Quick Start:
```bash
cd instruction_tuning

# Digital training
python instruction_tuning.py --mode train_only

# Analog training with noise
python instruction_tuning.py --use_analog --weight_noise_level 0.067 --mode train_only --analog_optimizer AnalogAdam --analog_lr 0.0002

# Comprehensive evaluation (all tasks, before and after LoRA)
python run_eval_full.py
```

See [instruction_tuning/README.md](instruction_tuning/README.md) for detailed documentation.

---

## GRPO Training with Analog Hardware

This directory implements GRPO (Group Relative Policy Optimization) training for language models with analog hardware support, specialized for mathematical reasoning tasks.

### Key Features:
- **GRPO Algorithm**: Advanced training for reasoning models
- **GSM8K Dataset**: Mathematical reasoning benchmark integration
- **Analog Training**: Direct training on analog hardware simulation
- **Noise Models**: Configurable Gaussian and PCM noise characteristics
- **Multiple Models**: Support for Llama, Gemma, and other architectures

### Quick Start:
```bash
cd reinforcement_learning

# Train digital model with GRPO
python grpo_gsm8k.py --mode train_only --eval_samples 1

# Train analog model
python grpo_gsm8k.py --use_analog --mode train_only --eval_samples 1 --weight_noise_level 0.03 --analog_optimizer AnalogAdam --analog_lr 0.000005

# Evaluate digital model with LoRA
python grpo_gsm8k.py --mode eval_lora --lora_path saved_lora_models/1_1_baseline --eval_samples -1 --eval_batch_size 50
```

See [reinforcement_learning/README.md](reinforcement_learning/README.md) for detailed documentation.

---

## Scaling

The evaluated model is MobileBERT as its parameters (25.3M) can fit on modern analog chips. The proposed method can be applied to other models, simply by specifying `--model_name_or_path` to the desired model in the above scripts. The results on BERT\_BASE (110M) and BERT\_LARGE (340M) can be found in the [paper](https://arxiv.org/abs/2411.17367).

---
## Better LoRA and AIHWKIT Settings

We employ naive [LoRA](https://arxiv.org/abs/2106.09685) to keep the implementation simple and establish baseline results. Leveraging more advanced LoRA variants and better LoRA hyperparameters have the potential to achieve superior performance compared to the results presented in our paper.

Additionally, the final performance is influenced by the settings in AIHWKIT. Most tunable parameters are configurable through the `gen_rpu_config` function.

The training hyperparameters such as the learning rate and the number of epochs, are selected using a heuristic approach. Further fine-tuning of these hyperparameters may enhance performance.


---

## Citation

If you use this repository in your research or project, please consider citing it using the following format:

```
@article{li2026efficient,
  title={Efficient transformer adaptation for analog in-memory computing via low-rank adapters},
  author={Li, Chen and Ferro, Elena and Lammie, Corey and Le Gallo, Manuel and Boybat, Irem and Rajendran, Bipin},
  journal={Neuromorphic Computing and Engineering},
  volume={6},
  number={1},
  pages={014011},
  year={2026},
  publisher={IOP Publishing},
  doi={10.1088/2634-4386/ae405e}
}
```

