# GRPO Training with Analog Hardware

Implementations for training language models using GRPO (Group Relative Policy Optimization) with support for analog hardware simulation.

## Installation


```bash
conda create -n analora-llm python=3.11
conda activate analora-llm
pip install unsloth
pip install wandb
pip install git+https://github.com/IBM/aihwkit-lightning.git

# login wandb, for experiment tracking (optional)
huggingface-cli login # for using llama3.1 
```



## Features

### Training Capabilities

- ✅ **GRPO Training** - Group Relative Policy Optimization for reasoning tasks
- ✅ **Analog Hardware Support** - Train directly on analog hardware simulation
- ✅ **GSM8K Integration** - Specialized for mathematical reasoning tasks
- ✅ **LoRA Adapters** - Efficient fine-tuning with Low-Rank Adaptation
- ✅ **Multiple Models** - Support for Llama, Gemma, and other models
- ✅ **Noise Injection** - Various noise models during training and inference

### Hardware Simulation

- **Digital Mode**: Standard PyTorch training
- **Analog Mode**: aihwkit-lightning integration
- **PCM Models**: Phase Change Memory simulation
- **Noise Models**: Configurable noise characteristics

## Usage

```bash
# digital train
python grpo_gsm8k.py \
    --mode train_only \
    --eval_samples 1 \

# analog train
python grpo_gsm8k.py \
--use_analog \
--mode train_only \
--eval_samples 1 \
--weight_noise_level 0.03 \
--analog_optimizer AnalogAdam \
--analog_lr 0.000005

#evaluate digital model
python grpo_gsm8k.py \
    --mode eval_original \
    --eval_samples -1 \
    --eval_batch_size 50

#evaluate digital model with lora
python grpo_gsm8k.py \
    --mode eval_lora \
    --lora_path saved_lora_models \
    --eval_samples -1 \
    --eval_batch_size 50

# evaluate analog model with Gaussian noise
python grpo_gsm8k.py \
    --mode eval_original \
    --use_analog \
    --eval_with_noise \
    --weight_noise_level 0.03 \
    --eval_samples -1 \
    --eval_batch_size 50


# evaluate analog model with Gaussian noise and lora
python grpo_gsm8k.py \
    --mode eval_lora \
    --use_analog \
    --eval_with_noise \
    --weight_noise_level 0.03 \
    --lora_path saved_lora_models \
    --eval_samples -1 \
    --eval_batch_size 50

# evaluate analog model with PCM noise
python grpo_gsm8k.py \
    --mode eval_original \
    --use_analog \
    --evaluate_with_pcm_model \
    --pcm_max_inp_size 512 \
    --eval_samples -1 \
    --eval_batch_size 50

# evaluate analog model with PCM noise and lora
python grpo_gsm8k.py \
    --mode eval_lora \
    --use_analog \
    --evaluate_with_pcm_model \
    --pcm_max_inp_size 512 \
    --lora_path saved_lora_models \
    --eval_samples -1 \
    --eval_batch_size 50
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "train_only" | Operation mode (train_only, eval_lora, train_and_eval) |
| `use_analog` | bool | False | Enable analog hardware simulation |
| `pcm_model` | str | "PCM_Gmax25" | PCM model type |
| `output_noise_level` | float | 0.0 | Output noise level |
| `weight_noise_level` | float | 0.03 | Weight noise level |
| `max_steps` | int | 500 | Maximum training steps |
| `eval_samples` | int | -1 | Number of evaluation samples (-1 for all) |
| `lora_path` | str | None | Path to LoRA adapters |


## Algorithm and Dataset Overview

### GRPO

1. **Reward Function**: Based on correctness of mathematical reasoning
2. **Relative Ranking**: Responses are ranked within groups


### GSM8K

- **Training**: 7,473 problems for training
- **Testing**: 1,319 problems for evaluation


## 📈 Performance

### Results

1. **Digital Baseline**: 67.63%
2. **Digital + GRPO**: 85.06%
1. **Analog Baseline**: 37.98%
3. **Analog + GRPO**: 70.74%

### Pretrained LoRA Model
You can download the trained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1ysoBDs6ljeN0fcrmQi-Xb80c5ujFeRe9?usp=sharing).


## 📄 References

This implementation is based on:
- GRPO: Group Relative Policy Optimization
- Unsloth: Efficient model fine-tuning
- aihwkit-lightning: Analog hardware simulation
- GSM8K: Grade School Math dataset

