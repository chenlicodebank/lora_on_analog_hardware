
# LoRA on analog hardware

This repository provides code for reproducing the [Hardware-Aware LoRA Training](https://arxiv.org/pdf/2411.17367) results.

---

## Table of Contents
- [Installation](#installation)
- [Hardware-Aware Training on SQuAD](#hardware-aware-training)
- [Hardware-Aware LoRA Training on SQuAD](#hardware-aware-lora-training)
- [Hardware-Aware LoRA Training on GLUE](#training-on-glue)
- [Contributing](#contributing)
- [License](#license)

---


## Installation

1. **Clone the Repository:**
   ```
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

## Hardware-Aware Training

This section provides a straightforward application of Hardware-Aware Training using MobileBERT on SQuAD1.1. 

### Example:
```
cd hardware_aware_training
```
```
export SQUAD_DIR=pwd/data
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


**Supported Tasks:**
- `MRPC`, `SST-2`, `MNLI`, `QNLI`, and more.

---

## Citation

## Citation

If you use this repository in your research or project, please consider citing it using the following format:

```
@article{li2024efficient,
  title={Efficient Deployment of Transformer Models in Analog In-Memory Computing Hardware},
  author={Li, Chen and Lammie, Corey and Gallo, Manuel Le and Rajendran, Bipin},
  journal={arXiv preprint arXiv:2411.17367},
  year={2024}
}
```

