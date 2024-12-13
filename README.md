
# LoRA on analog hardware

This repository provides code for reproducing the [Hardware-Aware LoRA Training](https://arxiv.org/pdf/2411.17367) results.

---

## Table of Contents
- [Installation](#installation)
- [Hardware-Aware Training on SQuAD](#hardware-aware-training-on-squad)
- [Hardware-Aware LoRA Training on SQuAD](#hardware-aware-lora-training-on-squad)
- [Hardware-Aware LoRA Training on GLUE](#hardware-aware-lora-training-on-glue)
- [Scaling](#scaling)
- [Better LoRA and AIHWKIT Settings](#better-lora-and-aihwkit-settings)
- [Citation](#citation)

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

## Hardware-Aware Training on SQuAD

This section provides a straightforward application of Hardware-Aware Training with MobileBERT on the SQuAD v1.1 dataset. 

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

---

## Hardware-Aware LoRA Training on SQuAD

This section provides an application of the Hardware-Aware LoRA Training with MobileBERT on the SQuAD v1.1 dataset. In traditional Hardware-Aware Training, the process typically involves two steps: fine-tuning the model in full precision, and then using the fine-tuned model for hardware-aware training. However, in Hardware-Aware LoRA Training, we skip the full precision fine-tuning step and directly use the pretrained model. Specifically, we use `--model_name_or_path google/mobilebert-uncased` instead of `--model_name_or_path csarron/mobilebert-uncased-squad-v1`. This approach retains the flexibility for post-deployment adaptations to new tasks (e.g., GLUE) and hardware configurations (e.g., ADC bit settings), see [paper](https://arxiv.org/pdf/2411.17367) for details.

### Example:
```
cd lora_training
```
```
export SQUAD_DIR=pwd/data
export SQUAD_DIR=pwd/data
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

## Scaling

The evaluated model is MobileBERT as its parameters (25.3M) can fit on modern analog chips. The proposed method can be applied to other models by specifying `--model_name_or_path`. The results on BERT\_BASE (110M) and BERT\_LARGE (340M) can be found on [paper](https://arxiv.org/pdf/2411.17367).

---
## Better LoRA and AIHWKIT settings

We employ naive [LoRA](https://arxiv.org/pdf/2106.09685) to keep the implementation simple and establish baseline results. Leveraging more advanced LoRA variants and better LoRA hyperparameters have the potential to achieve superior performance compared to the results presented in our paper.

Additionally, the final performance is influenced by the settings in AIHWKIT. Most tunable parameters are configurable through the `gen_rpu_config` function.

The training hyperparameters such as the learning rate and the number of epochs, are selected using a heuristic approach. Further fine-tuning of these hyperparameters may enhance performance.


---

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

