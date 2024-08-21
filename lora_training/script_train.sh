#!/bin/bash -l
#SBATCH --output=./%j.out
#SBATCH --gres=gpu
source /software/spackages_prod/apps/linux-ubuntu20.04-zen2/gcc-9.4.0/anaconda3-2021.05-5d7m6vbj62rh6onwyyz6mdqatpag2b3b/etc/profile.d/conda.sh
module load cuda/11.8.0-gcc-13.2.0
conda activate  /scratch/users/k2258665/aihwkit_neurosoc_drift_fix/conda_env/neurosoc_aihwkit_drift_fix
export SQUAD_DIR=pwd/data
python run_qa.py \
--model_name_or_path google/mobilebert-uncased --dataset_name squad \
--do_train \
--do_eval \
--save_strategy no \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 128 \
--weight_decay 0.0001 \
--num_train_epochs 1 \
--max_seq_length 320 \
--evaluation_strategy epoch \
--doc_stride 128 \
--warmup_steps 0 \
--output_dir ./squad_models_train/ \
--pcm_model NeuroSoCLamina_Gmax55 \
--output_noise_level 0.04 \
--analog_optimizer AnalogAdam \
--analog_lr 0.00005 \
--num_evaluation_drift_values 7 \
--num_evaluation_repetition 1
