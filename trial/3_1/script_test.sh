#!/bin/bash -l
#SBATCH --output=./%j.out
#SBATCH --gres=gpu
source /software/spackages_prod/apps/linux-ubuntu20.04-zen2/gcc-9.4.0/anaconda3-2021.05-5d7m6vbj62rh6onwyyz6mdqatpag2b3b/etc/profile.d/conda.sh
module load cuda/11.8.0-gcc-13.2.0
conda activate /scratch/users/k2258665/conda_env/spikecat_cuda11p8
export SQUAD_DIR=pwd/data
python run_qa.py \
--model_name_or_path google/mobilebert-uncased --dataset_name squad \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /scratch/users/k2258665/mobilebert/experiments/squad_models_inference/

