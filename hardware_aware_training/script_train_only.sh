export SQUAD_DIR=pwd/data
python run_qa.py \
--model_name_or_path csarron/mobilebert-uncased-squad-v1 --dataset_name squad \
--do_train \
--report_to wandb \
--logging_steps 100 \
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

