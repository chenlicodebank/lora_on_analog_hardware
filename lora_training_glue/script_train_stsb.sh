export TASK_NAME=stsb
export EXP_INDEX=3
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