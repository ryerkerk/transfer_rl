#!/bin/bash

date
echo "Starting"

error_exit () {
  echo "${BASENAME} - ${1}" >&2
  exit 1
}

if [ ! -d "results" ]; then
  mkdir results
fi

if [ ! -d "trained_models" ]; then
  mkdir trained_models
fi

if [ ! ${INITIAL_MODEL} == "none" ]; then
  aws s3 cp "s3://transfer-rl/trained_models/${MODEL_NAME}.pt" - > "./trained_models/${MODEL_NAME}.pt" || error_exit "Failed to download initial model from s3 bucket."
fi

python3 -u run.py \
       --save_name=$SAVE_NAME \
       --env=$ENV \
       --leg_length=$LEG_LENGTH \
       --total_frames=$TOTAL_FRAMES \
       --initial_model=$INITIAL_MODEL \
       --max_time_steps=$MAX_TIME_STEPS \
       --batch_size=$BATCH_SIZE \
       --learning_rate=$LEARNING_RATE \
       --train_steps=$TRAIN_STEPS \
       --hidden_layers=$HIDDEN_LAYERS \
       --optimizer=$OPTIMIZER \
       --model=$MODEL \
       --action_std=$ACTION_STD \
       --gamma=$GAMMA | tee "${SAVE_NAME}.txt"

aws s3 cp "./results/${SAVE_NAME}.p" "s3://transfer-rl/results/${SAVE_NAME}.p" || error_exit "Failed to upload results to s3 bucket."
aws s3 cp "./trained_models/${SAVE_NAME}.pt" "s3://transfer-rl/trained_models/${SAVE_NAME}.pt" || error_exit "Failed to upload results to s3 bucket."
aws s3 cp "./${SAVE_NAME}.txt" "s3://transfer-rl/logs/${SAVE_NAME}.txt" || error_exit "Failed to upload logs to s3 bucket."

date
echo "Finished"
\
