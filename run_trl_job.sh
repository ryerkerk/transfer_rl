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
  aws s3 cp "s3://transfer-rl/trained_models/${INITIAL_MODEL}.pt" - > "./trained_models/${INITIAL_MODEL}.pt" || error_exit "Failed to download initial model from s3 bucket."
fi

# Environmental variables to check for
declare -a ENV_LIST=(save_name "env" "leg_length" "total_frames" \
                     "initial_model" "max_time_steps" "batch_size" \
                     "learning_rate" "train_steps" "hidden_layers" "optimizer" \
                     "model" "action_std" "gamma")

# Check if each environmental variables exist.
# If they do, add to argument list for eventual python call
arg_list=""
for E in "${ENV_LIST[@]}"
do
  if [[ !   ${!E} == "" ]]; then
    arg_list="${arg_list} --${E}=${!E}"    
  fi
done

if [[ -z ${!E+x} ]]; then
  echo ${!E}
fi

python3 -u run.py ${arg_list} | tee "${SAVE_NAME}.txt"

aws s3 cp "./results/${SAVE_NAME}.p" "s3://transfer-rl/results/${SAVE_NAME}.p" || error_exit "Failed to upload results to s3 bucket."
aws s3 cp "./trained_models/${SAVE_NAME}.pt" "s3://transfer-rl/trained_models/${SAVE_NAME}.pt" || error_exit "Failed to upload results to s3 bucket."
aws s3 cp "./${SAVE_NAME}.txt" "s3://transfer-rl/logs/${SAVE_NAME}.txt" || error_exit "Failed to upload logs to s3 bucket."

date
echo "Finished"
\
