#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_LAUNCH_BLOCKING=1  # uncomment this to debug CUDA errors

NET=$1
EXP_FULL_NAME=$2
GPU_ID=$3
# The following parameters are optional: a default value is provided and it is only substituted if the relative argument is unset or has a null value
# (e.g., the empty string ''). Remove the colon to only substitute if unset.

export CUDA_VISIBLE_DEVICES=$GPU_ID

# log
OUTPUT_DIR="output/${NET}"
EXP_DIR=${OUTPUT_DIR}/${EXP_FULL_NAME}
LOG="$EXP_DIR/log.txt"

exec &> >(tee -a "$LOG")
echo Logging ${EXP_DIR} to "$LOG"

python -u scripts/launch.py --model ${NET} --save_dir ${EXP_FULL_NAME} --resume "${@:4}"