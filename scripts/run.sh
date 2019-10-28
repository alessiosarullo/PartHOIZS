#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_LAUNCH_BLOCKING=1  # uncomment this to debug CUDA errors

NET=$1
EXP_NAME=$2
VARIANT_NAME=$3
NUM_RUNS=$4
GPU_ID=$5
# The following parameters are optional: a default value is provided and it is only substituted if the relative argument is unset or has a null value
# (e.g., the empty string ''). Remove the colon to only substitute if unset.

export CUDA_VISIBLE_DEVICES=$GPU_ID
OUTPUT_DIR="output/${NET}"
EXP_FULL_NAME="${EXP_NAME}/${VARIANT_NAME}"

EXPS=()
for IDX in $(seq 1 "${NUM_RUNS}")
do
  DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
  if [ "${NUM_RUNS}" -gt 1 ]; then
    RUN_NAME="${EXP_FULL_NAME}/${DATETIME}_RUN${IDX}"
  else
    RUN_NAME="${EXP_FULL_NAME}/${DATETIME}_SINGLE"
  fi
  EXP_DIR="${OUTPUT_DIR}/${RUN_NAME}"
  LOG="$EXP_DIR/log.txt"

  EXPS+=("${EXP_DIR}")
  mkdir -p "${EXP_DIR}"
  exec &> >(tee -a "$LOG")
  echo Logging "${EXP_DIR}" to "$LOG"

  if [ "${NUM_RUNS}" -gt 1 ]; then
    python -u scripts/run.py --model "${NET}" --save_dir "${RUN_NAME}"  --randomize "${@:6}"
  else
    python -u scripts/run.py --model "${NET}" --save_dir "${RUN_NAME}" "${@:6}"
  fi
done

set +x

if [ "${NUM_RUNS}" -gt 1 ]; then
  DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
  EXP_DIR="${OUTPUT_DIR}/${EXP_FULL_NAME}/${DATETIME}_AGGR${NUM_RUNS}"
  mkdir -p "${EXP_DIR}"
  python -u scripts/aggregate_tb_runs.py "${EXP_DIR}" "${EXPS[@]}"
fi
