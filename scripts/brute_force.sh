#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$1
EXP_NAME=$2
GPU_ID=$3
START=$4
END=$5

export CUDA_VISIBLE_DEVICES=$GPU_ID
OUTPUT_DIR="output/${NET}"

for IDX in $(seq "${START}" "${END}")
do
  python -u tools/create_hico_zs_split.py "${IDX}"

  DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')
  EXP_FULL_NAME="${DATETIME}_zs${IDX}_${EXP_NAME}"
  EXP_DIR=${OUTPUT_DIR}/${EXP_FULL_NAME}
  LOG="$EXP_DIR/log.txt"

  mkdir -p "${EXP_DIR}"
  echo Logging "${EXP_DIR}" to "$LOG"

  python -u scripts/run.py --model "${NET}" --save_dir "${EXP_FULL_NAME}" --seenf "${IDX}" "${@:6}" > "${LOG}" 2>&1
done
