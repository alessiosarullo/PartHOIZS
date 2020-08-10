#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

DIR="hc/gcn/zs1_nopart"

for EXP in "sgd" "sgd_awsu01"
do
  python -u tools/stat_test.py "output/${DIR}/${EXP}" zs_M-mAP 0.069
  python -u tools/stat_test.py "output/${DIR}/${EXP}" zs_unseen_acts_M-mAP 0.073
done
