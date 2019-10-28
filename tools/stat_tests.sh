#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

for DIR in "hico_zsk_gc_nobg/standard" "hico_zsk_gc_nobg_sl/asl1" "hico_zsk_gc_nobg_Ra/Ra-10-03" "hico_zsk_gc_nobg_sl_Ra/asl10_Ra-10-03"
do
  python -u tools/stat_test.py "output/skzs/${DIR}" pM-mAP 0.1194
  python -u tools/stat_test.py "output/skzs/${DIR}" zs_pM-mAP 0.075
done

python -u tools/stat_test.py "output/wemb/hico_zsk_nobg/standard" pM-mAP 0.1194
python -u tools/stat_test.py "output/wemb/hico_zsk_nobg/standard" zs_pM-mAP 0.075