#!/bin/bash
# 목발 분류 2차 (3090×2). NAS 1차와 같은 코드·같은 데이터·같은 fold 규칙, 다른 점은
#   (1) DL 순열 100회 (NAS는 cnn 20회만)   (2) seed 3개로 안정성   (3) GPU 2장에 event/cycle 분담
# 준비: result/ml/{epochs,cycles}.npz, result/ml/events.csv, result/stats/{grf_eag_pooled,cov_per_subject}.csv
#       (result/는 git 무시 대상이라 rsync로 가져와야 한다. 원본 29 GB data/는 필요 없음)
# 사용: ./run_crutch_gpu.sh            # 두 GPU에 병렬, 로그 result/ml/crutch_gpu_{event,cycle}.log
cd "$(dirname "$0")"
mkdir -p result/ml
run() { local gpu=$1; shift; local log=$1; shift
        echo "### $(date '+%F %T') $*" >> "$log"
        CUDA_VISIBLE_DEVICES=$gpu python3 ml_crutch.py --device cuda --threads 4 "$@" >> "$log" 2>&1; }

(
  L=result/ml/crutch_gpu_event.log
  for s in 20260919 1 2; do
    run 0 $L --window event --model cnn,cnn_lstm,cnn_tf,hybrid --seed $s --tag s$s
  done
  run 0 $L --window event --model cnn,hybrid --n-perm 100 --tag perm100
  echo "### $(date '+%F %T') EVENT DONE" >> $L
) &
(
  L=result/ml/crutch_gpu_cycle.log
  for s in 20260919 1 2; do
    run 1 $L --window cycle --model cnn,cnn_lstm,cnn_tf,hybrid --seed $s --tag s$s
  done
  run 1 $L --window cycle --model cnn,hybrid --n-perm 100 --tag perm100
  echo "### $(date '+%F %T') CYCLE DONE" >> $L
) &
wait
echo "ALL DONE $(date '+%F %T')"
