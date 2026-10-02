#!/bin/bash
# 목발 분류 1차 실험 러너 (NAS, CPU). 한 번에 한 작업만 돌려 bot/api-server를 보호한다.
#   event 창: DL 4종 + csp + lgb → cycle 창: DL 4종 + lgb + dur → 경량 모델 순열
# DL 순열은 NAS에서 비현실적(LOSO 1회 × 100)이라 cnn 20회만, 나머지는 3090으로 넘긴다.
cd "$(dirname "$0")"
LOG=result/ml/crutch_run.log
run() { echo "### $(date '+%F %T') $*" >> "$LOG"; nice -n 19 python3 ml_crutch.py "$@" >> "$LOG" 2>&1; }

run --window event --model cnn,cnn_lstm,cnn_tf,hybrid,csp,lgb

# cycle 추출이 끝날 때까지 대기
until [ -f result/ml/cycles.npz ] && ! pgrep -f "epoch_extractor.py --mode cycle" >/dev/null; do sleep 30; done
run --window cycle --model cnn,cnn_lstm,cnn_tf,hybrid,lgb,dur

# 경량 모델 순열 100회, cnn 20회 (event 창)
run --window event --model csp,lgb --n-perm 100 --tag perm
run --window cycle --model lgb,dur --n-perm 100 --tag perm
run --window event --model cnn --n-perm 20 --tag perm
echo "### $(date '+%F %T') ALL DONE" >> "$LOG"
