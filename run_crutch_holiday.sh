#!/bin/bash
# 연휴용 무인 러너 (NAS CPU, 2 threads, nice 19). 3090 대체. 가치 순서대로, 단계마다 STOP 파일 확인.
#   중단: touch result/ml/STOP   (현재 단계는 끝까지 돌고 다음 단계로 안 넘어감)
#   예산(1차 실측 기준): 1) 6.6 h  2) 19 h  3) 7 h  4) 24 h  ≈ 57 h
cd "$(dirname "$0")"
LOG=result/ml/crutch_holiday.log
STOP=result/ml/STOP
rm -f "$STOP"
run() { echo "### $(date '+%F %T') $*" >> "$LOG"
        nice -n 19 python3 ml_crutch.py --threads 2 "$@" >> "$LOG" 2>&1
        echo "### $(date '+%F %T') done: $*" >> "$LOG"; }
stage() { [ -f "$STOP" ] && { echo "### $(date '+%F %T') STOP 발견, 종료" >> "$LOG"; exit 0; }; echo "##### STAGE $1" >> "$LOG"; }

# 1) event 창 seed 재현 (seed 1, 2) × DL 4종  → seed 3개 평균 ± sd
stage "1 event seeds"
for s in 1 2; do run --window event --model cnn,cnn_lstm,cnn_tf,hybrid --seed $s --tag s$s; done

# 2) event CNN 순열 이어서 40회 (20 → 60). null은 crutch_event_cnn_perm2_null.txt 에 따로 저장, 합쳐서 p 계산
stage "2 event cnn perm +40"
run --window event --model cnn --n-perm 40 --perm-offset 20 --tag perm2

# 3) cycle 창 seed 재현 (seed 1, 2) × cnn, cnn_lstm (Transformer 계열은 시간 대비 가치 낮아 제외)
stage "3 cycle seeds"
for s in 1 2; do run --window cycle --model cnn,cnn_lstm --seed $s --tag s$s; done

# 4) cycle CNN 순열 20회 (창 B의 순열 p 확보)
stage "4 cycle cnn perm 20"
run --window cycle --model cnn --n-perm 20 --tag perm

echo "### $(date '+%F %T') HOLIDAY ALL DONE" >> "$LOG"
