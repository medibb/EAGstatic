#!/bin/bash
# event CNN 순열 61~100회 + 100회 pooled p. nohup으로 띄워 세션 종료에 영향받지 않게 한다.
#   nohup ./run_perm100.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"
L=result/ml/crutch_holiday.log
echo "##### STAGE 5 (재시작) event cnn perm +40 (60→100)" >> $L
echo "### $(date '+%F %T') --window event --model cnn --n-perm 40 --perm-offset 60 --tag perm3" >> $L
nice -n 19 python3 ml_crutch.py --threads 2 --window event --model cnn --n-perm 40 --perm-offset 60 --tag perm3 >> $L 2>&1
echo "### $(date '+%F %T') done: perm3" >> $L
python3 - >> $L 2>&1 <<'EOF'
import numpy as np, json
ML = 'result/ml/'
null = np.concatenate([np.loadtxt(ML + f) for f in
                       ['crutch_event_cnn_perm_null.txt', 'crutch_event_cnn_perm2_null.txt',
                        'crutch_event_cnn_perm3_null.txt']])
np.savetxt(ML + 'crutch_event_cnn_perm100_null.txt', null)
obs = json.load(open(ML + 'crutch_event_cnn_summary.json'))['bacc_subject_mean']
p = ((null >= obs).sum() + 1) / (len(null) + 1)
print(f"### POOLED event cnn n={len(null)} null={null.mean():.4f}±{null.std():.4f} "
      f"max={null.max():.4f} obs={obs:.4f} p={p:.4f} z={(obs - null.mean()) / null.std():.1f}")
EOF
echo "### $(date '+%F %T') PERM100 DONE" >> $L
