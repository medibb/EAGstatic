#!/bin/bash
# cycle CNN 순열 21~60회 (약 48 h, 회당 ~72 min). 값은 회마다 crutch_cycle_cnn_perm2_null_partial.txt 에 append.
#   nohup ./run_cycle_perm.sh > /dev/null 2>&1 &
#   중단: python 프로세스를 죽여도 partial 파일은 남는다. 이어 돌릴 때 --perm-offset 20+<partial 줄 수>.
# 끝나면(또는 부분 결과로) pooled p:  python3 run_cycle_perm.sh --pool   (bash가 아닌 python으로 아래 블록 실행)
cd "$(dirname "$0")"
L=result/ml/crutch_holiday.log
pool() { python3 - >> $L 2>&1 <<'EOF'
import numpy as np, json, os
ML = 'result/ml/'
parts = [ML + 'crutch_cycle_cnn_perm_null.txt']
for f in ['crutch_cycle_cnn_perm2_null.txt', 'crutch_cycle_cnn_perm2_null_partial.txt']:
    if os.path.exists(ML + f):
        parts.append(ML + f); break
null = np.concatenate([np.loadtxt(p) for p in parts])
np.savetxt(ML + f'crutch_cycle_cnn_perm{len(null)}_null.txt', null)
obs = json.load(open(ML + 'crutch_cycle_cnn_summary.json'))['bacc_subject_mean']
p = ((null >= obs).sum() + 1) / (len(null) + 1)
print(f"### POOLED cycle cnn n={len(null)} null={null.mean():.4f}±{null.std():.4f} "
      f"max={null.max():.4f} obs={obs:.4f} p={p:.4f} z={(obs - null.mean()) / null.std():.1f}")
EOF
}
if [ "$1" = "--pool" ]; then pool; tail -1 $L; exit 0; fi
echo "##### STAGE 6 cycle cnn perm +40 (20→60)" >> $L
echo "### $(date '+%F %T') --window cycle --model cnn --n-perm 40 --perm-offset 20 --tag perm2" >> $L
rm -f result/ml/crutch_cycle_cnn_perm2_null_partial.txt
nice -n 19 python3 ml_crutch.py --threads 2 --window cycle --model cnn --n-perm 40 --perm-offset 20 --tag perm2 >> $L 2>&1
echo "### $(date '+%F %T') done: cycle perm2" >> $L
pool
echo "### $(date '+%F %T') CYCLE PERM60 DONE" >> $L
