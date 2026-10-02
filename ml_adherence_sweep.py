#!/usr/bin/env python3
"""부하 디코더의 순응도 판별 성능을 문턱 10~95 %BW로 훑는다 (Table 4.11 확장).

ml_decoder의 LOSO 예측(result/ml/loso_predictions.csv, 실측 y vs 예측 pred)을 그대로 쓰므로
학습은 없다. 임상 질문은 "처방 한계 X %BW를 넘었는가"이고, 그 X가 어디까지 신뢰되는지가
표에 없었다. 이벤트 단위와 한발서기 단위(subject × condition × step 평균)를 함께 낸다.

주의: 실측 부하가 4단계(중앙값 19/47/73/92)에 뭉쳐 있어 AUC는 문턱에서 먼 쌍이 지배한다.
군집 안을 가르는 문턱(45, 55 등)의 AUC가 높다고 그 해상도가 있다는 뜻은 아니다. 그래서
문턱 ±10 %BW 안의 표본 비율(near)과, 그 근처 표본만으로 잰 AUC(auc_near)를 같이 낸다.

사용: python3 ml_adherence_sweep.py   → result/ml/adherence_sweep.csv, result/ml/fig/fig9_adherence_sweep.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ML = Path('result/ml')
FIG = ML / 'fig'
OBS_IMG = Path('/workspace/obsidian/images/eag_crutch')
STEP_EDGES = [34, 61, 84]
BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, SURF = ('#2a78d6', '#eb6834', '#0b0b0b', '#52514e',
                                                     '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb')


def youden(yb, p):
    fpr, tpr, thr = roc_curve(yb, p)
    k = int(np.argmax(tpr - fpr))
    return float(tpr[k]), float(1 - fpr[k]), float(thr[k])


def sweep(x, name):
    rows = []
    for thr in range(10, 96, 5):
        yb = (x.y > thr).astype(int)
        if yb.nunique() < 2 or yb.mean() < 0.03 or yb.mean() > 0.97:
            rows.append({'level': name, 'thr': thr, 'n': len(x), 'pos_frac': round(yb.mean(), 3)})
            continue
        near = (x.y > thr - 10) & (x.y <= thr + 10)
        r = {'level': name, 'thr': thr, 'n': len(x), 'pos_frac': round(yb.mean(), 3),
             'auc': round(roc_auc_score(yb, x.pred), 3), 'near_frac': round(near.mean(), 3)}
        if near.sum() >= 30 and yb[near].nunique() == 2:
            r['auc_near'] = round(roc_auc_score(yb[near], x.pred[near]), 3)
        se, sp, cut = youden(yb, x.pred)
        r.update(sens_youden=round(se, 3), spec_youden=round(sp, 3), cut_youden=round(cut, 1))
        # 처방 한계를 그대로 절단점으로 썼을 때 (보정 없는 운용)
        pr = (x.pred > thr).astype(int)
        r['sens_at_thr'] = round(((pr == 1) & (yb == 1)).sum() / max(1, (yb == 1).sum()), 3)
        r['spec_at_thr'] = round(((pr == 0) & (yb == 0)).sum() / max(1, (yb == 0).sum()), 3)
        rows.append(r)
    return rows


def main():
    d = pd.read_csv(ML / 'loso_predictions.csv')
    d['step'] = np.digitize(d.y, STEP_EDGES)
    st = d.groupby(['subject_id', 'condition', 'step']).agg(y=('y', 'mean'), pred=('pred', 'mean')).reset_index()
    t = pd.DataFrame(sweep(d, 'event') + sweep(st, 'stance'))
    t.to_csv(ML / 'adherence_sweep.csv', index=False)
    print(t.to_string(index=False))

    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                         'axes.edgecolor': AXIS, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.labelcolor': INK2, 'text.color': INK, 'font.size': 9,
                         'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'wspace': 0.28})
    for lvl, col in [('stance', BLUE), ('event', ORANGE)]:
        g = t[(t.level == lvl) & t.auc.notna()]
        ax[0].plot(g.thr, g.auc, '-o', color=col, ms=5, label=f'{lvl} level')
    ax[0].axhline(0.9, color=AXIS, lw=0.8)
    ax[0].text(94, 0.902, 'AUC 0.9', ha='right', va='bottom', fontsize=8, color=MUTED)
    ax[0].axvspan(25, 70, color=GRID, alpha=0.5, lw=0)
    ax[0].text(47.5, 0.997, 'usable range 25–70 %BW (stance AUC ≥ 0.94)', ha='center', va='top',
               fontsize=8, color=INK2)
    ax[0].set_xlabel('prescribed limit (%BW)'); ax[0].set_ylabel('AUC, LOSO'); ax[0].set_ylim(0.75, 1.0)
    ax[0].set_title('How far can the limit be moved?', loc='left'); ax[0].legend(fontsize=7.5, loc='lower left')
    ax[0].grid(axis='x', visible=False)
    g = t[(t.level == 'stance') & t.auc.notna()]
    ax[1].plot(g.thr, g.sens_at_thr, '-o', color=BLUE, ms=5, label='sensitivity (overload detected)')
    ax[1].plot(g.thr, g.spec_at_thr, '-o', color=ORANGE, ms=5, label='specificity (compliance recognised)')
    ax[1].axvspan(25, 70, color=GRID, alpha=0.5, lw=0)
    ax[1].set_xlabel('prescribed limit (%BW), used directly as cut-off'); ax[1].set_ylabel('stance level')
    ax[1].set_ylim(0, 1.02); ax[1].set_title('Operating point without recalibration', loc='left')
    ax[1].legend(fontsize=7.5, loc='lower left'); ax[1].grid(axis='x', visible=False)
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / 'fig9_adherence_sweep.png'
    fig.savefig(p, dpi=160, bbox_inches='tight')
    OBS_IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OBS_IMG / p.name, dpi=160, bbox_inches='tight')
    print('->', p)


if __name__ == '__main__':
    main()
