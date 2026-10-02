#!/usr/bin/env python3
"""목발 분류를 이벤트 단위에서 세션 단위로 올리면 얼마나 오르는가.

ml_crutch의 LOSO 예측(logit)을 세션(=take, 이벤트 8개) 또는 방문×조건 단위로 평균해
다시 판정한다. 학습 없음. 약한 증거가 독립적이라면 √n 만큼 모이고, 세션 안에서 상관되어
있으면 거의 안 오른다. 임상 질문이 "이 세션에서 목발을 썼나"라면 이 값이 해당 수치다.

사용: python3 ml_crutch_session.py  → result/ml/crutch_session_level.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ML = Path('result/ml')


def agg_metrics(df, keys):
    g = df.groupby(['subject_id'] + keys).agg(y=('y', 'first'), p=('p', 'mean'), n=('p', 'size')).reset_index()
    per = []
    for s, gs in g.groupby('subject_id'):
        if gs.y.nunique() < 2:
            continue
        per.append({'subject_id': s, 'n_units': len(gs),
                    'bacc': balanced_accuracy_score(gs.y, (gs.p > 0).astype(int)),
                    'auc': roc_auc_score(gs.y, gs.p)})
    per = pd.DataFrame(per)
    return {'n_units_total': int(len(g)), 'n_subject': int(len(per)),
            'bacc_subject_mean': round(per.bacc.mean(), 3), 'bacc_subject_sd': round(per.bacc.std(), 3),
            'auc_subject_mean': round(per.auc.mean(), 3), 'auc_pooled': round(roc_auc_score(g.y, g.p), 3)}


def main():
    from ml_crutch import load_window
    rows = []
    for window in ['event', 'cycle']:
        _, _, y, meta = load_window(window)
        for model in ['cnn', 'cnn_lstm', 'cnn_tf', 'hybrid', 'lgb']:
            # lgb/csp는 클래스 균형 보정본(perm 태그)이 정본
            order = ['_perm_preds.npy', '_preds.npy'] if model in ('lgb', 'csp') else ['_preds.npy', '_perm_preds.npy']
            f = next((ML / f'crutch_{window}_{model}{s}' for s in order
                      if (ML / f'crutch_{window}_{model}{s}').exists()), None)
            if f is None:
                continue
            p = np.load(f)
            ok = ~np.isnan(p)
            df = meta[ok].copy(); df['y'] = y[ok].astype(int); df['p'] = p[ok]
            # 한 사람이 방문 2회에 같은 세션명(s1…)을 쓰므로 visit_id 를 항상 키에 넣는다
            idc = df.columns[df.columns.str.contains('trans_id|cycle_id')][0]
            for unit, keys in [('event', ['visit_id', 'session', idc]),
                               ('session', ['visit_id', 'session']),
                               ('visit x condition', ['visit_id', 'condition'])]:
                m = agg_metrics(df, keys)
                rows.append({'window': window, 'model': model, 'unit': unit, **m})
    t = pd.DataFrame(rows)
    t.to_csv(ML / 'crutch_session_level.csv', index=False)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
