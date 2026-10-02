#!/usr/bin/env python3
"""목발 분류 피험자 이질성 — 왜 일부 피험자만 구분되는가 (탐색적).

ml_crutch 1차 결과: 모든 모델이 bacc 0.53~0.55이고, 피험자별 bacc가 창·구조와 무관하게
재현된다(DL 8모델 간 Spearman 중앙값 0.46). 즉 신호는 '모델'이 아니라 '사람'에 붙어 있다.
여기서는 피험자별 구분 가능도(decodability)를 공변량과 대조한다. 41명 탐색이므로
가설 생성용이며, Holm 보정 p를 같이 낸다.

구분 가능도 = DL 8모델(event/cycle × cnn/cnn_lstm/cnn_tf/hybrid) 피험자 bacc 평균.
공변량 묶음:
  체성분   sex, age, bmi, fat_pct, muscle                         (cov_per_subject.csv)
  EAG-부하  slope(전체), |diff_c_s| = c와 s의 dose 기울기 차, amp_mean
  세션 순서 c 세션 평균 take − s/f 평균 take (c가 뒤에 몰렸는가), c take 수, 방문 수
  부하 행동 |median load c − median load s/f|, load SD 비(c/sf)  (GRF 안정성 대리)
  채널 품질 n_pass 평균, SNR 평균
추가 검정:
  방문 1 vs 2 bacc (event cnn 예측으로 방문별 재계산, 짝지은 Wilcoxon)
  c 세션 초반 vs 후반 recall (숙련 효과, 짝지은 Wilcoxon)

사용:
  python3 ml_crutch_hetero.py
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings('ignore')
OUT_DIR = Path('result/ml')
COV = Path('result/stats/cov_per_subject.csv')


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def decodability():
    per = {}
    for f in glob.glob(str(OUT_DIR / 'crutch_*_per_subject.csv')):
        k = Path(f).name.replace('crutch_', '').replace('_per_subject.csv', '')
        if 'perm' in k or not any(t in k for t in ('cnn', 'hybrid')):
            continue
        per[k] = pd.read_csv(f).set_index('subject_id')['bacc']
    d = pd.DataFrame(per)
    out = pd.DataFrame({'score': d.mean(axis=1), 'score_sd': d.std(axis=1),
                        'n_models': d.notna().sum(axis=1)})
    if 'event_cnn' in d:
        out['bacc_event_cnn'] = d['event_cnn']
    return out


def session_covariates():
    from ml_crutch import load_window
    _, _, y, meta = load_window('event')
    meta = meta.copy()
    meta['is_c'] = y.astype(bool)
    rows = []
    for s, g in meta.groupby('subject_id'):
        c, sf = g[g.is_c], g[~g.is_c]
        rows.append({'subject_id': s,
                     'take_gap_c_minus_sf': c['take'].mean() - sf['take'].mean(),
                     'n_c_takes': c['session'].nunique(),
                     'n_visits': g['visit_id'].nunique(),
                     'load_med_diff': abs(c['load_pct'].median() - sf['load_pct'].median()),
                     'load_sd_ratio': c['load_pct'].std() / (sf['load_pct'].std() + 1e-9),
                     'n_pass_mean': g['n_pass'].mean(),
                     'snr_mean': g[[f'snr_ch{i}' for i in range(1, 9)]].mean().mean()})
    return pd.DataFrame(rows).set_index('subject_id'), meta, y


def per_visit_and_order(meta, y):
    """event cnn 예측으로 방문별 bacc, c 세션 초반/후반 recall."""
    from sklearn.metrics import balanced_accuracy_score
    p = np.load(OUT_DIR / 'crutch_event_cnn_preds.npy')
    ok = ~np.isnan(p)
    m = meta[ok].copy(); m['y'] = y[ok].astype(int); m['pr'] = (p[ok] > 0).astype(int)
    visit = []
    for (s, v), g in m.groupby(['subject_id', 'visit_id']):
        if g['y'].nunique() == 2:
            vn = 2 if str(v).rstrip().endswith(('_2', '_2.5')) else 1
            visit.append({'subject_id': s, 'visit_no': vn,
                          'bacc': balanced_accuracy_score(g['y'], g['pr'])})
    visit = pd.DataFrame(visit)
    piv = visit.pivot_table(index='subject_id', columns='visit_no', values='bacc')
    order = []
    for s, g in m[m['y'] == 1].groupby('subject_id'):
        med = g['take'].median()
        early, late = g[g['take'] <= med], g[g['take'] > med]
        if len(early) and len(late):
            order.append({'subject_id': s, 'recall_early': early['pr'].mean(),
                          'recall_late': late['pr'].mean()})
    return piv, pd.DataFrame(order).set_index('subject_id')


def main():
    sc = decodability()
    cov = pd.read_csv(COV).set_index('subject_id')
    ses, meta, y = session_covariates()
    d = sc.join(cov, how='left').join(ses, how='left')
    d['abs_diff_c_s'] = d['diff_c_s'].abs()
    d['sex_M'] = (d['sex'] == 'M').astype(float)
    d.to_csv(OUT_DIR / 'crutch_hetero_subject.csv', encoding='utf-8-sig')

    cont = ['age', 'bmi', 'fat_pct', 'muscle', 'slope', 'abs_diff_c_s', 'amp_mean',
            'take_gap_c_minus_sf', 'n_c_takes', 'n_visits', 'load_med_diff',
            'load_sd_ratio', 'n_pass_mean', 'snr_mean']
    rows = []
    for c in cont:
        x = d[[c, 'score']].dropna()
        r = sps.spearmanr(x[c], x['score'])
        rows.append({'covariate': c, 'test': 'spearman', 'n': len(x),
                     'stat': float(r.statistic), 'p': float(r.pvalue)})
    g = d.dropna(subset=['sex'])
    a, b = g.loc[g.sex == 'M', 'score'], g.loc[g.sex == 'F', 'score']
    mw = sps.mannwhitneyu(a, b)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    rows.append({'covariate': 'sex (M-F)', 'test': 'mannwhitney', 'n': len(g),
                 'stat': float((a.mean() - b.mean()) / sp), 'p': float(mw.pvalue)})
    tests = pd.DataFrame(rows)
    tests['p_holm'] = holm(tests['p'])
    tests.to_csv(OUT_DIR / 'crutch_hetero_tests.csv', index=False)

    piv, order = per_visit_and_order(meta, y)
    extra = {}
    both = piv.dropna()
    if len(both) >= 6:
        w = sps.wilcoxon(both[1], both[2])
        extra['visit1_vs_2'] = dict(n=len(both), bacc_v1=float(both[1].mean()),
                                    bacc_v2=float(both[2].mean()), p=float(w.pvalue))
    if len(order) >= 6:
        w = sps.wilcoxon(order['recall_early'], order['recall_late'])
        extra['c_early_vs_late'] = dict(n=len(order), recall_early=float(order['recall_early'].mean()),
                                        recall_late=float(order['recall_late'].mean()), p=float(w.pvalue))

    print(f"피험자 {len(d)}명 · 구분 가능도 score 평균 {d.score.mean():.3f} (sd {d.score.std():.3f}), "
          f"범위 {d.score.min():.3f}~{d.score.max():.3f}")
    print("\n=== 공변량 vs 구분 가능도 (탐색, Holm 보정) ===")
    print(tests.sort_values('p').to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== 방문·숙련 효과 (event cnn 예측) ===")
    for k, v in extra.items():
        print(f"  {k}: {v}")
    print("\n상위 6명:")
    print(d.sort_values('score', ascending=False)[['score', 'sex', 'fat_pct', 'abs_diff_c_s',
                                                    'take_gap_c_minus_sf', 'n_pass_mean']]
          .head(6).round(3).to_string())


if __name__ == '__main__':
    main()
