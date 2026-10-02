#!/usr/bin/env python3
"""EEGNet(원파형) 대 LightGBM(추출특징)을 **완전히 동일한 이벤트 집합**에서 비교한다.

1차 비교에서 두 모델의 표본이 어긋나 있었다. 원인이 둘이다.

  * 코호트 필터가 다름 — EEGNet은 ml_direction.load_data()의 axis_a,
    LightGBM은 stats 산출물의 axis_b를 물려받았다.
  * 결측 처리 지점이 다름 — LightGBM은 8채널이 전부 품질탈락한 이벤트를
    특징행 생성 단계에서 잃고, EEGNet은 원파형이 있으면 남는다.

베이스라인 MAE가 24.62 대 24.69로 거의 같아 난이도는 비슷해 보였지만, '동등하다'를
주장하려면 표본이 같아야 한다. (visit_id, session, trans_id) 교집합으로 양쪽을
자르고 같은 LOSO 분할에서 다시 돌린다.

사용:
  python3 ml_match_sets.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

OUT_DIR = Path('result/dl')
KEY = ['visit_id', 'session', 'trans_id']


def keyset(df, subj_col):
    return set(zip(df[subj_col].astype(str), df['session'].astype(str),
                   df['trans_id'].astype(int)))


def main():
    import torch
    torch.set_num_threads(2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import ml_decoder as MD
    import ml_direction as MDIR
    import ml_eegnet as EN

    # ---- 두 표본의 교집합 ----
    ev = pd.read_csv('result/ml/events.csv', low_memory=False)
    X, meta = MDIR.load_data()
    # load_data()가 이미 visit_key를 붙이므로 rename하면 열이 중복된다.
    meta = meta.loc[:, ~meta.columns.duplicated()].copy()
    m_load = meta['load_pct'].notna().values
    X, meta = X[m_load], meta[m_load].reset_index(drop=True)

    ka = keyset(ev, 'visit_id')
    kb = keyset(meta, 'subject')
    common = ka & kb
    print(f"events.csv {len(ka)} · epochs {len(kb)} · 교집합 {len(common)}")

    ev_k = list(zip(ev['visit_id'].astype(str), ev['session'].astype(str),
                    ev['trans_id'].astype(int)))
    mt_k = list(zip(meta['subject'].astype(str), meta['session'].astype(str),
                    meta['trans_id'].astype(int)))
    ev = ev[[k in common for k in ev_k]].reset_index(drop=True)
    sel = np.array([k in common for k in mt_k])
    X, meta = X[sel], meta[sel].reset_index(drop=True)

    # 두 표본의 행 순서를 키로 맞춰 타깃까지 일치하는지 확인한다.
    ev = ev.sort_values(KEY).reset_index(drop=True)
    order = np.lexsort((meta['trans_id'].values, meta['session'].values,
                        meta['subject'].values))
    X, meta = X[order], meta.iloc[order].reset_index(drop=True)
    assert len(ev) == len(meta) == len(common), (len(ev), len(meta), len(common))
    dy = np.abs(ev['load_pct'].values - meta['load_pct'].values.astype(float))
    print(f"  정렬 확인: 타깃 최대 불일치 {np.nanmax(dy):.4g} · "
          f"피험자 라벨 일치 {(ev['subject_id'].values == meta['subject_id'].values).all()}")

    # ---- LightGBM (동일 집합) ----
    print("\n=== LightGBM (nested LOSO, 교집합) ===", flush=True)
    feats = MD.feature_cols(ev)
    pred = MD.run_loso(ev, feats, nested=True)
    lgb = MD.per_subject_summary(pred)
    lgb_mae = lgb.set_index('subject_id')['mae']
    print(f"  MAE 피험자평균 {lgb['mae'].mean():.3f} · pooled "
          f"{np.abs(pred['y'] - pred['pred']).mean():.3f}")

    # ---- EEGNet (동일 집합) ----
    print("\n=== EEGNet (LOSO, 교집합) ===", flush=True)
    Xn = EN.normalise(X, 'none')
    y = meta['load_pct'].astype(np.float32).values
    per, preds = EN.run_loso(Xn, y, meta, 'reg')
    eeg_mae = per.set_index('subject_id')['mae']
    ok = ~np.isnan(preds)
    print(f"  MAE 피험자평균 {per['mae'].mean():.3f} · pooled "
          f"{np.abs(preds[ok] - y[ok]).mean():.3f}")

    # ---- 대응 비교 ----
    from scipy import stats as sps
    j = pd.concat([lgb_mae.rename('lgb'), eeg_mae.rename('eeg')], axis=1).dropna()
    d = j['eeg'] - j['lgb']
    se = d.std(ddof=1) / np.sqrt(len(d))
    t, p = sps.ttest_rel(j['eeg'], j['lgb'])
    out = {
        'n_event': int(len(common)), 'n_subject': int(len(j)),
        'lgb_mae_subj_mean': float(j['lgb'].mean()),
        'eeg_mae_subj_mean': float(j['eeg'].mean()),
        'lgb_r2_pooled': float(1 - ((pred['y'] - pred['pred']) ** 2).sum()
                               / ((pred['y'] - pred['y'].mean()) ** 2).sum()),
        'eeg_r2_pooled': float(1 - ((y[ok] - preds[ok]) ** 2).sum()
                               / ((y[ok] - y[ok].mean()) ** 2).sum()),
        'diff_mean': float(d.mean()),
        'diff_ci95_lo': float(d.mean() - 1.96 * se),
        'diff_ci95_hi': float(d.mean() + 1.96 * se),
        'p_paired_t': float(p),
        'p_wilcoxon': float(sps.wilcoxon(j['eeg'], j['lgb']).pvalue),
        'n_eeg_better': int((d < 0).sum()),
        'corr_subject_mae': float(np.corrcoef(j['lgb'], j['eeg'])[0, 1]),
    }
    j.to_csv(OUT_DIR / 'matched_per_subject.csv')
    (OUT_DIR / 'matched_summary.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== 동일 집합 비교 ===')
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
