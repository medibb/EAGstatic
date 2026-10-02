#!/usr/bin/env python3
"""EAG 부하 디코더 — EAG 신호만으로 검사측 다리가 받은 체중부하율을 추정한다.

통계 모형은 집단 수준 연관을 추정한다. 여기서 묻는 것은 다른 질문이다.
"한 사람의 무릎 신호만 보고 그 다리가 실제로 받은 부하를 알아낼 수 있는가."
착용형 모니터가 되려면 이 성질이 필요하고, 이건 회귀계수가 아니라 예측으로만
확인된다.

핵심 설계 3가지:
  1. GRF 파생 변수 전면 배제 — 타깃(load_pct)이 GRF에서 나오므로 grf_step 등을
     넣으면 타깃 누출이다. EAG에서 나온 값만 쓴다.
  2. LOSO — 같은 사람이 학습과 평가에 동시에 들어가지 않는다. 개인차가 큰
     생체신호에서 이걸 안 하면 성능이 통째로 과대평가된다(3장의 교훈).
  3. cross-condition transfer — 한 조건에서 배운 디코더가 다른 조건에서도
     작동하는지가 중심 평가다. s→f는 부하 방향 불변성을, {s,f}→c는 목발로
     체중 일부를 흘렸을 때 실제 다리 부하를 따라가는지를 본다.

사용:
  python3 ml_decoder.py --stage loso        # 기본 LOSO 성능 + 베이스라인 대비
  python3 ml_decoder.py --stage transfer    # 조건 간 일반화 행렬
  python3 ml_decoder.py --stage perm --n-perm 100
  python3 ml_decoder.py --stage all
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

warnings.filterwarnings('ignore')

IN_CSV = Path('result/stats/grf_eag_pooled.csv')
OUT_DIR = Path('result/ml')
CHANNELS = list(range(1, 9))
MEDIAL, LATERAL = [1, 2, 3, 4], [5, 6, 7, 8]
# 이벤트 단위로 펼칠 채널별 파라미터. 전부 EAG에서 산출된 값이다.
PER_CH = ['amplitude', 'abs_amp', 'slope', 'transition_time', 'latency']
# 타깃이 GRF에서 나오므로 GRF 파생은 특징에서 완전히 뺀다(타깃 누출 방지).
BANNED = ['grf_step', 'grf_from_level', 'grf_to_level', 'grf_direction',
          'load_pct', 'cycle_id', 'event_kind', 'trans_time', 'test_side']


# ==================== 이벤트 단위 특징 ====================

def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """행(이벤트×채널)을 이벤트 1행으로 접고 8채널 공간벡터를 특징으로 만든다.

    품질필터로 빠진 채널은 결측으로 남긴다. LightGBM은 결측을 분기에서 직접
    처리하므로, 8채널이 모두 살아있는 이벤트(3,151개)만 쓰는 것보다 전체
    7,783개를 쓰는 쪽이 표본을 훨씬 크게 유지한다.
    """
    d = df.dropna(subset=['load_pct']).copy()
    d['abs_amp'] = d['amplitude'].abs()
    key = ['subject_id', 'visit_id', 'session', 'condition', 'trans_id']

    wide = d.pivot_table(index=key, columns='channel', values=PER_CH)
    wide.columns = [f'{p}_ch{int(c)}' for p, c in wide.columns]
    wide = wide.reset_index()

    meta = (d.groupby(key)
              .agg(load_pct=('load_pct', 'first'),
                   eag_direction=('eag_direction', 'first'),
                   n_ch=('channel', 'nunique'))
              .reset_index())
    ev = wide.merge(meta, on=key, how='left')

    # 공간 요약 — 개별 채널이 빠져도 남는 정보
    amp_cols = [f'abs_amp_ch{c}' for c in CHANNELS if f'abs_amp_ch{c}' in ev]
    med = [f'abs_amp_ch{c}' for c in MEDIAL if f'abs_amp_ch{c}' in ev]
    lat = [f'abs_amp_ch{c}' for c in LATERAL if f'abs_amp_ch{c}' in ev]
    ev['amp_mean'] = ev[amp_cols].mean(axis=1)
    ev['amp_max'] = ev[amp_cols].max(axis=1)
    ev['amp_sd'] = ev[amp_cols].std(axis=1)
    ev['amp_medial'] = ev[med].mean(axis=1)
    ev['amp_lateral'] = ev[lat].mean(axis=1)
    ev['med_lat_ratio'] = ev['amp_medial'] / (ev['amp_lateral'] + 1e-6)
    # 진폭을 지운 순수 지형 — 크기가 아니라 패턴만 남긴 특징
    for c in CHANNELS:
        col = f'abs_amp_ch{c}'
        if col in ev:
            ev[f'topo_ch{c}'] = ev[col] / (ev['amp_mean'] + 1e-6)
    ev['is_rise'] = (ev['eag_direction'] == 'rise').astype(int)
    return ev


def feature_cols(ev: pd.DataFrame):
    drop = set(['subject_id', 'visit_id', 'session', 'condition', 'trans_id',
                'eag_direction']) | set(BANNED)
    return [c for c in ev.columns
            if c not in drop and pd.api.types.is_numeric_dtype(ev[c])]


# ==================== 모델 ====================

def make_model(params=None):
    import lightgbm as lgb
    p = dict(objective='regression', n_estimators=400, learning_rate=0.05,
             num_leaves=15, max_depth=4, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             reg_lambda=5.0, verbose=-1, n_jobs=2)
    # 독립 피험자가 43명뿐이라 얕고 강하게 정칙화한 트리를 쓴다.
    if params:
        p.update(params)
    return lgb.LGBMRegressor(**p)


GRID = [dict(num_leaves=7, max_depth=3, learning_rate=0.05),
        dict(num_leaves=15, max_depth=4, learning_rate=0.05),
        dict(num_leaves=31, max_depth=5, learning_rate=0.03)]


def _fit_predict(tr, te, X, y, params=None):
    m = make_model(params)
    m.fit(tr[X], tr[y])
    return m.predict(te[X]), m


def metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - np.mean(y_true)) ** 2).sum())
    return {'n': len(err), 'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt((err ** 2).mean())),
            'r2': float(1 - ss_res / ss_tot) if ss_tot else float('nan'),
            'bias': float(err.mean()),
            'loa_lo': float(err.mean() - 1.96 * err.std(ddof=1)),
            'loa_hi': float(err.mean() + 1.96 * err.std(ddof=1))}


# ==================== LOSO ====================

def run_loso(ev, X, nested=True, params=None, seed=0):
    """피험자 단위 leave-one-out. nested=True면 안쪽에서 하이퍼파라미터를 고른다."""
    from sklearn.model_selection import GroupKFold
    subs = sorted(ev['subject_id'].unique())
    preds = []
    for s in subs:
        tr, te = ev[ev['subject_id'] != s], ev[ev['subject_id'] == s]
        if len(te) < 5 or len(tr) < 50:
            continue
        best = params
        if nested and params is None:
            gkf = GroupKFold(n_splits=3)
            scores = []
            for cand in GRID:
                errs = []
                for itr, iva in gkf.split(tr, groups=tr['subject_id']):
                    p, _ = _fit_predict(tr.iloc[itr], tr.iloc[iva], X, 'load_pct', cand)
                    errs.append(np.abs(p - tr.iloc[iva]['load_pct'].values).mean())
                scores.append(np.mean(errs))
            best = GRID[int(np.argmin(scores))]
        p, _ = _fit_predict(tr, te, X, 'load_pct', best)
        preds.append(pd.DataFrame({'subject_id': s, 'condition': te['condition'].values,
                                   'y': te['load_pct'].values, 'pred': p}))
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def per_subject_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """피험자별 성능을 먼저 내고 그 요약을 보고한다.

    이벤트를 통째로 모아 하나의 MAE를 내면 이벤트가 많은 피험자가 결과를
    끌고 간다. 독립 단위가 피험자이므로 피험자별로 낸 뒤 평균낸다.
    """
    rows = []
    for s, g in pred.groupby('subject_id'):
        m = metrics(g['y'].values, g['pred'].values)
        m['subject_id'] = s
        rows.append(m)
    return pd.DataFrame(rows)


def baselines(ev, X):
    """ML을 채택할 근거가 되는 비교선.

    4.2.11에서 'ML은 혼합모형 베이스라인을 이길 때만 채택한다'고 규정했으므로
    (1) 평균만 내는 모형, (2) 진폭 하나로 예측하는 선형회귀, (3) 전체 특징
    Ridge 를 같은 LOSO로 돌려 비교한다.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    out = {}
    subs = sorted(ev['subject_id'].unique())

    for name in ('mean', 'amp_only', 'ridge_all'):
        rows = []
        for s in subs:
            tr, te = ev[ev['subject_id'] != s], ev[ev['subject_id'] == s]
            if len(te) < 5 or len(tr) < 50:
                continue
            if name == 'mean':
                p = np.full(len(te), tr['load_pct'].mean())
            elif name == 'amp_only':
                t = tr.dropna(subset=['amp_mean'])
                lr = sps.linregress(t['amp_mean'], t['load_pct'])
                p = lr.slope * te['amp_mean'].fillna(t['amp_mean'].mean()) + lr.intercept
            else:
                pipe = make_pipeline(SimpleImputer(strategy='median'),
                                     StandardScaler(), Ridge(alpha=10.0))
                pipe.fit(tr[X], tr['load_pct'])
                p = pipe.predict(te[X])
            rows.append(pd.DataFrame({'subject_id': s, 'y': te['load_pct'].values,
                                      'pred': np.asarray(p)}))
        out[name] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


# ==================== cross-condition transfer ====================

def run_transfer(ev, X, params=None):
    """학습 조건과 평가 조건을 다르게 두되, 피험자 분리는 그대로 유지한다.

    같은 사람이 두 조건에 모두 있으므로 조건만 나누면 개인 정보가 새어
    전이 성능이 부풀려진다. 피험자 i를 평가할 때는 학습에서도 i를 뺀다.
    """
    combos = [(('s',), 's'), (('f',), 'f'), (('c',), 'c'),
              (('s',), 'f'), (('f',), 's'),
              (('s', 'f'), 'c'), (('c',), 's'), (('s', 'f', 'c'), 'c')]
    rows = []
    for tr_conds, te_cond in combos:
        preds = []
        for s in sorted(ev['subject_id'].unique()):
            tr = ev[(ev['condition'].isin(tr_conds)) & (ev['subject_id'] != s)]
            te = ev[(ev['condition'] == te_cond) & (ev['subject_id'] == s)]
            if len(te) < 5 or len(tr) < 50:
                continue
            p, _ = _fit_predict(tr, te, X, 'load_pct', params)
            preds.append(pd.DataFrame({'subject_id': s, 'y': te['load_pct'].values,
                                       'pred': p}))
        if not preds:
            continue
        pr = pd.concat(preds, ignore_index=True)
        ps = per_subject_summary(pr)
        rows.append({'train': '+'.join(tr_conds), 'test': te_cond,
                     'n_subj': pr['subject_id'].nunique(), 'n_events': len(pr),
                     'mae_pooled': round(metrics(pr['y'], pr['pred'])['mae'], 2),
                     'mae_subj_mean': round(ps['mae'].mean(), 2),
                     'mae_subj_sd': round(ps['mae'].std(), 2),
                     'r2_pooled': round(metrics(pr['y'], pr['pred'])['r2'], 3),
                     'bias': round(metrics(pr['y'], pr['pred'])['bias'], 2),
                     'within_condition': tr_conds == (te_cond,)})
    return pd.DataFrame(rows)


# ==================== permutation ====================

def run_permutation(ev, X, n_perm=100, params=None, seed=20260804):
    """피험자 내에서 타깃을 섞어 귀무분포를 만든다.

    전체를 섞으면 '피험자 간 평균 차이'만 깨져 검정이 느슨해진다. 피험자
    안에서만 섞으면 개인 수준 구조는 유지한 채 EAG-부하 대응만 끊긴다.
    """
    rng = np.random.default_rng(seed)
    obs = per_subject_summary(run_loso(ev, X, nested=False, params=params))['mae'].mean()
    null = []
    for i in range(n_perm):
        e = ev.copy()
        e['load_pct'] = e.groupby('subject_id')['load_pct'].transform(
            lambda v: rng.permutation(v.values))
        null.append(per_subject_summary(
            run_loso(e, X, nested=False, params=params))['mae'].mean())
        if (i + 1) % 10 == 0:
            print(f"  perm {i+1}/{n_perm} · null MAE 중앙값 "
                  f"{np.median(null):.2f} (관측 {obs:.2f})", flush=True)
    null = np.array(null)
    return {'observed_mae': float(obs), 'null_mae_mean': float(null.mean()),
            'null_mae_sd': float(null.std(ddof=1)),
            'p_value': float((null <= obs).mean()), 'n_perm': n_perm}


# ==================== 순서형 · 임상판정 · 해석 ====================

def load_to_step(load_pct):
    """실측 부하율을 설계 4단계(20/50/80/100%)로 되돌린다.

    실측이 설계값보다 일관되게 낮으므로 경계는 설계값의 중점이 아니라
    관측 분포의 골(약 34 / 61 / 84%)에 둔다.
    """
    return np.digitize(load_pct, [34, 61, 84])       # 0,1,2,3


def run_ordinal(ev, X, params=None):
    """4단계 순서형 판정. 등급이 순서를 가지므로 QWK로 평가한다.

    정확도는 '20%를 100%로 본 오류'와 '20%를 50%로 본 오류'를 똑같이 세지만,
    임상에서 두 오류의 무게는 다르다. quadratic weighted kappa는 등급 거리의
    제곱으로 벌점을 줘 그 차이를 반영한다.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    e = ev.copy()
    e['step'] = load_to_step(e['load_pct'])
    pred = run_loso(e, X, nested=False, params=params)
    if pred.empty:
        return {}, pd.DataFrame()
    pred['step_true'] = load_to_step(pred['y'])
    pred['step_pred'] = np.clip(load_to_step(pred['pred']), 0, 3)
    qwk = cohen_kappa_score(pred['step_true'], pred['step_pred'], weights='quadratic')
    acc = float((pred['step_true'] == pred['step_pred']).mean())
    adj = float((abs(pred['step_true'] - pred['step_pred']) <= 1).mean())
    cm = pd.DataFrame(confusion_matrix(pred['step_true'], pred['step_pred'],
                                       labels=[0, 1, 2, 3]),
                      index=[f'true_{s}' for s in ['20', '50', '80', '100']],
                      columns=[f'pred_{s}' for s in ['20', '50', '80', '100']])
    return {'qwk': round(qwk, 3), 'accuracy': round(acc, 3),
            'within_1_step': round(adj, 3), 'n': len(pred)}, cm


def run_adherence_roc(pred: pd.DataFrame, thresholds=(30, 50, 80)) -> pd.DataFrame:
    """처방 준수 판정 성능. '정확히 몇 %인가'보다 임상에서 먼저 필요한 질문이다.

    이벤트 하나로 판정하는 것과 세션 단위로 평균 내 판정하는 것을 함께 본다.
    개별 이벤트의 LoA가 넓으므로, 실용성은 집계 수준에서 나올 가능성이 크다.
    """
    from sklearn.metrics import roc_auc_score
    # 집계 단위에 부하단계를 남겨야 한다. 피험자×조건으로만 묶으면 한 세션 안의
    # 20~100%가 모두 평균되어 실제 부하가 55% 근처로 뭉개지고 판정 대상이 사라진다.
    # 임상 단위는 '한 번의 하중 유지(cycle)'이므로 단계를 유지한 채 평균한다.
    p = pred.copy()
    p['step'] = load_to_step(p['y'])
    agg = (p.groupby(['subject_id', 'condition', 'step'], as_index=False)
             .agg(y=('y', 'mean'), pred=('pred', 'mean')))
    rows = []
    for thr in thresholds:
        for level, d in [('event', pred), ('stance (subject x condition x step)', agg)]:
            yt = (d['y'] > thr).astype(int)
            if yt.nunique() < 2:
                continue
            auc = roc_auc_score(yt, d['pred'])
            yp = (d['pred'] > thr).astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
            rows.append({'threshold_pctBW': thr, 'level': level, 'n': len(d),
                         'auc': round(float(auc), 3),
                         'sensitivity': round(tp / (tp + fn), 3) if tp + fn else np.nan,
                         'specificity': round(tn / (tn + fp), 3) if tn + fp else np.nan,
                         'accuracy': round((tp + tn) / len(d), 3)})
    return pd.DataFrame(rows)


def run_shap(ev, X, params=None, n_sample=3000, seed=20260804):
    """held-out 예측에 대한 SHAP. 학습셋에서 뽑으면 과적합된 설명이 된다."""
    try:
        import shap
    except Exception:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    subs = sorted(ev['subject_id'].unique())
    cut = len(subs) // 2
    tr = ev[ev['subject_id'].isin(subs[:cut])]
    te = ev[ev['subject_id'].isin(subs[cut:])]
    if len(te) > n_sample:
        te = te.iloc[rng.choice(len(te), n_sample, replace=False)]
    m = make_model(params)
    m.fit(tr[X], tr['load_pct'])
    sv = shap.TreeExplainer(m).shap_values(te[X])
    imp = pd.DataFrame({'feature': X, 'mean_abs_shap': np.abs(sv).mean(axis=0)})
    return imp.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def run_channel_ablation(ev, X, params=None):
    """전극을 몇 개까지 줄일 수 있는지. 착용형 실현 가능성의 핵심 질문이다.

    SHAP 중요도가 아니라 실제 재학습 성능으로 판단한다. 중요도가 높은 특징을
    지워도 상관된 다른 특징이 대신하는 경우가 흔하기 때문이다.
    """
    sets = [('all 8', CHANNELS), ('medial only (1-4)', MEDIAL),
            ('lateral only (5-8)', LATERAL), ('Ch2+Ch3 (medial hot)', [2, 3]),
            ('Ch1+Ch2+Ch3', [1, 2, 3]), ('Ch2 only', [2]), ('Ch3 only', [3]),
            ('Ch2+Ch7 (med+lat)', [2, 7])]
    rows = []
    for name, chs in sets:
        keep = [c for c in X
                if (not any(f'ch{k}' in c for k in CHANNELS))
                or any(c.endswith(f'ch{k}') for k in chs)]
        # 8채널 전체에서 산출한 공간요약은 채널을 줄이면 쓸 수 없다
        if len(chs) < 8:
            keep = [c for c in keep if c not in
                    ('amp_mean', 'amp_max', 'amp_sd', 'amp_medial',
                     'amp_lateral', 'med_lat_ratio')]
        pred = run_loso(ev, keep, nested=False, params=params)
        if pred.empty:
            continue
        ps = per_subject_summary(pred)
        rows.append({'channel_set': name, 'n_channels': len(chs),
                     'n_features': len(keep),
                     'mae_subj_mean': round(ps['mae'].mean(), 2),
                     'mae_subj_sd': round(ps['mae'].std(), 2),
                     'r2_pooled': round(metrics(pred['y'], pred['pred'])['r2'], 3)})
    return pd.DataFrame(rows)


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='loso',
                    choices=['loso', 'transfer', 'perm', 'ordinal', 'roc',
                             'shap', 'ablation', 'interpret', 'all'])
    ap.add_argument('--n-perm', type=int, default=100)
    ap.add_argument('--axis', choices=['none', 'a', 'b'], default='b')
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV, low_memory=False)
    if a.axis != 'none' and f'axis_{a.axis}' in df.columns:
        df = df[df[f'axis_{a.axis}'].fillna(False).astype(bool)]
    ev = build_events(df)
    X = feature_cols(ev)
    print(f"이벤트 {len(ev)} · 피험자 {ev['subject_id'].nunique()}명 · 특징 {len(X)}개 "
          f"(axis={a.axis})")
    print(f"  조건별 이벤트: {ev['condition'].value_counts().to_dict()}")
    ev.to_csv(OUT_DIR / 'events.csv', index=False)

    if a.stage in ('loso', 'all'):
        pred = run_loso(ev, X, nested=True)
        pred.to_csv(OUT_DIR / 'loso_predictions.csv', index=False)
        ps = per_subject_summary(pred)
        ps.to_csv(OUT_DIR / 'loso_per_subject.csv', index=False)
        pooled = metrics(pred['y'].values, pred['pred'].values)
        rows = [{'model': 'LightGBM (nested LOSO)', 'mae_subj_mean': ps['mae'].mean(),
                 'mae_subj_sd': ps['mae'].std(), 'mae_pooled': pooled['mae'],
                 'r2_pooled': pooled['r2'], 'bias': pooled['bias'],
                 'loa_lo': pooled['loa_lo'], 'loa_hi': pooled['loa_hi']}]
        for name, bp in baselines(ev, X).items():
            if len(bp):
                b = metrics(bp['y'].values, bp['pred'].values)
                bs = per_subject_summary(bp)
                rows.append({'model': f'baseline: {name}',
                             'mae_subj_mean': bs['mae'].mean(),
                             'mae_subj_sd': bs['mae'].std(), 'mae_pooled': b['mae'],
                             'r2_pooled': b['r2'], 'bias': b['bias'],
                             'loa_lo': b['loa_lo'], 'loa_hi': b['loa_hi']})
        comp = pd.DataFrame(rows).round(3)
        comp.to_csv(OUT_DIR / 'loso_vs_baselines.csv', index=False)
        print("\n=== LOSO 성능 (부하 %BW) ===")
        print(comp.to_string(index=False))

    if a.stage in ('transfer', 'all'):
        tr = run_transfer(ev, X)
        tr.to_csv(OUT_DIR / 'cross_condition_transfer.csv', index=False)
        print("\n=== Cross-condition transfer (피험자 분리 유지) ===")
        print(tr.to_string(index=False))

    if a.stage in ('ordinal', 'interpret', 'all'):
        res, cm = run_ordinal(ev, X)
        if res:
            pd.DataFrame([res]).to_csv(OUT_DIR / 'ordinal_metrics.csv', index=False)
            cm.to_csv(OUT_DIR / 'ordinal_confusion.csv')
            print("\n=== 순서형 4단계 (LOSO) ===")
            print(f"  QWK={res['qwk']} · 정확도={res['accuracy']} · "
                  f"±1단계 이내={res['within_1_step']} (n={res['n']})")
            print(cm.to_string())

    if a.stage in ('roc', 'interpret', 'all'):
        p = OUT_DIR / 'loso_predictions.csv'
        if p.exists():
            pred = pd.read_csv(p)
            roc = run_adherence_roc(pred)
            roc.to_csv(OUT_DIR / 'adherence_roc.csv', index=False)
            print("\n=== 처방 준수 판정 (ROC) ===")
            print(roc.to_string(index=False))

    if a.stage in ('ablation', 'interpret', 'all'):
        ab = run_channel_ablation(ev, X)
        if len(ab):
            ab.to_csv(OUT_DIR / 'channel_ablation.csv', index=False)
            print("\n=== 채널 축소 ablation ===")
            print(ab.to_string(index=False))

    if a.stage in ('shap', 'interpret', 'all'):
        sh = run_shap(ev, X)
        if len(sh):
            sh.to_csv(OUT_DIR / 'shap_importance.csv', index=False)
            print("\n=== SHAP 상위 15 특징 (held-out) ===")
            print(sh.head(15).to_string(index=False))
        else:
            print("\n[SHAP] shap 미설치 → 건너뜀")

    if a.stage in ('perm', 'all'):
        r = run_permutation(ev, X, a.n_perm)
        pd.DataFrame([r]).to_csv(OUT_DIR / 'permutation_test.csv', index=False)
        print("\n=== Permutation (피험자 내 라벨 셔플) ===")
        print(f"  관측 MAE {r['observed_mae']:.2f} · 귀무 MAE "
              f"{r['null_mae_mean']:.2f} ± {r['null_mae_sd']:.2f} · p={r['p_value']:.4f}")

    print(f"\n산출물: {OUT_DIR}/")


if __name__ == '__main__':
    main()
