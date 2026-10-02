#!/usr/bin/env python3
"""방향 디코더 — 부하 크기를 지운 뒤 공간 패턴만으로 부하 방향을 분류한다.

묻는 것: 같은 크기의 부하라도 좌우(s)로 실었을 때와 전후(f)로 실었을 때
무릎 주변 전위의 **공간 패턴**이 다른가. 다르다면 EAG는 단순한 부하 센서가
아니라 연골 접촉 분포에 민감한 센서다.

부하 교란을 두 겹으로 막는다:
  1. CSP 공분산을 trace로 정규화 — 시행 전체의 크기 정보가 수학적으로 사라지고
     채널 간 상대 구조만 남는다. 사후 보정이 아니라 구조적 제거다.
  2. 같은 부하 단계 안에서만 s와 f를 비교 — 부하가 방향의 대리변수가 되지 못한다.

CSP(common spatial patterns)는 두 클래스의 분산비를 최대로 가르는 공간 필터를
찾는다. 이를 위해 시행별 채널 간 공분산이 필요하므로 입력은 스칼라 벡터가 아니라
(채널 × 시간) 시계열이어야 한다(epoch_extractor.py).

사용:
  python3 ml_direction.py                     # LOSO + 단계별 + 베이스라인
  python3 ml_direction.py --n-perm 200        # permutation 포함
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

EPOCHS = Path('result/ml/epochs.npz')
POOLED = Path('result/stats/grf_eag_pooled.csv')
OUT_DIR = Path('result/ml')
N_FILT = 2          # 각 클래스에서 고를 필터 수 → 특징 4개. 43명 규모에 맞춘 보수적 설정.


# ==================== 유도(derivation) ====================

# 몽타주: 내측 1(MPFL) 2(근위내측) 3(내측전방) 4(내측후방)
#         외측 5(LPFL) 6(근위외측) 7(외측전방) 8(외측후방)
# 8채널 모두 경골의 한 기준전극을 공유하므로, 기준전극의 잡음·드리프트가 8채널에
# 동일하게 실린다(공통모드). CSP는 채널 간 공분산으로 작동하므로 공통모드가
# 인위적 상관을 만들어 국소 차이를 덮는다. 표면 근전도가 양극 유도를 기본으로
# 쓰는 이유와 같고, BCI에서 CSP 앞단에 Laplacian을 거는 것도 같은 취지다.
BIPOLAR_WITHIN = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)]   # 컬럼 내 인접쌍
BIPOLAR_CROSS = [(1, 5), (2, 6), (3, 7), (4, 8)]                    # 내·외측 대응쌍
CH_ADJ = {1: [2, 3], 2: [1, 3, 6], 3: [1, 2, 4], 4: [3, 8],
          5: [6, 7], 6: [2, 5, 7], 7: [5, 6, 8], 8: [4, 7]}

DERIVATIONS = ['referential', 'bipolar_within', 'bipolar_cross', 'laplacian']


def derivation_matrix(name: str) -> np.ndarray:
    """유도를 8채널에 대한 선형 변환 행렬로 표현한다(재추출 불필요)."""
    if name == 'referential':
        return np.eye(8)
    if name in ('bipolar_within', 'bipolar_cross'):
        pairs = BIPOLAR_WITHIN if name == 'bipolar_within' else BIPOLAR_CROSS
        M = np.zeros((len(pairs), 8))
        for r, (a, b) in enumerate(pairs):
            M[r, a - 1], M[r, b - 1] = 1.0, -1.0
        return M
    if name == 'laplacian':
        M = np.zeros((8, 8))
        for c, nb in CH_ADJ.items():
            M[c - 1, c - 1] = 1.0
            for n in nb:
                M[c - 1, n - 1] = -1.0 / len(nb)
        return M
    raise ValueError(name)


def apply_derivation(X: np.ndarray, name: str) -> np.ndarray:
    M = derivation_matrix(name)
    return np.einsum('ij,njt->nit', M, X)


# ==================== CSP ====================

def _cov(x: np.ndarray) -> np.ndarray:
    """시행 하나의 정규화 공분산. trace로 나눠 진폭 스케일을 제거한다."""
    x = x - x.mean(axis=1, keepdims=True)
    c = x @ x.T
    tr = np.trace(c)
    return c / tr if tr > 0 else c


def csp_fit(X: np.ndarray, y: np.ndarray, n_filt=None):
    """이진 CSP 공간필터. C1 w = λ (C1+C2) w 의 양끝 고유벡터를 쓴다.

    유도에 따라 채널 수가 달라지므로(양극 대응쌍은 4채널) 필터 수를 채널 수에
    맞춰 줄인다. 4채널에서 필터 4개를 뽑으면 공간 전체를 쓰는 셈이라 과적합된다.
    """
    from scipy.linalg import eigh
    n_ch = X.shape[1]
    n_filt = min(N_FILT, max(1, n_ch // 4)) if n_filt is None else n_filt
    cls = np.unique(y)
    c1 = np.mean([_cov(x) for x in X[y == cls[0]]], axis=0)
    c2 = np.mean([_cov(x) for x in X[y == cls[1]]], axis=0)
    reg = 1e-6 * np.trace(c1 + c2) / c1.shape[0]      # 수치 안정용 릿지
    w, v = eigh(c1 + reg * np.eye(c1.shape[0]),
                c1 + c2 + 2 * reg * np.eye(c1.shape[0]))
    idx = np.argsort(w)
    return v[:, np.r_[idx[:n_filt], idx[-n_filt:]]].T   # (2*n_filt, n_ch)


def csp_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """log-variance 특징. 시행 내 총분산으로 나눠 다시 한번 크기를 지운다."""
    out = np.empty((len(X), W.shape[0]))
    for i, x in enumerate(X):
        z = W @ (x - x.mean(axis=1, keepdims=True))
        v = z.var(axis=1)
        out[i] = np.log(v / (v.sum() + 1e-12) + 1e-12)
    return out


# ==================== 데이터 ====================

def load_data():
    z = np.load(EPOCHS, allow_pickle=True)
    X = z['X']
    meta = pd.DataFrame({k: z[k] for k in z.files if k != 'X'})
    meta['condition'] = meta['session'].astype(str).str[0].str.lower()
    # 피험자(사람) 라벨과 코호트 플래그는 통계 단계 산출물에서 가져온다
    if POOLED.exists():
        p = pd.read_csv(POOLED, low_memory=False,
                        usecols=['subject', 'subject_id', 'visit_id', 'axis_a', 'axis_b'])
        p = p.drop_duplicates('subject').rename(columns={'subject': 'visit_key'})
        meta = meta.merge(p, left_on='subject', right_on='visit_key', how='left')
    meta['subject_id'] = meta['subject_id'].fillna(meta['subject'])
    meta['step'] = np.digitize(meta['load_pct'].astype(float), [34, 61, 84])
    keep = meta['axis_a'].fillna(False).astype(bool) if 'axis_a' in meta else np.ones(len(meta), bool)
    return X[np.asarray(keep)], meta[np.asarray(keep)].reset_index(drop=True)


# ==================== 평가 ====================

def loso_csp(X, meta, mask=None, seed=0):
    """피험자 단위 LOSO. CSP 필터도 학습셋에서만 적합한다(평가셋 누출 방지)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    m = np.ones(len(meta), bool) if mask is None else np.asarray(mask)
    Xs, ms = X[m], meta[m].reset_index(drop=True)
    y = (ms['condition'] == 'f').astype(int).values
    subs = ms['subject_id'].values
    rows = []
    for s in pd.unique(subs):
        tr, te = subs != s, subs == s
        if te.sum() < 4 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        W = csp_fit(Xs[tr], y[tr])
        clf = LDA().fit(csp_features(Xs[tr], W), y[tr])
        pr = clf.predict(csp_features(Xs[te], W))
        rows.append({'subject_id': s, 'n': int(te.sum()),
                     'acc': float((pr == y[te]).mean())})
    return pd.DataFrame(rows)


def loso_topo_baseline(meta, mask=None):
    """비교선: 시계열 없이 이벤트당 8채널 스칼라 지형만 쓰는 LDA.

    CSP를 도입한 값어치가 있는지 보려면, 원래 계획했던 '진폭 정규화 8채널
    벡터' 방식과 견줘야 한다. 이 베이스라인이 CSP와 대등하면 시계열은 불필요하다.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    ev = pd.read_csv(OUT_DIR / 'events.csv', low_memory=False)
    cols = [f'topo_ch{c}' for c in range(1, 9) if f'topo_ch{c}' in ev]
    ev = ev[ev['condition'].isin(['s', 'f'])].dropna(subset=cols)
    ev['step'] = np.digitize(ev['load_pct'], [34, 61, 84])
    if mask is not None:
        ev = ev[ev['step'].isin(mask)]
    y = (ev['condition'] == 'f').astype(int).values
    subs = ev['subject_id'].values
    rows = []
    for s in pd.unique(subs):
        tr, te = subs != s, subs == s
        if te.sum() < 4 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LDA().fit(ev.loc[tr, cols], y[tr])
        pr = clf.predict(ev.loc[te, cols])
        rows.append({'subject_id': s, 'n': int(te.sum()),
                     'acc': float((pr == y[te]).mean())})
    return pd.DataFrame(rows)


def permutation(X, meta, n_perm, seed=20260804):
    """피험자 내 라벨 셔플. 개인 수준 구조는 두고 방향-패턴 대응만 끊는다."""
    rng = np.random.default_rng(seed)
    obs = loso_csp(X, meta)['acc'].mean()
    null = []
    for i in range(n_perm):
        mm = meta.copy()
        mm['condition'] = mm.groupby('subject_id')['condition'].transform(
            lambda v: rng.permutation(v.values))
        r = loso_csp(X, mm)
        null.append(r['acc'].mean() if len(r) else np.nan)
        if (i + 1) % 20 == 0:
            print(f"  perm {i+1}/{n_perm} · null {np.nanmedian(null):.3f} "
                  f"(관측 {obs:.3f})", flush=True)
    null = np.array(null, dtype=float)
    null = null[np.isfinite(null)]
    return {'observed_acc': float(obs), 'null_mean': float(null.mean()),
            'null_sd': float(null.std(ddof=1)),
            'p_value': float((null >= obs).mean()), 'n_perm': len(null)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-perm', type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, meta = load_data()
    sf = meta['condition'].isin(['s', 'f']).values
    X, meta = X[sf], meta[sf].reset_index(drop=True)
    print(f"epoch {X.shape} · 피험자 {meta['subject_id'].nunique()}명 · "
          f"s {int((meta['condition']=='s').sum())} / f {int((meta['condition']=='f').sum())}")

    rows = []
    r = loso_csp(X, meta)
    rows.append({'analysis': 'CSP+LDA (all load steps)', 'n_subj': len(r),
                 'acc_mean': round(r['acc'].mean(), 3), 'acc_sd': round(r['acc'].std(), 3)})
    for st, lab in zip(range(4), ['~20%', '~50%', '~80%', '~100%']):
        rs = loso_csp(X, meta, mask=(meta['step'] == st).values)
        if len(rs):
            rows.append({'analysis': f'CSP+LDA (load step {lab})', 'n_subj': len(rs),
                         'acc_mean': round(rs['acc'].mean(), 3),
                         'acc_sd': round(rs['acc'].std(), 3)})
    b = loso_topo_baseline(meta)
    if len(b):
        rows.append({'analysis': 'baseline: scalar topography LDA', 'n_subj': len(b),
                     'acc_mean': round(b['acc'].mean(), 3),
                     'acc_sd': round(b['acc'].std(), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'direction_decoder.csv', index=False)
    r.to_csv(OUT_DIR / 'direction_per_subject.csv', index=False)
    print("\n=== 방향 디코더 (s vs f, LOSO) ===")
    print(res.to_string(index=False))
    print("  * 우연 수준 0.500")

    # 전체 데이터로 적합한 필터의 공간 가중치 — LMM·SHAP과 대조할 지형
    W = csp_fit(X, (meta['condition'] == 'f').astype(int).values)
    pd.DataFrame(W, columns=[f'ch{c}' for c in range(1, 9)],
                 index=[f'filter{i+1}' for i in range(W.shape[0])]
                 ).to_csv(OUT_DIR / 'csp_filters.csv')
    print("\n=== CSP 공간필터 가중치 ===")
    print(pd.DataFrame(W, columns=[f'ch{c}' for c in range(1, 9)],
                       index=[f'filter{i+1}' for i in range(W.shape[0])]).round(3).to_string())

    if a.n_perm:
        p = permutation(X, meta, a.n_perm)
        pd.DataFrame([p]).to_csv(OUT_DIR / 'direction_permutation.csv', index=False)
        print(f"\n=== Permutation ===\n  관측 {p['observed_acc']:.3f} · 귀무 "
              f"{p['null_mean']:.3f} ± {p['null_sd']:.3f} · p={p['p_value']:.4f}")
    print(f"\n산출물: {OUT_DIR}/")


if __name__ == '__main__':
    main()
