#!/usr/bin/env python3
"""Filter-bank CSP — 부하와 방향이 서로 다른 주파수 대역에 실려 있는지 검정한다.

관찰: 고역통과를 올릴수록 방향 디코딩은 좋아지고(0.519→0.547→0.553) 부하 반응
진폭은 감쇠했다(187→101→20 µV). 두 정보가 다른 대역에 있다는 가설이 선다.

여기서는 대역을 나눠 각각 (1) 방향 디코딩 정확도와 (2) 부하-반응 상관을 재어
가설을 직접 검정하고, 마지막에 전 대역 특징을 합친 FBCSP를 돌린다.

창이 5초라 0.2 Hz 미만은 분해되지 않으므로(주기 5초 이상) 그 이하는 나누지 않는다.
epoch이 이미 0.10 Hz 고역통과를 거쳤으므로 재추출 없이 대역만 나눈다.
"""
import os, sys, warnings
os.chdir('/workspace/research/EAGstatic'); sys.path.insert(0, '/workspace/research/EAGstatic')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scipy.stats as sps
from scipy.signal import butter, filtfilt
import ml_direction as MD

FS = 25.0
BANDS = [(0.10, 0.30), (0.30, 0.70), (0.70, 1.50), (1.50, 3.00), (3.00, 5.00)]


def bandpass(X, lo, hi, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), min(hi, fs / 2 - 0.01) / (fs / 2)], btype='band')
    return filtfilt(b, a, X, axis=-1, padlen=min(X.shape[-1] - 1, 40))


def main():
    X, meta = MD.load_data()
    sf = meta['condition'].isin(['s', 'f']).values
    Xs, ms = X[sf], meta[sf].reset_index(drop=True)
    print(f"epoch {Xs.shape} · 피험자 {ms['subject_id'].nunique()}명", flush=True)

    rows, feats = [], []
    for lo, hi in BANDS:
        Xb = bandpass(Xs, lo, hi)
        # (1) 방향 디코딩
        r = MD.loso_csp(Xb, ms); a = r['acc'].values
        t, p = sps.ttest_1samp(a, 0.5); se = a.std(ddof=1) / np.sqrt(len(a))
        # (2) 부하 정보: 대역별 반응 크기와 실측 부하의 상관 (전 조건 사용)
        Xall = bandpass(X, lo, hi)
        mag = np.abs(Xall).mean(axis=(1, 2))
        rho = sps.spearmanr(meta['load_pct'].astype(float), mag).statistic
        rows.append({'band': f'{lo}-{hi}Hz', 'dir_acc': round(a.mean(), 4),
                     'sd': round(a.std(ddof=1), 4),
                     'CI': f"{a.mean()-1.96*se:.3f}~{a.mean()+1.96*se:.3f}",
                     't': round(float(t), 2), 'p': f'{p:.4f}',
                     'above': f"{(a>0.5).sum()}/{len(a)}",
                     'rho_load': round(float(rho), 3),
                     'mag_med': round(float(np.median(mag)), 1)})
        print(rows[-1], flush=True)
        feats.append(Xb)

    df = pd.DataFrame(rows)
    df.to_csv('result/ml/fbcsp_bands.csv', index=False)

    # FBCSP: 대역별 CSP 특징을 이어붙여 하나의 LDA로
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    y = (ms['condition'] == 'f').astype(int).values
    subs = ms['subject_id'].values
    accs = []
    for s in pd.unique(subs):
        tr, te = subs != s, subs == s
        if te.sum() < 4 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Ftr, Fte = [], []
        for Xb in feats:
            W = MD.csp_fit(Xb[tr], y[tr])
            Ftr.append(MD.csp_features(Xb[tr], W)); Fte.append(MD.csp_features(Xb[te], W))
        clf = LDA(solver='lsqr', shrinkage='auto').fit(np.hstack(Ftr), y[tr])
        accs.append(float((clf.predict(np.hstack(Fte)) == y[te]).mean()))
    accs = np.array(accs)
    t, p = sps.ttest_1samp(accs, 0.5); se = accs.std(ddof=1) / np.sqrt(len(accs))
    fb = {'method': 'FBCSP (5 bands, shrinkage LDA)', 'n_subj': len(accs),
          'acc': round(accs.mean(), 4), 'sd': round(accs.std(ddof=1), 4),
          'CI': f"{accs.mean()-1.96*se:.3f}~{accs.mean()+1.96*se:.3f}",
          't': round(float(t), 2), 'p': f'{p:.5f}',
          'above': f"{(accs>0.5).sum()}/{len(accs)}"}
    pd.DataFrame([fb]).to_csv('result/ml/fbcsp_combined.csv', index=False)

    print("\n=== 대역별 (방향 디코딩 · 부하 상관) ===")
    print(df.to_string(index=False))
    print("\n=== FBCSP 통합 ===")
    print(pd.DataFrame([fb]).to_string(index=False))
    print("  * 광대역 CSP 기준선: 0.547 (0.525~0.569)")


if __name__ == '__main__':
    main()
