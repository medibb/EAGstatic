#!/usr/bin/env python3
"""EEGNet(compact CNN) LOSO — 원파형에서 직접 학습하면 특징기반 모델을 넘는가.

두 가지를 묻는다.

  1. **방향(s vs f)**: CSP+LDA가 0.551에서 멈춘 것이 '신호에 정보가 없어서'인지
     '선형 공분산 방법의 한계'인지. CSP는 시행별 공분산 하나로 시행을 요약하므로
     시간 구조를 통째로 버린다. CNN은 그 시간 구조를 볼 수 있다.
  2. **부하(load_pct 회귀)**: parameter_extractor가 뽑은 스칼라 특징(진폭·기울기·
     오프셋·지연)이 **충분통계량**인가. 원파형 end-to-end가 같은 성능에 그치면
     특징 집합이 충분하다는 적극적 근거가 된다. 넘어서면 파형 모양에 아직
     쓰지 않은 정보가 있다는 뜻이다.

설계 원칙은 ml_decoder/ml_direction과 동일하게 유지한다.

  * **LOSO**: 같은 사람이 학습과 평가에 동시에 들어가지 않는다. 3장 CNN이
    pooled event 위에서 무작위 분할을 쓴 것이 이 장에서 재현되지 않도록 한다.
  * **조기종료용 내부 검증셋도 피험자 단위로 뗀다.** 학습 피험자 40명 중 일부를
    통째로 빼서 쓴다. 시행 단위로 떼면 조기종료 시점 자체가 누출된다.
  * **시행별 정규화로 개인 이득(gain)을 제거한다.** 전극 임피던스와 무릎 형상이
    사람마다 진폭 스케일을 정한다. 그대로 두면 CNN이 '누구인가'를 먼저 배우고
    그 사람의 라벨 분포를 따라간다(= 소표본에서 가장 흔한 누출 경로).
    방향 과제는 CSP의 trace 정규화와 같은 취지로 시행 전체를 하나의 스칼라로
    나눠 공간 패턴만 남긴다.

아키텍처는 3장의 10층 2D CNN이 아니라 EEGNet 계열 compact 모델이다. 독립 단위가
41명뿐이라 파라미터를 수천 개 수준으로 묶어야 한다(3장 아키텍처는 입력 형태도
분할 전제도 다르므로 그대로 옮기지 않는다).

사용:
  python3 ml_eegnet.py --task direction
  python3 ml_eegnet.py --task load
  python3 ml_eegnet.py --task direction --n-perm 20
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

OUT_DIR = Path('result/dl')
FS = 25.0                 # epoch_extractor 리샘플 주파수 (125 sample / 5 s)
SEED = 20260815


# ==================== 모델 ====================

def build_eegnet(n_ch, n_time, n_out, F1=8, D=2, F2=16, drop=0.5, kern=13):
    """EEGNet-compact.

    depthwise-separable 구조라 파라미터가 수천 개에 그친다. 세 단계로 읽는다.

      1. temporal conv   (1, kern) : 채널을 건드리지 않고 시간 필터 F1개 학습.
                                     대역통과 필터뱅크를 데이터에서 배우는 셈.
      2. depthwise conv  (n_ch, 1) : 각 시간필터마다 공간 필터 D개. CSP가 하던
                                     '채널 가중 조합'에 해당하되 필터뱅크별로 따로.
      3. separable conv            : 시간 방향 요약 + 특징 혼합.

    kern=13은 25 Hz에서 약 0.5 s. 관심 대역(0.3~1.5 Hz)의 반주기 규모를 덮는다.
    """
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                # groups=F1 → 시간필터별로 독립적인 공간필터. 채널 혼합을 여기서만 한다.
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 2)), nn.Dropout(drop))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 7), padding=(0, 3),
                          groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.head = nn.Sequential(nn.Flatten(),
                                      nn.Linear(F2 * (n_time // 8), n_out))

        def forward(self, x):
            return self.head(self.block2(self.block1(x)))

    return EEGNet()


# ==================== 데이터 ====================

def normalise(X, mode):
    """시행별 정규화. 개인 이득을 제거하는 단계이자 누출 방지 장치.

    trial  : 시행 전체를 하나의 스칼라로 나눈다. 채널 간 상대 크기가 보존되므로
             공간 패턴은 살아 있고 전체 진폭만 사라진다(CSP trace 정규화와 동형).
    channel: 채널별로 각각 z화. 공간 패턴까지 지우므로 방향 과제에는 부적절하고
             '시간 모양만으로 되는가'를 볼 때만 쓴다.
    none   : 부하 회귀용. 진폭이 곧 타깃 정보라 지우면 안 된다.
    """
    X = np.asarray(X, np.float32)
    if mode == 'none':
        return X
    if mode == 'channel':
        mu = X.mean(2, keepdims=True)
        sd = X.std(2, keepdims=True) + 1e-8
        return (X - mu) / sd
    mu = X.mean((1, 2), keepdims=True)
    sd = X.std((1, 2), keepdims=True) + 1e-8
    return (X - mu) / sd


def get_data(task):
    """ml_direction.load_data()를 그대로 재사용해 코호트·라벨을 일치시킨다.

    비교 대상(CSP+LDA 0.551)과 같은 표본이 아니면 성능 차이를 모델 탓으로
    돌릴 수 없다.
    """
    import ml_direction as md
    X, meta = md.load_data()
    if task == 'direction':
        m = meta['condition'].isin(['s', 'f']).values
        X, meta = X[m], meta[m].reset_index(drop=True)
        y = (meta['condition'] == 'f').astype(np.float32).values
        return normalise(X, 'trial'), y, meta, 'clf'
    m = meta['load_pct'].notna().values
    X, meta = X[m], meta[m].reset_index(drop=True)
    return normalise(X, 'none'), meta['load_pct'].astype(np.float32).values, meta, 'reg'


# ==================== 학습 ====================

def fit_fold(Xtr, ytr, Xva, yva, Xte, kind, max_ep=60, patience=10, seed=0):
    """한 fold 학습. 내부 검증셋 손실로 조기종료하고 최선 가중치를 복원한다."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = 'cpu'
    n_out = 1
    net = build_eegnet(Xtr.shape[1], Xtr.shape[2], n_out).to(dev)
    lossf = nn.BCEWithLogitsLoss() if kind == 'clf' else nn.SmoothL1Loss()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)

    # 회귀 타깃은 학습셋 통계로만 표준화한다(평가셋 통계를 쓰면 누출).
    if kind == 'reg':
        mu, sd = float(ytr.mean()), float(ytr.std() + 1e-8)
        ytr_, yva_ = (ytr - mu) / sd, (yva - mu) / sd
    else:
        mu, sd, ytr_, yva_ = 0.0, 1.0, ytr, yva

    T = lambda a: torch.tensor(np.asarray(a, np.float32))
    xtr, xva, xte = (T(a)[:, None] for a in (Xtr, Xva, Xte))
    ttr, tva = T(ytr_)[:, None], T(yva_)[:, None]

    n, bs = len(xtr), 128
    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(max_ep):
        net.train()
        for idx in np.array_split(rng.permutation(n), max(1, n // bs)):
            opt.zero_grad()
            lossf(net(xtr[idx]), ttr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(xva), tva))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(xte).numpy().ravel()
    n_par = sum(p.numel() for p in net.parameters())
    return (out if kind == 'clf' else out * sd + mu), ep + 1, n_par


def run_loso(X, y, meta, kind, n_val=5, seed=SEED, verbose=True):
    """피험자 단위 LOSO. 내부 검증셋도 피험자 통째로 뗀다."""
    subs = meta['subject_id'].values
    uniq = pd.unique(subs)
    rng = np.random.default_rng(seed)
    rows, preds = [], np.full(len(y), np.nan)
    for i, s in enumerate(uniq):
        te = subs == s
        tr_subs = np.array([u for u in uniq if u != s])
        if te.sum() < 4 or len(tr_subs) < n_val + 2:
            continue
        val_subs = rng.choice(tr_subs, size=n_val, replace=False)
        va = np.isin(subs, val_subs)
        tr = (~te) & (~va)
        if kind == 'clf' and (len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2
                              or len(np.unique(y[va])) < 2):
            continue
        out, n_ep, n_par = fit_fold(X[tr], y[tr], X[va], y[va], X[te], kind,
                                    seed=seed + i)
        preds[te] = out
        if kind == 'clf':
            acc = float(((out > 0).astype(int) == y[te]).mean())
            rows.append({'subject_id': s, 'n': int(te.sum()), 'acc': acc,
                         'n_epoch_run': n_ep})
        else:
            err = np.abs(out - y[te])
            rows.append({'subject_id': s, 'n': int(te.sum()),
                         'mae': float(err.mean()), 'n_epoch_run': n_ep})
        if verbose:
            k = 'acc' if kind == 'clf' else 'mae'
            print(f"  [{i + 1}/{len(uniq)}] {s}: n={te.sum():4d} "
                  f"{k}={rows[-1][k]:.3f} (ep {n_ep}, par {n_par:,})", flush=True)
    return pd.DataFrame(rows), preds


# ==================== 요약 ====================

def summarise(per, preds, y, meta, kind):
    ok = ~np.isnan(preds)
    out = {'n_subject': int(len(per)), 'n_event': int(ok.sum())}
    if kind == 'clf':
        out['acc_pooled'] = float(((preds[ok] > 0).astype(int) == y[ok]).mean())
        out['acc_subject_mean'] = float(per['acc'].mean())
        out['acc_subject_sd'] = float(per['acc'].std())
        # 독립 단위는 피험자다. 0.5 대비 1표본 t로 검정한다.
        from scipy import stats as sps
        t, p = sps.ttest_1samp(per['acc'].values, 0.5)
        out['t_vs_chance'], out['p_vs_chance'] = float(t), float(p)
        out['majority_baseline'] = float(max(y[ok].mean(), 1 - y[ok].mean()))
    else:
        e = np.abs(preds[ok] - y[ok])
        ss = ((y[ok] - preds[ok]) ** 2).sum()
        out['mae_pooled'] = float(e.mean())
        out['mae_subject_mean'] = float(per['mae'].mean())
        out['r2_pooled'] = float(1 - ss / ((y[ok] - y[ok].mean()) ** 2).sum())
        out['mae_mean_baseline'] = float(np.abs(y[ok] - y[ok].mean()).mean())
    return out


def permutation(X, y, meta, kind, n_perm, seed=SEED):
    """피험자 내 라벨 셔플. 전체를 섞으면 피험자 간 평균 차이만 깨져 느슨해진다."""
    rng = np.random.default_rng(seed)
    subs = meta['subject_id'].values
    vals = []
    for k in range(n_perm):
        yp = y.copy()
        for s in pd.unique(subs):
            m = subs == s
            yp[m] = rng.permutation(yp[m])
        per, pr = run_loso(X, yp, meta, kind, seed=seed + 1000 + k, verbose=False)
        v = per['acc'].mean() if kind == 'clf' else per['mae'].mean()
        vals.append(float(v))
        print(f"  perm {k + 1}/{n_perm}: {v:.4f}", flush=True)
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=['direction', 'load'], default='direction')
    ap.add_argument('--n-perm', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suf = f'_{a.tag}' if a.tag else ''

    X, y, meta, kind = get_data(a.task)
    print(f"task={a.task} kind={kind}  X={X.shape}  "
          f"피험자 {meta['subject_id'].nunique()}명 · 이벤트 {len(y)}")
    if kind == 'clf':
        print(f"  클래스 균형: f={y.mean():.3f}")

    per, preds = run_loso(X, y, meta, kind)
    s = summarise(per, preds, y, meta, kind)

    if a.n_perm:
        print(f"\n--- permutation {a.n_perm}회 ---")
        null = permutation(X, y, meta, kind, a.n_perm)
        obs = s['acc_subject_mean'] if kind == 'clf' else s['mae_subject_mean']
        s['null_mean'], s['null_sd'] = float(null.mean()), float(null.std())
        s['p_perm'] = float(((null >= obs).sum() + 1) / (len(null) + 1)) if kind == 'clf' \
            else float(((null <= obs).sum() + 1) / (len(null) + 1))
        np.savetxt(OUT_DIR / f'eegnet_{a.task}_null{suf}.txt', null)

    per.to_csv(OUT_DIR / f'eegnet_{a.task}_per_subject{suf}.csv', index=False)
    np.save(OUT_DIR / f'eegnet_{a.task}_preds{suf}.npy', preds)
    (OUT_DIR / f'eegnet_{a.task}_summary{suf}.json').write_text(
        json.dumps(s, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== 요약 ===')
    for k, v in s.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
