#!/usr/bin/env python3
"""목발(c) vs 비목발(s, f) 분류 — CNN·LSTM·Transformer 혼합 모델, 피험자 LOSO.

질문: 검사측 무릎 EAG가 **하중을 어디로 보내는가**(반대측 상지 목발 vs 반대측 다리)를
구분하는가. 세 조건은 검사측에 동일한 graded PWB를 적용하므로, 구분이 된다면
부하량이 아닌 다른 것(자세 안정화 전략, plateau 변동, 감쇠 역학)이 신호에 있다는 뜻이다.

사전 경고: 방향(s vs f) 디코더는 chance 근처였고(CSP 0.547, EEGNet 0.51), 부하 회귀
transfer에서 s+f→c MAE 14.97 ≈ c→c 14.94 였다. 양성이 나오면 교란(부하 분포·cycle
길이·take 순서)을 먼저 의심한다. 그래서 학습 없이 예측값만으로 하는 매칭 재평가를
같이 낸다.

설계 원칙은 ml_eegnet과 동일: LOSO, 내부 검증셋도 피험자 단위, 시행별 정규화로
개인 이득 제거, 순열은 피험자 내에서 **세션 단위**로 라벨을 섞는다(이벤트 단위로
섞으면 같은 세션의 이벤트가 양쪽 라벨에 흩어져 null이 너무 쉬워진다).

창 두 종류를 같은 fold·seed로 돌린다(사전등록, 창 비교 자체가 결과):
  event : epochs.npz  (이벤트 -1~+4 s, 125 샘플)      — 전이 반응
  cycle : cycles.npz  (onset-2 s ~ offset+3 s, 가변, pad 750) — plateau·감쇠 포함

모델(파라미터 2만~4만, 41명 독립 단위에 맞춘 규모):
  cnn      : EEGNet 프론트 + masked mean pool
  cnn_lstm : + BiLSTM(16)
  cnn_tf   : + Transformer(d=32, 2층) + attention pool
  hybrid   : + BiLSTM(16) → Transformer(d=32, 2층) + attention pool
  csp      : CSP+LDA 베이스라인 (event 창만)
  lgb      : events.csv 스칼라 특징 LightGBM 베이스라인 (창 무관)
  dur      : cycle 길이만 쓰는 로지스틱 (cycle 창 교란 베이스라인)

사용:
  python3 ml_crutch.py --window event --model hybrid
  python3 ml_crutch.py --window cycle --model all --n-perm 100
  python3 ml_crutch.py --window event --model cnn --subjects 6     # smoke test
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

OUT_DIR = Path('result/ml')
POOLED = Path('result/stats/grf_eag_pooled.csv')
DEVICE = 'cpu'                     # --device cuda 로 바꾼다(3090). 결과 파일 형식은 동일.
EVENTS = OUT_DIR / 'events.csv'
FS = 25.0
SEED = 20260919
DL_MODELS = ['cnn', 'cnn_lstm', 'cnn_tf', 'hybrid']
STEP_EDGES = [34, 61, 84]          # ml_direction과 동일한 부하 단계 경계


# ==================== 데이터 ====================

def load_window(window):
    """epochs.npz / cycles.npz → (X, length, y, meta). axis_b(s,f,c 모두 ≥3 take) 코호트."""
    path = OUT_DIR / ('epochs.npz' if window == 'event' else 'cycles.npz')
    z = np.load(path, allow_pickle=True)
    X = z['X'].astype(np.float32)
    meta = pd.DataFrame({k: z[k] for k in z.files if k != 'X'})
    meta['condition'] = meta['session'].astype(str).str[0].str.lower()
    meta['take'] = meta['session'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    if POOLED.exists():
        p = pd.read_csv(POOLED, low_memory=False,
                        usecols=['subject', 'subject_id', 'visit_id', 'axis_a', 'axis_b'])
        p = p.drop_duplicates('subject').rename(columns={'subject': 'visit_key'})
        meta = meta.merge(p, left_on='subject', right_on='visit_key', how='left')
    meta['subject_id'] = meta['subject_id'].fillna(meta['subject'])
    keep = meta['axis_b'].fillna(False).astype(bool).values if 'axis_b' in meta \
        else np.ones(len(meta), bool)
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    if 'length' not in meta:
        meta['length'] = X.shape[2]
    length = meta['length'].astype(int).values
    meta['step'] = np.digitize(meta['load_pct'].astype(float), STEP_EDGES)
    y = (meta['condition'] == 'c').astype(np.float32).values
    return normalise_masked(X, length), length, y, meta


def normalise_masked(X, length):
    """시행별 정규화(유효 구간만). 개인 이득 제거 + 누출 방지. 패딩은 0으로 유지."""
    out = np.zeros_like(X, dtype=np.float32)
    for i, n in enumerate(length):
        x = X[i, :, :n]
        out[i, :, :n] = (x - x.mean()) / (x.std() + 1e-8)
    return out


# ==================== 모델 ====================

def build_model(kind, n_ch, n_time, F1=8, D=2, F2=16, kern=13, h_lstm=16, d_tf=32,
                n_layers=2, drop=0.5, drop_tf=0.3):
    import torch
    import torch.nn as nn

    class Front(nn.Module):
        """EEGNet block1+block2. (B,1,C,T) → (B,F2,T//8). 시간필터→공간필터→요약."""
        def __init__(self):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(drop))
            self.b2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 7), padding=(0, 3), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))

        def forward(self, x):
            return self.b2(self.b1(x)).squeeze(2).transpose(1, 2)   # (B, T', F2)

    def sinusoid(n, d):
        pe = torch.zeros(n, d)
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = Front()
            t_ds = n_time // 8
            dim = F2
            if kind in ('cnn_lstm', 'hybrid'):
                self.lstm = nn.LSTM(dim, h_lstm, batch_first=True, bidirectional=True)
                dim = 2 * h_lstm
            if kind in ('cnn_tf', 'hybrid'):
                self.proj = nn.Linear(dim, d_tf)
                self.register_buffer('pe', sinusoid(t_ds, d_tf))
                layer = nn.TransformerEncoderLayer(d_tf, 4, 2 * d_tf, drop_tf,
                                                   batch_first=True)
                self.tf = nn.TransformerEncoder(layer, n_layers)
                self.att = nn.Linear(d_tf, 1)
                dim = d_tf
            self.head = nn.Linear(dim, 1)

        def forward(self, x, len_ds):
            h = self.front(x)                                   # (B, T', F2)
            B, T, _ = h.shape
            ar = torch.arange(T, device=h.device)[None, :]
            valid = ar < len_ds[:, None]                        # (B, T') True=유효
            if hasattr(self, 'lstm'):
                packed = nn.utils.rnn.pack_padded_sequence(
                    h, len_ds.clamp(max=T).cpu(), batch_first=True, enforce_sorted=False)
                h, _ = self.lstm(packed)
                h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=T)
            if hasattr(self, 'tf'):
                h = self.proj(h) + self.pe[:T][None]
                h = self.tf(h, src_key_padding_mask=~valid)
                s = self.att(h).squeeze(-1).masked_fill(~valid, -1e4)
                w = torch.softmax(s, 1)[:, :, None]
                pooled = (h * w).sum(1)
            else:
                m = valid[:, :, None].float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(pooled)

    return Net()


# ==================== 학습 ====================

def augment(x, rng):
    """학습 배치 증강. 진폭 스케일·시간 이동(0 채움)·채널 드롭·잡음. (B,1,C,T)"""
    import torch
    B, _, C, T = x.shape
    x = x * torch.tensor(rng.uniform(0.8, 1.2, (B, 1, 1, 1)), dtype=x.dtype, device=x.device)
    out = torch.zeros_like(x)
    for i in range(B):
        s = int(rng.integers(-5, 6))
        if s >= 0:
            out[i, :, :, s:] = x[i, :, :, :T - s]
        else:
            out[i, :, :, :T + s] = x[i, :, :, -s:]
    drop = rng.random(B) < 0.3
    ch = rng.integers(0, C, B)
    for i in np.where(drop)[0]:
        out[i, :, ch[i], :] = 0
    return out + 0.05 * torch.randn_like(out)


def fit_fold(kind, Xtr, ltr, ytr, Xva, lva, yva, Xte, lte, aug=True,
             max_ep=60, patience=10, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = DEVICE
    net = build_model(kind, Xtr.shape[1], Xtr.shape[2]).to(dev)
    pos = float(ytr.mean())
    # 클래스비 32:68 → 양성 가중으로 balanced BCE
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3), device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
    L = lambda a: torch.tensor(np.maximum(1, np.asarray(a) // 8), dtype=torch.long, device=dev)
    xtr, xva, xte = (T(a)[:, None] for a in (Xtr, Xva, Xte))
    ltr_, lva_, lte_ = L(ltr), L(lva), L(lte)
    ttr, tva = T(ytr)[:, None], T(yva)[:, None]

    n, bs = len(xtr), 128
    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(max_ep):
        net.train()
        for idx in np.array_split(rng.permutation(n), max(1, n // bs)):
            xb = augment(xtr[idx], rng) if aug else xtr[idx]
            opt.zero_grad()
            lossf(net(xb, ltr_[idx]), ttr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(xva, lva_), tva))
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
        out = net(xte, lte_).cpu().numpy().ravel()
    return out, ep + 1, sum(p.numel() for p in net.parameters())


# ==================== 베이스라인 ====================

def fit_csp(Xtr, ytr, Xte):
    import ml_direction as md
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    W = md.csp_fit(Xtr, ytr.astype(int))
    # 클래스비 32:68이라 기본 prior로는 다수 클래스만 찍어 balanced acc가 0.5에 붙는다.
    clf = LDA(priors=[0.5, 0.5]).fit(md.csp_features(Xtr, W), ytr.astype(int))
    return clf.decision_function(md.csp_features(Xte, W))


def fit_lgb(Ftr, ytr, Fte, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                           class_weight='balanced', verbose=-1)
    m.fit(Ftr, ytr.astype(int))
    p = m.predict_proba(Fte)[:, 1]
    return np.log(p + 1e-6) - np.log(1 - p + 1e-6)


def fit_logit(Ftr, ytr, Fte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    m = LogisticRegression(class_weight='balanced', max_iter=500).fit(sc.transform(Ftr), ytr.astype(int))
    return m.decision_function(sc.transform(Fte))


def lgb_features(meta):
    """events.csv 스칼라 특징을 붙인다. load_pct 제외.

    event 창: (visit, session, trans_id)로 1:1.
    cycle 창: trans_id가 없으므로 epochs.npz의 trans_id→cycle_id 대응을 빌려 이벤트 특징을
              cycle 단위(on/off 2개)로 평균한 뒤 (visit, session, cycle_id)로 붙인다.
    """
    ev = pd.read_csv(EVENTS, low_memory=False)
    drop = {'subject_id', 'visit_id', 'session', 'condition', 'trans_id', 'load_pct',
            'eag_direction', 'is_rise'}
    cols = [c for c in ev.columns if c not in drop and ev[c].dtype != object]
    if 'trans_id' in meta:
        key = ['visit_id', 'session', 'trans_id']
        m = meta[['subject', 'session', 'trans_id']].rename(columns={'subject': 'visit_id'})
        F = m.merge(ev[key + cols], on=key, how='left')[cols].values.astype(np.float32)
        return F, cols
    z = np.load(OUT_DIR / 'epochs.npz', allow_pickle=True)
    link = pd.DataFrame({'visit_id': z['subject'], 'session': z['session'],
                         'trans_id': z['trans_id'], 'cycle_id': z['cycle_id']}).drop_duplicates()
    ev = ev.merge(link, on=['visit_id', 'session', 'trans_id'], how='inner')
    agg = ev.groupby(['visit_id', 'session', 'cycle_id'])[cols].mean().reset_index()
    m = meta[['subject', 'session', 'cycle_id']].rename(columns={'subject': 'visit_id'})
    F = m.merge(agg, on=['visit_id', 'session', 'cycle_id'], how='left')[cols].values.astype(np.float32)
    return F, cols


# ==================== LOSO ====================

def run_loso(kind, X, length, y, meta, F=None, n_val=5, seed=SEED, aug=True, verbose=True):
    subs = meta['subject_id'].values
    uniq = pd.unique(subs)
    rng = np.random.default_rng(seed)
    rows, preds = [], np.full(len(y), np.nan)
    n_par = 0
    for i, s in enumerate(uniq):
        te = subs == s
        tr_subs = np.array([u for u in uniq if u != s])
        if te.sum() < 4 or len(np.unique(y[te])) < 2:
            continue
        nv = min(n_val, max(1, len(tr_subs) // 4))     # 소수 피험자 smoke test 보호
        val_subs = rng.choice(tr_subs, size=nv, replace=False)
        va = np.isin(subs, val_subs)
        tr = (~te) & (~va)
        n_ep = 0
        if kind in DL_MODELS:
            out, n_ep, n_par = fit_fold(kind, X[tr], length[tr], y[tr], X[va], length[va],
                                        y[va], X[te], length[te], aug=aug, seed=seed + i)
        elif kind == 'csp':
            tr = ~te
            out = fit_csp(X[tr], y[tr], X[te])
        elif kind == 'lgb':
            # events.csv는 exclusions 반영본이라 epochs의 약 21%에 특징이 없다(NaN).
            # LightGBM은 NaN을 자체 처리하므로 행을 버리지 않고 그대로 넘긴다.
            tr = ~te
            out = fit_lgb(F[tr], y[tr], F[te], seed=seed + i)
        elif kind == 'dur':
            tr = ~te
            out = fit_logit(F[tr], y[tr], F[te])
        else:
            raise ValueError(kind)
        preds[te] = out
        rows.append(dict(subject_id=s, n=int(te.sum()), n_epoch_run=n_ep,
                         **subject_metrics(out, y[te])))
        if verbose:
            r = rows[-1]
            print(f"  [{i + 1}/{len(uniq)}] {s}: n={r['n']:4d} bacc={r['bacc']:.3f} "
                  f"auc={r['auc']:.3f} (ep {n_ep}, par {n_par:,})", flush=True)
    return pd.DataFrame(rows), preds, n_par


def subject_metrics(out, yt):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    pr = (out > 0).astype(int)
    return {'bacc': float(balanced_accuracy_score(yt, pr)),
            'acc': float((pr == yt).mean()),
            'auc': float(roc_auc_score(yt, out)) if len(np.unique(yt)) == 2 else np.nan}


# ==================== 요약·매칭 재평가 ====================

def matched_bacc(preds, y, meta, by):
    """예측값만으로 하는 교란 매칭 재평가: 피험자 내 `by` bin 중 양쪽 클래스가 모두 있는
    bin만 남겨 balanced acc를 다시 잰다. 정확도가 여기서 사라지면 모델은 `by`를 배운 것."""
    ok = ~np.isnan(preds)
    d = meta.loc[ok, ['subject_id', by]].copy()
    d['y'], d['p'] = y[ok], (preds[ok] > 0).astype(int)
    both = d.groupby(['subject_id', by])['y'].transform(lambda s: s.nunique() == 2)
    d = d[both]
    if d.empty:
        return {'n': 0}
    from sklearn.metrics import balanced_accuracy_score
    per = d.groupby('subject_id').apply(lambda g: balanced_accuracy_score(g['y'], g['p']))
    return {'n': int(len(d)), 'n_subject': int(len(per)),
            'bacc_subject_mean': float(per.mean()), 'bacc_subject_sd': float(per.std())}


def summarise(per, preds, y, meta, n_par, window):
    from scipy import stats as sps
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(preds)
    out = {'window': window, 'n_subject': int(len(per)), 'n_sample': int(ok.sum()),
           'n_param': int(n_par),
           'bacc_subject_mean': float(per['bacc'].mean()),
           'bacc_subject_sd': float(per['bacc'].std()),
           'auc_subject_mean': float(per['auc'].mean()),
           'acc_pooled': float(((preds[ok] > 0).astype(int) == y[ok]).mean()),
           'auc_pooled': float(roc_auc_score(y[ok], preds[ok])),
           'majority_baseline': float(max(y[ok].mean(), 1 - y[ok].mean())),
           'n_epoch_mean': float(per['n_epoch_run'].mean())}
    t, p = sps.ttest_1samp(per['bacc'].values, 0.5)
    out['t_vs_chance'], out['p_vs_chance'] = float(t), float(p)
    out['matched_load_step'] = matched_bacc(preds, y, meta, 'step')
    if 'duration' in meta:
        meta = meta.copy()
        meta['dur_bin'] = pd.qcut(meta['duration'].astype(float), 4, labels=False, duplicates='drop')
        out['matched_duration'] = matched_bacc(preds, y, meta, 'dur_bin')
    return out


def permutation(kind, X, length, y, meta, F, n_perm, seed=SEED, offset=0, partial=None):
    """피험자 내 세션 단위 라벨 셔플. 세션의 이벤트는 같은 라벨을 유지한다.

    offset: 이미 돌린 순열 수. k=offset부터 시작해 이전 실행과 다른 셔플·fold seed를 쓰므로
            null 파일을 이어 붙여(pooled) 순열 수를 늘릴 수 있다.
    partial: 경로를 주면 순열 한 회가 끝날 때마다 값을 append 한다(회당 1 h 넘는 DL 순열이
             중간에 죽어도 그때까지의 null이 남는다). 재시작 시 그 파일 줄 수를 offset에 더할 것.
    """
    rng = np.random.default_rng(seed + offset)
    subs, sess = meta['subject_id'].values, meta['session'].values
    vals = []
    for k in range(offset, offset + n_perm):
        yp = y.copy()
        for s in pd.unique(subs):
            m = subs == s
            u = pd.unique(sess[m])
            lab = {ss: y[m & (sess == ss)][0] for ss in u}
            new = dict(zip(u, rng.permutation([lab[ss] for ss in u])))
            for ss in u:
                yp[m & (sess == ss)] = new[ss]
        per, _, _ = run_loso(kind, X, length, yp, meta, F, seed=seed + 1000 + k, verbose=False)
        vals.append(float(per['bacc'].mean()))
        print(f"  perm {k + 1}/{offset + n_perm}: {vals[-1]:.4f}", flush=True)
        if partial is not None:
            with open(partial, 'a') as fh:
                fh.write(f"{vals[-1]:.6f}\n")
    return np.array(vals)


# ==================== main ====================

def run_one(kind, window, X, length, y, meta, n_perm, tag, aug, perm_offset=0):
    F = None
    if kind == 'lgb':
        F, cols = lgb_features(meta)
        print(f"  lgb 특징 {len(cols)}개, 결측행 {int(np.isnan(F).any(1).sum())}")
    if kind == 'dur':
        F = meta[['duration']].astype(float).values
    if kind == 'csp' and window != 'event':
        print("  csp 베이스라인은 event 창에서만 정의됨 → skip"); return
    if kind == 'dur' and 'duration' not in meta:
        print("  dur 베이스라인은 cycle 창에서만 → skip"); return

    print(f"\n=== window={window} model={kind} ===", flush=True)
    # seed는 호출 시점의 전역(SEED)을 쓴다(--seed 반영). 기본인자로 두면 정의 시점에 고정된다.
    per, preds, n_par = run_loso(kind, X, length, y, meta, F, aug=aug, seed=SEED)
    s = summarise(per, preds, y, meta, n_par, window)
    s['model'], s['seed'], s['device'] = kind, int(SEED), DEVICE
    suf = f'_{tag}' if tag else ''
    stem = OUT_DIR / f'crutch_{window}_{kind}{suf}'
    if n_perm:
        print(f"--- permutation {n_perm}회 ---")
        null = permutation(kind, X, length, y, meta, F, n_perm, seed=SEED, offset=perm_offset,
                           partial=f'{stem}_null_partial.txt')
        s['perm_offset'] = int(perm_offset)
        s['null_mean'], s['null_sd'] = float(null.mean()), float(null.std())
        s['p_perm'] = float(((null >= s['bacc_subject_mean']).sum() + 1) / (len(null) + 1))
        np.savetxt(f'{stem}_null.txt', null)
    per.to_csv(f'{stem}_per_subject.csv', index=False)
    np.save(f'{stem}_preds.npy', preds)
    Path(f'{stem}_summary.json').write_text(json.dumps(s, indent=2, ensure_ascii=False),
                                            encoding='utf-8')
    print('--- 요약 ---')
    for k, v in s.items():
        print(f"  {k}: {v}")


def main():
    global DEVICE, SEED
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', choices=['event', 'cycle'], default='event')
    ap.add_argument('--model', default='hybrid',
                    help="cnn|cnn_lstm|cnn_tf|hybrid|csp|lgb|dur|all (쉼표로 여러 개)")
    ap.add_argument('--n-perm', type=int, default=0)
    ap.add_argument('--perm-offset', type=int, default=0,
                    help='이미 돌린 순열 수. 이어서 돌릴 때 (null 파일은 태그를 바꿔 따로 저장)')
    ap.add_argument('--subjects', type=int, default=0, help='smoke test: 앞 N명만')
    ap.add_argument('--no-aug', action='store_true')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='cpu', help="cpu | cuda | cuda:1")
    ap.add_argument('--seed', type=int, default=SEED, help='fold 분할·초기화 seed (seed 스윕용)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    DEVICE, SEED = a.device, a.seed
    if DEVICE.startswith('cuda'):
        assert torch.cuda.is_available(), 'CUDA 사용 불가: torch CUDA 빌드·드라이버 확인'
        print(f"device={DEVICE} ({torch.cuda.get_device_name(DEVICE)})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, length, y, meta = load_window(a.window)
    if a.subjects:
        keep = meta['subject_id'].isin(pd.unique(meta['subject_id'])[:a.subjects]).values
        X, length, y, meta = X[keep], length[keep], y[keep], meta[keep].reset_index(drop=True)
    print(f"window={a.window}  X={X.shape}  피험자 {meta['subject_id'].nunique()}명 · "
          f"샘플 {len(y)} · c 비율 {y.mean():.3f}")
    if 'duration' in meta:
        print("  duration 중앙값(s):",
              meta.groupby('condition')['duration'].median().round(1).to_dict())

    models = DL_MODELS + ['csp', 'lgb', 'dur'] if a.model == 'all' else a.model.split(',')
    for kind in models:
        run_one(kind, a.window, X, length, y, meta, a.n_perm, a.tag, aug=not a.no_aug,
                perm_offset=a.perm_offset)


if __name__ == '__main__':
    main()
