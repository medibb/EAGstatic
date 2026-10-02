#!/usr/bin/env python3
"""목발 분류 1차 실험 보고서용 그림 8장. result/ml/fig/ 에 저장.

  fig1_decomposition   실제 세션 1개에서 GRF·EAG 신호를 event 창 / cycle 창으로 자르는 방식
  fig2_grand_average   모델 입력(정규화 epoch)의 조건별 평균 파형 — 왜 어려운 문제인지
  fig3_models          모델 4종의 블록 구성과 파라미터 수 (ablation 설계)
  fig4_results         창 × 모델 balanced accuracy (subject mean ± sd)
  fig5_permutation     순열 null 분포와 관측값
  fig6_subjects        피험자별 구분 가능도와 창 간 재현성
  fig7_confound        조건별 load_pct 분포 (부하 교란 점검)
  fig8_hetero          구분 가능도 vs 공변량 Spearman rho

사용: python3 ml_crutch_report.py [--session 주창민_1/c2]
"""
import argparse
import glob
import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
OUT = Path('result/ml/fig')
ML = Path('result/ml')
OBS_IMG = Path('/workspace/obsidian/images/eag_crutch')

# 팔레트(검증 완료: 3 슬롯 all-pairs PASS). 조건: c=blue, s=orange, f=aqua. 창: event=blue, cycle=orange.
COL = {'c': '#2a78d6', 's': '#eb6834', 'f': '#1baf7a', 'event': '#2a78d6', 'cycle': '#eb6834'}
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
COND_NAME = {'c': 'crutch (c)', 's': 'side (s)', 'f': 'front (f)'}
MODELS = ['cnn', 'cnn_lstm', 'cnn_tf', 'hybrid', 'csp', 'lgb', 'dur']
MODEL_NAME = {'cnn': 'CNN\n(EEGNet)', 'cnn_lstm': 'CNN\n+LSTM', 'cnn_tf': 'CNN\n+Transformer',
              'hybrid': 'CNN+LSTM\n+Transformer', 'csp': 'CSP+LDA', 'lgb': 'LightGBM\n(features)',
              'dur': 'duration\nonly'}

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK2, 'text.color': INK,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'font.family': 'DejaVu Sans'})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f'{name}.png'
    fig.savefig(p, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ->', p)


def load_summary(window, model):
    """csp/lgb는 클래스 균형 보정본(perm 태그)이 정본. 없으면 무태그."""
    for suf in ('_perm', ''):
        p = ML / f'crutch_{window}_{model}{suf}_summary.json'
        if p.exists():
            return json.load(open(p, encoding='utf-8'))
    return None


# ---------------------------------------------------------------- fig1
def fig1_decomposition(session_key):
    from sync_analyzer import SyncAnalyzer, find_all_pairs
    from eag_analyzer import get_data_dir
    from grf_triggered_annotator import compute_offset
    import grf_triggered_annotator as G
    from offset_manager import get_manual_offset
    from epoch_extractor import highpass, WIN_LO, WIN_HI, CYC_PRE, CYC_POST
    from scipy.signal import detrend

    subj, sess = session_key.split('/')
    pair = [p for p in find_all_pairs(get_data_dir())
            if subj in p.subject_name and p.session_name == sess][0]
    sa = SyncAnalyzer(pair)
    has_manual = get_manual_offset(pair.subject_name, pair.session_name) is not None
    off, trans, _, _ = compute_offset(sa, ref_ch=0, recompute_offset=not has_manual)
    if not has_manual and off.residual != 0.0:
        sa = SyncAnalyzer(pair, manual_offset=off.corrected_offset)
        _, trans, _, _ = compute_offset(sa, ref_ch=0, recompute_offset=False)
    tg = sa.unified_time_grf
    signed = G.signed_imbalance(sa.grf_left, sa.grf_right)
    _, cycles, _ = G.detect_load_cycles_expected(tg, signed, sa.grf_left, sa.grf_right)
    anchors = G.cycles_to_transitions(cycles, trans)
    te = sa.unified_time_eag
    sig = detrend(highpass(np.asarray(sa.eag_filtered, float), float(sa.eag.sample_rate)),
                  axis=0, type='linear')
    ch = 0
    ons = [a.time for a in anchors[0::2]]
    offs = [a.time for a in anchors[1::2]]
    t0, t1 = ons[0] - 5, offs[-1] + 6
    mg, me = (tg >= t0) & (tg <= t1), (te >= t0) & (te <= t1)

    fig, ax = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1.2, 1.2], 'hspace': 0.18})
    ax[0].plot(tg[mg] - t0, signed[mg], color=INK2, lw=1.2)
    ax[0].set_ylabel('GRF signed\nimbalance')
    for k, (a, b) in enumerate(zip(ons, offs)):
        ax[0].axvspan(a - t0, b - t0, color=GRID, alpha=0.6, lw=0)
        ax[0].text((a + b) / 2 - t0, ax[0].get_ylim()[1] * 0.9 if k == 0 else ax[0].get_ylim()[1] * 0.9,
                   f'{cycles[k].load_pct:.0f}%BW', ha='center', va='top', fontsize=8, color=INK2)
    pooled = pd.read_csv('result/stats/grf_eag_pooled.csv', low_memory=False,
                         usecols=['subject', 'subject_no']).drop_duplicates('subject')
    hit = pooled[pooled['subject'] == pair.subject_name]['subject_no']
    subj_no = int(hit.iloc[0]) if len(hit) else '?'
    visit_no = pair.subject_name.rsplit('_', 1)[-1]
    ax[0].set_title(f'One session (subject {subj_no}, visit {visit_no}, session {sess}): '
                    f'4 load cycles, loading = grey band')
    for r, (name, spans, col) in enumerate([
            ('event window  [-1, +4] s around each on/off anchor  (125 samples @ 25 Hz)',
             [(t + WIN_LO, t + WIN_HI) for t in ons + offs], COL['event']),
            ('cycle window  [onset-2, offset+3] s  (variable length, zero-padded to 750)',
             [(a - CYC_PRE, b + CYC_POST) for a, b in zip(ons, offs)], COL['cycle'])], start=1):
        ax[r].plot(te[me] - t0, sig[me, ch], color=INK, lw=0.9)
        for a, b in spans:
            ax[r].axvspan(a - t0, b - t0, color=col, alpha=0.18, lw=0)
        ax[r].set_ylabel(f'EAG ch{ch + 1} (µV)')
        ax[r].set_title(name, color=col)
    for a_ in ax:
        for t in ons:
            a_.axvline(t - t0, color=COL['c'], lw=0.8, alpha=0.7)
        for t in offs:
            a_.axvline(t - t0, color=COL['s'], lw=0.8, alpha=0.7)
    ax[2].set_xlabel('time (s)')
    ax[0].plot([], [], color=COL['c'], lw=1, label='load onset anchor')
    ax[0].plot([], [], color=COL['s'], lw=1, label='load offset anchor')
    ax[0].legend(loc='lower right', fontsize=8, ncol=2)
    save(fig, 'fig1_decomposition')


# ---------------------------------------------------------------- fig2
def fig2_grand_average():
    from ml_crutch import load_window
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    for r, (window, fs) in enumerate([('event', 25.0), ('cycle', 25.0)]):
        X, length, y, meta = load_window(window)
        T = X.shape[2]
        t = np.arange(T) / fs - (1.0 if window == 'event' else 2.0)
        valid = np.arange(T)[None, :] < length[:, None]          # (n, T) 패딩 제외 마스크
        # cycle 창은 가변 길이라 시점별로 유효한 cycle만 평균하고, 절반 이상 유효한 시점까지만 그린다
        frac = valid.mean(0)
        keep_t = frac >= 0.5 if window == 'cycle' else np.ones(T, bool)
        for c_i, ch in enumerate([0, 4]):
            a = ax[r, c_i]
            for cond in ['s', 'f', 'c']:
                m = (meta['condition'] == cond).values
                xv = np.where(valid[m], X[m, ch], np.nan)
                mu = np.nanmean(xv, 0)
                se = np.nanstd(xv, 0) / np.sqrt(valid[m].sum(0))
                a.plot(t[keep_t], mu[keep_t], color=COL[cond], label=f'{COND_NAME[cond]}  n={m.sum():,}')
                a.fill_between(t[keep_t], (mu - 1.96 * se)[keep_t], (mu + 1.96 * se)[keep_t],
                               color=COL[cond], alpha=0.15, lw=0)
            a.axvline(0, color=AXIS, lw=0.8)
            if window == 'cycle':
                a.axvline(4.8, color=AXIS, lw=0.8)
                a.text(4.8, a.get_ylim()[1], ' median offset', va='top', fontsize=7.5, color=MUTED)
            a.set_title(f'{window} window · ch{ch + 1} ({"medial" if ch < 4 else "lateral"})')
            a.set_xlabel('time from load onset (s)' if window == 'cycle' else 'time from anchor (s)')
            a.set_ylabel('normalised EAG (z)')
            if c_i == 0:
                a.legend(fontsize=8, loc='lower right')
    fig.suptitle('What the model sees: trial-normalised mean ± 95% CI by condition',
                 fontsize=11, fontweight='bold', y=0.98)
    save(fig, 'fig2_grand_average')


# ---------------------------------------------------------------- fig3
def fig3_models():
    blocks = ['Input\n8 ch × T', 'Temporal\nconv\n8 filt, k=13', 'Depthwise\nspatial conv\n8→16',
              'Separable\nconv\n+ pool ÷8', 'BiLSTM\nhidden 16', 'Linear→32\n+ pos.enc.\nTransformer ×2',
              'Pooling\nmasked mean\n/ attention', 'FC → logit\ncrutch\nvs not']
    rows = [('cnn', [1, 1, 1, 1, 0, 0, 1, 1], 697),
            ('cnn_lstm', [1, 1, 1, 1, 1, 0, 1, 1], 5065),
            ('cnn_tf', [1, 1, 1, 1, 0, 1, 1, 1], 18378),
            ('hybrid', [1, 1, 1, 1, 1, 1, 1, 1], 23242)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.grid(False); ax.set_axis_off()
    W, H, gx = 1.0, 0.62, 0.42
    for j, b in enumerate(blocks):
        ax.text(j * (W + gx) + W / 2, len(rows) * 0.9 + 0.35, b, ha='center', va='bottom',
                fontsize=7.5, color=INK2, linespacing=1.15)
    for i, (name, inc, par) in enumerate(rows):
        yy = (len(rows) - 1 - i) * 0.9
        ax.text(-0.25, yy + H / 2, MODEL_NAME[name].replace('\n', ' '), ha='right', va='center',
                fontsize=9, fontweight='bold', color=INK)
        ax.text(len(blocks) * (W + gx) + 0.05, yy + H / 2, f'{par:,} params', ha='left',
                va='center', fontsize=8.5, color=INK2)
        for j, on in enumerate(inc):
            x = j * (W + gx)
            shared = j in (0, 1, 2, 3, 6, 7)
            col = (MUTED if shared else COL['event']) if on else SURF
            ec = col if on else GRID
            ax.add_patch(plt.Rectangle((x, yy), W, H, facecolor=col, edgecolor=ec, lw=0.8,
                                       alpha=0.85 if on else 1, joinstyle='round'))
            if j < len(blocks) - 1:
                ax.annotate('', xy=(x + W + gx, yy + H / 2), xytext=(x + W, yy + H / 2),
                            arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8))
    ax.set_xlim(-2.6, len(blocks) * (W + gx) + 1.3)
    ax.set_ylim(-0.3, len(rows) * 0.9 + 1.6)
    ax.set_title('Model decomposition (ablation): grey = shared EEGNet front & head, '
                 'blue = added sequence block', loc='left')
    save(fig, 'fig3_models')


# ---------------------------------------------------------------- fig4
def fig4_results():
    rows = []
    for w in ['event', 'cycle']:
        for m in MODELS:
            s = load_summary(w, m)
            if s:
                rows.append(dict(window=w, model=m, bacc=s['bacc_subject_mean'],
                                 sd=s['bacc_subject_sd'], auc=s['auc_subject_mean'],
                                 par=s['n_param'], p=s['p_vs_chance'],
                                 p_perm=s.get('p_perm', np.nan)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'results_table.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = np.arange(len(MODELS))
    for k, w in enumerate(['event', 'cycle']):
        g = d[d.window == w].set_index('model').reindex(MODELS)
        off = -0.16 if w == 'event' else 0.16
        ax.errorbar(xs + off, g['bacc'], yerr=g['sd'], fmt='o', color=COL[w], ms=7,
                    capsize=0, elinewidth=1.2, label=f'{w} window')
        for x, v in zip(xs + off, g['bacc']):
            if np.isfinite(v):
                ax.text(x, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.text(len(MODELS) - 0.5, 0.503, 'chance (balanced)', ha='right', va='bottom', fontsize=8, color=MUTED)
    ax.axhline(0.60, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(len(MODELS) - 0.5, 0.603, 'pre-registered target 0.60', ha='right', va='bottom',
            fontsize=8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_NAME[m] for m in MODELS], fontsize=8.5)
    pars = d[d.window == 'event'].set_index('model')['par'].reindex(MODELS)
    for x, p in zip(xs, pars):
        if np.isfinite(p):
            ax.text(x, 0.395, f'{int(p):,} par' if p > 0 else 'no NN', ha='center', fontsize=7.5, color=MUTED)
    ax.set_ylim(0.38, 0.72)
    ax.set_ylabel('balanced accuracy, LOSO (subject mean ± sd, n = 41)')
    ax.set_title('Crutch vs non-crutch: every model lands at 0.53–0.55; bigger models do not help',
                 loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)
    save(fig, 'fig4_results')
    return d


# ---------------------------------------------------------------- fig5
def fig5_permutation():
    panels = [('event', 'cnn'), ('cycle', 'cnn'), ('event', 'csp'), ('event', 'lgb'),
              ('cycle', 'lgb'), ('cycle', 'dur')]
    fig, axes = plt.subplots(1, len(panels), figsize=(17.5, 3.6), sharey=False,
                             gridspec_kw={'wspace': 0.32})
    for a, (w, m) in zip(axes, panels):
        # pooled null(예: perm60 = 20 + 40회)이 있으면 그것을 쓴다
        nf = next((ML / f'crutch_{w}_{m}_{t}_null.txt' for t in ('perm100', 'perm60', 'perm')
                   if (ML / f'crutch_{w}_{m}_{t}_null.txt').exists()), None)
        s = load_summary(w, m)
        if nf is None or s is None:
            a.set_axis_off(); continue
        null = np.loadtxt(nf)
        obs = s['bacc_subject_mean']
        s = dict(s, p_perm=((null >= obs).sum() + 1) / (len(null) + 1))
        a.hist(null, bins=12, color=COL[w], alpha=0.35, edgecolor=SURF, lw=1)
        a.axvline(obs, color=INK, lw=1.8)
        right = obs > null.mean()
        a.text(obs, a.get_ylim()[1] * 0.97, f"obs {obs:.3f} " if right else f" obs {obs:.3f}",
               va='top', ha='right' if right else 'left', fontsize=8, fontweight='bold')
        a.set_title(f"{w} · {MODEL_NAME[m].replace(chr(10), ' ')}\np = {s['p_perm']:.3f}", fontsize=9)
        a.set_xlabel(f"balanced accuracy\nnull {null.mean():.3f} ± {null.std():.3f} (n = {len(null)})")
        a.grid(axis='y', visible=False)
        lo, hi = min(null.min(), obs), max(null.max(), obs)
        a.set_xlim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    axes[0].set_ylabel('permutations')
    fig.suptitle('Session-level label permutation within subject: observed vs null', fontsize=11,
                 fontweight='bold', y=1.06)
    save(fig, 'fig5_permutation')


# ---------------------------------------------------------------- fig6
def fig6_subjects():
    per = {}
    for f in glob.glob(str(ML / 'crutch_*_per_subject.csv')):
        k = Path(f).name.replace('crutch_', '').replace('_per_subject.csv', '')
        if 'perm' in k or not any(t in k for t in ('cnn', 'hybrid')):
            continue
        per[k] = pd.read_csv(f).set_index('subject_id')['bacc']
    d = pd.DataFrame(per).dropna()
    score = d.mean(axis=1).sort_values(ascending=False)
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.6, 1], 'wspace': 0.3})
    xs = np.arange(len(score))
    lo, hi = d.min(axis=1)[score.index], d.max(axis=1)[score.index]
    ax[0].vlines(xs, lo, hi, color=GRID, lw=2.2)
    ax[0].scatter(xs, score, color=COL['event'], s=28, zorder=3, label='mean of 8 DL models')
    ax[0].scatter(xs, d['event_cnn'][score.index], marker='_', color=INK, s=70, zorder=4,
                  label='event · CNN')
    ax[0].axhline(0.5, color=AXIS, lw=1)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([s.split('.')[0] for s in score.index], fontsize=7)
    ax[0].set_xlabel('subject (sorted by decodability)'); ax[0].set_ylabel('balanced accuracy')
    ax[0].set_title('Per-subject decodability (line = range over 8 DL models)', loc='left')
    ax[0].legend(loc='upper right', fontsize=8); ax[0].grid(axis='x', visible=False)
    x, y = d['event_cnn'], d['cycle_cnn']
    r = spearmanr(x, y)
    ax[1].scatter(x, y, color=COL['event'], s=30, edgecolor=SURF, lw=1)
    lim = (0.35, 0.8)
    ax[1].plot(lim, lim, color=AXIS, lw=0.8); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel('event window · CNN bacc'); ax[1].set_ylabel('cycle window · CNN bacc')
    ax[1].set_title(f'Same subjects decodable in both windows\nSpearman ρ = {r.statistic:.2f}, p = {r.pvalue:.3f}',
                    loc='left')
    save(fig, 'fig6_subjects')


# ---------------------------------------------------------------- fig7
def fig7_confound():
    ev = pd.read_csv(ML / 'events.csv', low_memory=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'wspace': 0.3})
    bins = np.arange(0, 105, 5)
    for cond in ['s', 'f', 'c']:
        v = ev.loc[ev.condition == cond, 'load_pct'].dropna()
        ax[0].hist(v, bins=bins, histtype='step', lw=1.8, color=COL[cond],
                   label=f'{COND_NAME[cond]}  median {v.median():.0f}%')
    ax[0].set_xlabel('measured load on tested limb (%BW)'); ax[0].set_ylabel('events')
    ax[0].set_title('Load distribution is the same across conditions', loc='left')
    ax[0].legend(fontsize=8); ax[0].grid(axis='x', visible=False)
    z = np.load(ML / 'cycles.npz', allow_pickle=True)
    m = pd.DataFrame({'cond': pd.Series(z['session']).astype(str).str[0], 'dur': z['duration']})
    for cond in ['s', 'f', 'c']:
        v = m.loc[m.cond == cond, 'dur']
        ax[1].hist(v, bins=np.arange(0, 20, 0.5), histtype='step', lw=1.8, color=COL[cond],
                   label=f'{COND_NAME[cond]}  median {v.median():.1f} s')
    ax[1].set_xlabel('load hold duration per cycle (s)'); ax[1].set_ylabel('cycles')
    ax[1].set_title('Cycle length is the same across conditions', loc='left')
    ax[1].legend(fontsize=8); ax[1].grid(axis='x', visible=False)
    save(fig, 'fig7_confound')


# ---------------------------------------------------------------- fig8
def fig8_hetero():
    t = pd.read_csv(ML / 'crutch_hetero_tests.csv')
    t = t[t.test == 'spearman'].sort_values('stat')
    nice = {'abs_diff_c_s': '|dose slope c − s|', 'amp_mean': 'EAG response amplitude',
            'slope': 'dose slope (all)', 'n_pass_mean': 'channels passing QC', 'age': 'age',
            'muscle': 'skeletal muscle mass', 'n_visits': 'visits', 'fat_pct': 'body fat %',
            'take_gap_c_minus_sf': 'crutch sessions later in visit', 'load_sd_ratio': 'load SD ratio c/sf',
            'load_med_diff': '|median load c − sf|', 'n_c_takes': 'crutch takes', 'snr_mean': 'mean SNR',
            'bmi': 'BMI'}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ys = np.arange(len(t))
    ax.hlines(ys, 0, t['stat'], color=GRID, lw=2.2)
    ax.scatter(t['stat'], ys, color=COL['event'], s=34, zorder=3)
    for y_, (s_, p_) in enumerate(zip(t['stat'], t['p'])):
        ax.text(s_ + (0.02 if s_ >= 0 else -0.02), y_, f'p={p_:.2f}', va='center',
                ha='left' if s_ >= 0 else 'right', fontsize=7.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([nice.get(c, c) for c in t['covariate']], fontsize=8.5)
    ax.set_xlabel('Spearman ρ with subject decodability (mean bacc, 8 DL models)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_title('Nothing explains who is decodable (all Holm-adjusted p ≥ 0.77; sex p = 0.85)', loc='left')
    ax.grid(axis='y', visible=False)
    save(fig, 'fig8_hetero')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='주창민_1/c2')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    steps = {'1': lambda: fig1_decomposition(a.session), '2': fig2_grand_average, '3': fig3_models,
             '4': fig4_results, '5': fig5_permutation, '6': fig6_subjects, '7': fig7_confound,
             '8': fig8_hetero}
    for k, fn in steps.items():
        if a.only and k not in a.only:
            continue
        print(f'fig{k}')
        fn()
    OBS_IMG.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob('fig*.png'):
        shutil.copy(p, OBS_IMG / p.name)
    print('copied ->', OBS_IMG)


if __name__ == '__main__':
    main()
