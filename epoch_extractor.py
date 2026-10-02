#!/usr/bin/env python3
"""이벤트 주변 EAG 시계열 epoch 추출 — 방향 디코더(CSP)의 입력을 만든다.

부하 디코더는 이벤트당 스칼라 8채널 벡터로 충분했지만, 방향 디코더에 쓰기로 한
common spatial patterns(CSP)는 **채널 간 공분산**을 추정해야 하므로 스칼라
벡터로는 성립하지 않는다. CSP는 채널 사이의 분산 구조를 최대로 가르는 공간
필터를 찾는 방법이고, 그 공분산은 한 시행 안의 시계열에서만 나온다.

전처리 (v2). 공분산 기반 분류는 '분산'으로 작동하므로, 드리프트와 불량 채널이
곧바로 특징이 된다. 초판에서 빠져 있던 네 가지를 넣었다:

  1. 고역통과 0.05 Hz  — 저역 5 Hz만 걸린 신호에는 sub-0.05 Hz 드리프트가 남아
     분산을 지배한다. 이걸 두면 CSP가 방향이 아니라 드리프트의 공간 패턴을 배운다.
  2. 연속신호 detrend  — 파라미터 추출 경로(detrend(sa.eag_filtered[:,ch]))와 일치시킨다.
     **epoch별 detrend가 아니다.** 반응 자체가 창 안에서 단조 편향이라 창 단위로
     선형 추세를 빼면 신호가 함께 깎인다. 60초 트레이스의 추세 제거는 안전하다.
  3. epoch 기저선 보정 — anchor 이전 [-1.0, -0.5]s 평균을 뺀다. 시행 간 DC 차이를
     없애 공분산이 '변화'만 담게 한다.
  4. 채널 품질 마스크  — evaluate_channel_quality의 PASS 여부를 epoch마다 기록한다.
     불량 채널을 지우지 않고 마스크로 남겨, 분류 단계에서 정책을 고를 수 있게 한다.

추출과 동시에 QC를 계산해 전처리가 의도대로 됐는지 검증한다(--qc-report).
검증 항목: 드리프트 기울기, 기저선 편차, 저주파 분산비의 전/후 비교와,
epoch에서 다시 잰 반응 크기가 부하량 및 파이프라인 amplitude와 맞는지.

사용:
  python3 epoch_extractor.py                    # 전체 배치 → result/ml/epochs.npz
  python3 epoch_extractor.py --limit 20         # 소수 세션으로 점검
  python3 epoch_extractor.py --no-preproc       # 초판과 동일(전/후 비교용)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

OUT = Path('result/ml/epochs.npz')
WIN_LO, WIN_HI = -1.0, 4.0
BASE_LO, BASE_HI = -1.0, -0.5      # anchor 이전 기저선 구간
FS_OUT = 25.0
N_CH = 8
# 고역통과 차단주파수. 8세션 스윕으로 정했다(--hp 로 재현 가능):
#   0.05Hz 드리프트 41.3 / 부하-반응 ρ=0.922   ← 드리프트가 거의 안 빠짐
#   0.10Hz 드리프트 20.7 / ρ=0.900             ← 채택
#   0.20Hz 드리프트  2.9 / ρ=0.846, 반응 89% 감쇠
# 반응 자체가 0.05~0.2Hz에 살기 때문에(시상수 2~5초 계단반응) 더 올리면 신호가 깎인다.
HP_HZ = 0.10
RESP_AT = 2.0                       # 반응 크기를 재는 시점(anchor 기준 초)


def highpass(x, fs, fc=None, order=2):
    """영위상 Butterworth 고역통과. 위상 왜곡이 없어야 이벤트 시각이 안 밀린다.

    기본인자로 HP_HZ를 박으면 정의 시점 값이 고정되어 이후 변경이 먹지 않는다.
    """
    from scipy.signal import butter, filtfilt
    fc = HP_HZ if fc is None else fc
    if fs <= 2 * fc:
        return x
    b, a = butter(order, fc / (fs / 2), btype='high')
    pad = min(len(x) - 1, int(fs * 20))
    return filtfilt(b, a, x, axis=0, padlen=pad)


def _drift_slope(ep, t):
    """epoch별 채널별 선형 추세 기울기(µV/s). 전처리 검증용 지표."""
    tc = t - t.mean()
    return (ep * tc).sum(axis=-1) / (tc ** 2).sum()


def _lf_ratio(ep, fs):
    """전체 분산 중 저주파(대략 <0.1 Hz) 성분이 차지하는 비율.

    창 길이가 5초라 0.1 Hz를 주파수 영역에서 분해할 수 없으므로, 창 전체에
    걸친 매우 느린 성분(선형 추세)이 설명하는 분산 비율로 대신 잰다.
    """
    t = np.arange(ep.shape[-1]) / fs
    tc = t - t.mean()
    sl = (ep * tc).sum(axis=-1) / (tc ** 2).sum()
    var_tr = (sl ** 2) * (tc ** 2).mean()
    var_all = ep.var(axis=-1)
    return np.where(var_all > 0, var_tr / (var_all + 1e-12), 0.0)


def _prepare_session(pair, apply_corrected=True, preproc=True):
    """세션 공통 전처리. 이벤트 epoch과 cycle 창이 같은 신호·anchor를 쓰도록 한 곳에 둔다.

    Returns: (cycles, anchors, te, sig, sig_raw, ch_pass, ch_snr) 또는 None
    """
    from sync_analyzer import SyncAnalyzer
    from grf_triggered_annotator import compute_offset
    import grf_triggered_annotator as _G
    from offset_manager import get_manual_offset
    from parameter_extractor import evaluate_channel_quality
    from scipy.signal import detrend as _detrend

    sa = SyncAnalyzer(pair)
    has_manual = get_manual_offset(pair.subject_name, pair.session_name) is not None
    off, trans, _s, _g = compute_offset(sa, ref_ch=0, recompute_offset=not has_manual)
    if apply_corrected and not has_manual and off.residual != 0.0:
        sa = SyncAnalyzer(pair, manual_offset=off.corrected_offset)
        _o2, trans, _s, _g = compute_offset(sa, ref_ch=0, recompute_offset=False)

    signed = _G.signed_imbalance(sa.grf_left, sa.grf_right)
    _rest, cycles, _info = _G.detect_load_cycles_expected(
        sa.unified_time_grf, signed, sa.grf_left, sa.grf_right)
    if len(cycles) != _G.EXPECTED_CYCLES:
        return None
    anchors = _G.cycles_to_transitions(cycles, trans)

    te = sa.unified_time_eag
    raw = np.asarray(sa.eag_filtered, dtype=float)          # (n_samples, 8)
    fs_in = float(sa.eag.sample_rate)

    # (4) 채널 품질 — 지우지 않고 마스크로 남긴다
    q = evaluate_channel_quality(sa.eag)
    ch_pass = np.array([q[c]['flag'] == 'PASS' for c in sorted(q)][:N_CH], dtype=bool)
    ch_snr = np.array([q[c]['snr_db'] for c in sorted(q)][:N_CH], dtype=float)

    sig_raw = raw
    if preproc:
        sig = highpass(raw, fs_in)                          # (1) 고역통과
        sig = _detrend(sig, axis=0, type='linear')          # (2) 연속신호 detrend
    else:
        sig = raw
    return cycles, anchors, te, sig, sig_raw, ch_pass, ch_snr


# cycle 창(부하 시작 → 이탈 전 과정). 실측 부하 유지 중앙값 c 10 s / s 14.5 s / f 5.5 s,
# p95 26 s. onset -2 s ~ offset +3 s, 30 s 초과분은 자르고 truncated 플래그를 남긴다.
# 길이를 리샘플로 고정하지 않는다: 감쇠 시정수 같은 절대 시간 정보가 사라진다.
CYC_PRE, CYC_POST, CYC_MAX = 2.0, 3.0, 30.0
CYC_LEN = int(CYC_MAX * FS_OUT)     # 750


def extract_session_cycles(pair, apply_corrected=True, preproc=True):
    """한 세션에서 cycle 단위 가변 길이 창 (n_cycles, 8, CYC_LEN)과 길이·메타를 만든다.

    이벤트 epoch과 동일한 anchor(snap 보정된 on/off 시각)를 써서 두 창이 같은 사건을
    가리키게 한다. zero-pad 뒤 `length`로 mask를 만들어 모델에 넘긴다.
    """
    prep = _prepare_session(pair, apply_corrected, preproc)
    if prep is None:
        return None, None
    cycles, anchors, te, sig, sig_raw, ch_pass, ch_snr = prep
    dt = 1.0 / FS_OUT
    bsel_t = np.arange(-CYC_PRE, 0, dt)
    bsel = (bsel_t >= BASE_LO) & (bsel_t < BASE_HI)
    eps, lens, meta = [], [], []
    for k, cyc in enumerate(cycles):
        if 2 * k + 1 >= len(anchors):
            break
        t_on, t_off = float(anchors[2 * k].time), float(anchors[2 * k + 1].time)
        if not (t_off > t_on):
            continue
        t_start = t_on - CYC_PRE
        dur_win = min((t_off + CYC_POST) - t_start, CYC_MAX)
        n = int(round(dur_win * FS_OUT))
        tt = t_start + np.arange(n) * dt
        if tt[0] < te[0] or tt[-1] > te[-1]:
            continue
        ep = np.zeros((N_CH, CYC_LEN), dtype=np.float32)
        for c in range(N_CH):
            x = np.interp(tt, te, sig[:, c])
            if preproc:
                x = x - x[:len(bsel)][bsel].mean()          # onset 이전 기저선 보정
            ep[c, :n] = x
        eps.append(ep); lens.append(n)
        meta.append({'subject': pair.subject_name, 'session': pair.session_name,
                     'cycle_id': cyc.cycle_id,
                     'load_pct': round(float(cyc.load_pct), 1),
                     'duration': round(t_off - t_on, 3),
                     'truncated': bool((t_off + CYC_POST) - t_start > CYC_MAX),
                     'n_pass': int(ch_pass.sum())})
    if not eps:
        return None, None
    for m in meta:
        m.update({f'pass_ch{c+1}': bool(ch_pass[c]) for c in range(N_CH)})
        m.update({f'snr_ch{c+1}': float(ch_snr[c]) for c in range(N_CH)})
    for m, n in zip(meta, lens):
        m['length'] = int(n)
    return np.stack(eps), meta


def extract_session(pair, apply_corrected=True, preproc=True):
    """한 세션에서 (n_events, 8, n_time) epoch과 메타·QC를 만든다."""
    prep = _prepare_session(pair, apply_corrected, preproc)
    if prep is None:
        return None, None
    cycles, anchors, te, sig, sig_raw, ch_pass, ch_snr = prep

    grid = np.arange(WIN_LO, WIN_HI, 1.0 / FS_OUT)
    bsel = (grid >= BASE_LO) & (grid < BASE_HI)
    ridx = int(np.argmin(np.abs(grid - RESP_AT)))
    eps, meta, qc = [], [], []
    for i, a in enumerate(anchors):
        t0 = getattr(a, 'time', None)
        if t0 is None:
            continue
        tt = t0 + grid
        if tt[0] < te[0] or tt[-1] > te[-1]:
            continue
        ep = np.empty((N_CH, len(grid)), dtype=np.float64)
        ep0 = np.empty_like(ep)
        for c in range(N_CH):
            ep[c] = np.interp(tt, te, sig[:, c])
            ep0[c] = np.interp(tt, te, sig_raw[:, c])
        # QC: 보정 전 상태
        d_before = _drift_slope(ep0, grid)
        b_before = ep0[:, bsel].mean(axis=1)
        lf_before = _lf_ratio(ep0, FS_OUT)
        if preproc:
            ep = ep - ep[:, bsel].mean(axis=1, keepdims=True)   # (3) 기저선 보정
        d_after = _drift_slope(ep, grid)
        b_after = ep[:, bsel].mean(axis=1)
        lf_after = _lf_ratio(ep, FS_OUT)

        cyc = cycles[i // 2]
        eps.append(ep.astype(np.float32))
        meta.append({'subject': pair.subject_name, 'session': pair.session_name,
                     'trans_id': i, 'cycle_id': cyc.cycle_id,
                     'load_pct': round(float(cyc.load_pct), 1),
                     'event_kind': 'on' if i % 2 == 0 else 'off'})
        qc.append({'drift_before': float(np.abs(d_before).mean()),
                   'drift_after': float(np.abs(d_after).mean()),
                   'base_before': float(np.abs(b_before).mean()),
                   'base_after': float(np.abs(b_after).mean()),
                   'lf_before': float(lf_before.mean()),
                   'lf_after': float(lf_after.mean()),
                   # epoch에서 다시 잰 반응 크기 — 파이프라인 amplitude와 대조할 값
                   'resp_amp': float(np.abs(ep[:, ridx]).mean()),
                   'n_pass': int(ch_pass.sum())})
    if not eps:
        return None, None
    for m, k in zip(meta, range(len(meta))):
        m.update({f'pass_ch{c+1}': bool(ch_pass[c]) for c in range(N_CH)})
        m.update({f'snr_ch{c+1}': float(ch_snr[c]) for c in range(N_CH)})
        m.update(qc[k])
    return np.stack(eps), meta


def qc_report(meta, out_dir: Path):
    """전처리가 의도대로 됐는지 수치로 확인한다."""
    import pandas as pd
    m = pd.DataFrame(meta)
    rows = []
    for lab, a, b in [('드리프트 |기울기| (µV/s)', 'drift_before', 'drift_after'),
                      ('기저선 |편차| (µV)', 'base_before', 'base_after'),
                      ('저주파 분산비', 'lf_before', 'lf_after')]:
        rows.append({'지표': lab,
                     '전(중앙값)': round(float(m[a].median()), 4),
                     '후(중앙값)': round(float(m[b].median()), 4),
                     '감소율(%)': round(100 * (1 - m[b].median() / (m[a].median() + 1e-12)), 1)})
    qc = pd.DataFrame(rows)

    # 신호 보존 검증: 반응 크기가 부하량을 따라가는가
    sub = m.dropna(subset=['resp_amp', 'load_pct'])
    import scipy.stats as sps
    r = sps.spearmanr(sub['load_pct'], sub['resp_amp'])
    lin = sps.linregress(sub['load_pct'], sub['resp_amp'])
    preserve = pd.DataFrame([{
        'n_epochs': len(sub),
        'spearman_rho(load, resp)': round(float(r.statistic), 3),
        'p': f'{r.pvalue:.2e}',
        'slope_uV_per_pctBW': round(float(lin.slope), 3),
        'r2': round(float(lin.rvalue ** 2), 3),
        'pass_ch_mean': round(float(m['n_pass'].mean()), 2)}])
    qc.to_csv(out_dir / 'epoch_qc.csv', index=False, encoding='utf-8-sig')
    preserve.to_csv(out_dir / 'epoch_qc_preservation.csv', index=False)
    print("\n=== 전처리 검증 (전 → 후) ===")
    print(qc.to_string(index=False))
    print("\n=== 신호 보존 검증 (epoch에서 다시 잰 반응 vs 실측 부하) ===")
    print(preserve.to_string(index=False))
    print("  * 전처리가 반응을 깎았다면 이 상관이 무너진다. "
          "파라미터 경로의 dose 기울기는 1.45 µV/%BW.")
    return qc, preserve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--mode', choices=['event', 'cycle'], default='event',
                    help='event: 이벤트 ±창(기본, epochs.npz) · cycle: 부하 시작~이탈 전 과정(cycles.npz)')
    ap.add_argument('--no-preproc', action='store_true', help='초판과 동일(비교용)')
    ap.add_argument('--hp', type=float, default=None, help='고역통과 차단주파수(Hz)')
    a = ap.parse_args()
    if a.hp is not None:
        globals()['HP_HZ'] = a.hp
    if a.out is None:
        a.out = str(OUT) if a.mode == 'event' else str(OUT.with_name('cycles.npz'))

    from sync_analyzer import find_all_pairs
    from eag_analyzer import get_data_dir
    pairs = find_all_pairs(get_data_dir())
    if a.limit:
        pairs = pairs[:a.limit]
    win = (f"{WIN_LO}~{WIN_HI}s" if a.mode == 'event'
           else f"onset-{CYC_PRE}s ~ offset+{CYC_POST}s (max {CYC_MAX}s, pad {CYC_LEN})")
    print(f"세션 {len(pairs)}개 · mode={a.mode} · 창 {win} @ {FS_OUT}Hz · "
          f"전처리={'off' if a.no_preproc else f'HP{HP_HZ}Hz+detrend+baseline+qmask'}")
    extract = extract_session if a.mode == 'event' else extract_session_cycles

    X, M, fails = [], [], 0
    for k, p in enumerate(pairs, 1):
        try:
            x, m = extract(p, preproc=not a.no_preproc)
            if x is not None:
                X.append(x); M.extend(m)
        except Exception as e:
            fails += 1
            if fails <= 5:
                print(f"  [skip] {p.subject_name}/{p.session_name}: {str(e)[:70]}")
        if k % 50 == 0:
            print(f"  {k}/{len(pairs)} · epoch {sum(len(x) for x in X)} · 실패 {fails}",
                  flush=True)
    if not X:
        print("추출된 epoch 없음"); return
    X = np.concatenate(X)
    import pandas as pd
    meta = pd.DataFrame(M)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, X=X, **{c: meta[c].values for c in meta.columns})
    print(f"\nepoch {X.shape} · 세션실패 {fails}")
    print(f"  조건별: {meta['session'].str[0].value_counts().to_dict()}")
    if a.mode == 'event':
        qc_report(M, outp.parent)
    else:
        cond = meta['session'].str[0]
        print("  부하 유지 duration(s) 조건별 중앙값:",
              meta.groupby(cond)['duration'].median().round(1).to_dict(),
              f"· truncated {int(meta['truncated'].sum())}")
    print(f"\n-> {outp}")


if __name__ == '__main__':
    main()
