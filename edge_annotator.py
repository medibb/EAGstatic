"""
Edge/Knee Annotator — plateau 전이(엣지)의 corner point 자동 검출

manual annotation(comparison_annotated.png의 빨간 점)을 자동화한다.
빨간 점 = 각 상승/하강 전이의 시작 knee(직전 plateau 끝) + 끝 knee(다음 plateau 시작).
peak/valley(극값)가 아니라 corner point(곡률 최대 지점)이므로
sync_analyzer.detect_eag_inflections(find_peaks)와는 다른 알고리즘을 쓴다.

파이프라인은 기존 EAGAnalyzer(LP 5Hz)를 재사용한다.

사용법:
  python3 edge_annotator.py --file <BrainFlow-RAW.csv> --channel 1
  python3 edge_annotator.py --file <csv> --channel 1 --no-plot   # CSV만
"""

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from eag_analyzer import EAGAnalyzer, FilterConfig, setup_korean_font, SAMPLE_RATE


# ==================== 파라미터 dataclass ====================

@dataclass
class EdgeParams:
    """단일 전이(엣지)의 파라미터."""
    edge_id: int
    direction: str          # 'rise' | 'fall'
    onset_time: float       # 시작 knee 시각 (s)
    offset_time: float      # 끝 knee 시각 (s)
    onset_amp: float        # 시작 knee 진폭 (µV)
    offset_amp: float       # 끝 knee 진폭 (µV)
    transition_time: float  # offset - onset (s)
    amplitude: float        # offset_amp - onset_amp (µV, 부호 포함)
    slope: float            # amplitude / transition_time (µV/s)
    plateau_before: float   # 직전 plateau 레벨 (µV)
    plateau_after: float    # 다음 plateau 레벨 (µV)
    plateau_before_dur: float  # 직전 plateau 지속시간 (s)


# ==================== 검출 알고리즘 ====================

def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """이동평균 (경계 반사)."""
    if win <= 1:
        return x
    pad = win // 2
    xp = np.pad(x, pad, mode='reflect')
    k = np.ones(win) / win
    return np.convolve(xp, k, mode='same')[pad:-pad]


def detect_edges(time: np.ndarray,
                 sig: np.ndarray,
                 fs: int = SAMPLE_RATE,
                 smooth_sec: float = 0.20,
                 slope_k: float = 3.0,
                 min_gap_sec: float = 0.30,
                 min_amp: float = 25.0,
                 plateau_win_sec: float = 0.40,
                 settle_frac: float = 0.10) -> List[EdgeParams]:
    """plateau 전이의 knee(corner) 두 개씩을 검출한다.

    Args:
        smooth_sec: 미분 전 평활 윈도우
        slope_k: robust slope 임계 배수 (MAD 기준)
        min_gap_sec: 인접 전이 병합 간격
        min_amp: 유효 전이로 볼 최소 진폭 (µV)
        plateau_win_sec: plateau 레벨 추정 윈도우
        settle_frac: knee를 plateau에 안착시키는 허용 밴드 (진폭 대비 비율)
    """
    n = len(sig)
    sm = _smooth(sig, max(1, int(smooth_sec * fs)))
    slope = np.gradient(sm) * fs  # µV/s

    # robust 임계 (MAD)
    med = np.median(slope)
    mad = np.median(np.abs(slope - med)) + 1e-9
    thr = slope_k * 1.4826 * mad

    active = np.abs(slope - med) > thr

    # 연속 active 구간 → segment
    segs = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            segs.append([i, j - 1])
            i = j
        else:
            i += 1
    if not segs:
        return []

    # 간격 작은 segment 병합
    gap = int(min_gap_sec * fs)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    pw = max(1, int(plateau_win_sec * fs))
    edges: List[EdgeParams] = []
    eid = 0
    for k, (s, e) in enumerate(merged):
        # 직전/직후 plateau 레벨 (segment 바깥 윈도우의 중앙값)
        pre_a, pre_b = max(0, s - pw), s
        post_a, post_b = e, min(n, e + pw)
        if pre_b - pre_a < 3 or post_b - post_a < 3:
            continue
        lvl_before = float(np.median(sig[pre_a:pre_b]))
        lvl_after = float(np.median(sig[post_a:post_b]))
        amp = lvl_after - lvl_before
        if abs(amp) < min_amp:
            continue  # 미세 wiggle 무시

        band = abs(amp) * settle_frac
        # onset knee: pre-level 밴드를 벗어나는 마지막 시점 (전이 시작)
        onset = s
        for idx in range(s, e + 1):
            if abs(sig[idx] - lvl_before) > band:
                onset = idx
                break
        # offset knee: post-level 밴드에 안착하는 첫 시점 (전이 끝)
        offset = e
        for idx in range(e, s - 1, -1):
            if abs(sig[idx] - lvl_after) > band:
                offset = idx
                break
        if offset <= onset:
            offset = min(n - 1, onset + 1)

        # 직전 plateau 지속시간 (이전 edge offset ~ 이번 onset)
        prev_off_t = edges[-1].offset_time if edges else time[0]
        edges.append(EdgeParams(
            edge_id=eid,
            direction='rise' if amp > 0 else 'fall',
            onset_time=float(time[onset]),
            offset_time=float(time[offset]),
            onset_amp=float(sig[onset]),
            offset_amp=float(sig[offset]),
            transition_time=float(time[offset] - time[onset]),
            amplitude=float(amp),
            slope=float(amp / max(1e-6, time[offset] - time[onset])),
            plateau_before=lvl_before,
            plateau_after=lvl_after,
            plateau_before_dur=float(time[onset] - prev_off_t),
        ))
        eid += 1

    return edges


# ==================== 출력 ====================

def edges_to_df(edges: List[EdgeParams]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in edges])


def plot_annotated(time, raw, filt, edges: List[EdgeParams], title: str, out_png: Path):
    """comparison.png 스타일 + knee 빨간 점. (라벨 ASCII — 폰트 독립)"""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    ax0.plot(time, raw, lw=0.8, color='gray', alpha=0.8)
    ax0.set_title('Ch - RAW', fontsize=11)
    ax0.set_ylabel('amplitude (uV)')
    ax0.grid(True, alpha=0.3)

    ax1.plot(time, filt, lw=1.0, color='#1f77b4')
    ax1.set_title('Ch - (filtered: LP 5.0Hz) + auto knee', fontsize=11)
    ax1.set_ylabel('amplitude (uV)')
    ax1.set_xlabel('time (s)')
    ax1.grid(True, alpha=0.3)

    kx, ky = [], []
    for e in edges:
        kx += [e.onset_time, e.offset_time]
        ky += [e.onset_amp, e.offset_amp]
    ax1.plot(kx, ky, 'o', color='red', ms=6, zorder=5)

    for ax in (ax0, ax1):
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.set_xlim(time[0], time[-1])
    fig.suptitle(f'auto edge annotation - {title}', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ==================== main ====================

def run(filepath: str, channel: int = 1, lowpass: float = 5.0,
        start_time: float = 5.0, drift: bool = True, plot: bool = True,
        **det_kw):
    config = FilterConfig()
    config.lowpass_enabled = True
    config.lowpass_cutoff = lowpass
    config.drift_enabled = drift
    config.drift_method = 'detrend' if drift else 'none'
    config.start_time = start_time

    an = EAGAnalyzer(filepath, config=config)
    t_all = an.get_time_axis()
    filt_all = an.get_filtered_data()
    raw_all = an.eeg_data
    mask = t_all >= start_time
    t = t_all[mask]
    ch = channel - 1
    filt = filt_all[mask, ch]
    raw = raw_all[mask, ch]

    edges = detect_edges(t, filt, fs=an.sample_rate, **det_kw)
    df = edges_to_df(edges)

    out_dir = Path(filepath).parent
    stem = Path(filepath).stem
    csv_path = out_dir / f'{stem}_ch{channel}_edges.csv'
    df.to_csv(csv_path, index=False)
    print(f"검출 엣지: {len(edges)}개 (knee {2*len(edges)}개)")
    print(f"  → {csv_path}")

    if plot:
        png_path = out_dir / f'{stem}_ch{channel}_auto_annotated.png'
        plot_annotated(t, raw, filt, edges, f'{stem} Ch{channel}', png_path)
        print(f"  → {png_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description='plateau 전이 knee 자동 검출')
    ap.add_argument('--file', '-f', help='BrainFlow-RAW CSV (단일)')
    ap.add_argument('--dir', '-d', help='디렉터리 재귀 batch (BrainFlow-RAW_*.csv)')
    ap.add_argument('--channel', '-c', type=int, default=1, help='채널 (1-8)')
    ap.add_argument('--lowpass', '-lp', type=float, default=5.0)
    ap.add_argument('--start-time', '-st', type=float, default=5.0)
    ap.add_argument('--no-drift', action='store_true',
                    help='drift(detrend) 보정 OFF (기본 ON)')
    ap.add_argument('--min-amp', type=float, default=25.0,
                    help='최소 전이 진폭 µV (기본 25: clean 세션 real/noise gap)')
    ap.add_argument('--slope-k', type=float, default=3.0, help='slope 임계 배수')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    if args.dir:
        files = sorted(str(p) for p in Path(args.dir).rglob('BrainFlow-RAW_*.csv')
                       if 'st' not in p.stem.split('_')[-1])  # RAW 데이터만
        # 위 필터가 과하면 전체:
        files = sorted(str(p) for p in Path(args.dir).rglob('BrainFlow-RAW_*.csv'))
        print(f"batch: {len(files)}개 파일")
        for fp in files:
            try:
                run(fp, channel=args.channel, lowpass=args.lowpass,
                    start_time=args.start_time, drift=not args.no_drift, plot=not args.no_plot,
                    min_amp=args.min_amp, slope_k=args.slope_k)
            except Exception as e:
                print(f"  [SKIP] {fp}: {e}")
        return

    if not args.file:
        ap.error('--file 또는 --dir 필요')
    run(args.file, channel=args.channel, lowpass=args.lowpass,
        start_time=args.start_time, drift=not args.no_drift, plot=not args.no_plot,
        min_amp=args.min_amp, slope_k=args.slope_k)


if __name__ == '__main__':
    main()
