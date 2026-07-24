"""
EAG-GRF 동기화 정렬 검증 시각화 (Filtered EAG)
- 4패널: EAG filtered, GRF, timestamp-aligned overlay, event-aligned overlay
- 각 피험자의 첫 세션(s1)에 대해 초기 5초 구간 시각화
- SyncAnalyzer의 실제 정렬 결과를 사용

Usage:
    python3 plot_alignment_verification.py --subject 주창민
    python3 plot_alignment_verification.py --batch
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sync_analyzer import (
    SyncAnalyzer, find_all_pairs, SessionPair,
    parse_eag_start_sod, parse_grf_start_sod,
)
from eag_analyzer import (
    EAGAnalyzer, FilterConfig, setup_korean_font,
    SAMPLE_RATE, EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS,
)
from grf_viewer import load_grf_data, extract_body_weight


def compute_eag_event_time(eag_filtered, eag_time, search_start=1.0, search_end=10.0):
    """EAG filtered 신호에서 초기 이벤트 시점을 찾는다 (채널별 max gradient의 median)."""
    mask = (eag_time >= search_start) & (eag_time <= search_end)
    candidates = []
    for ch in range(EEG_CHANNELS):
        data_ch = eag_filtered[mask, ch]
        time_ch = eag_time[mask]
        grad = np.abs(np.gradient(data_ch))
        grad_smooth = pd.Series(grad).rolling(5, center=True, min_periods=1).mean().values
        peak_idx = np.argmax(grad_smooth)
        candidates.append(time_ch[peak_idx])
    return float(np.median(candidates)), candidates


def compute_grf_onset_time(grf_df, body_weight, search_start=1.0, search_end=10.0):
    """GRF에서 초기 체중이동 이벤트 onset을 찾는다."""
    time = grf_df['time'].values - grf_df['time'].values[0]
    left = grf_df['left_grf'].values
    right = grf_df['right_grf'].values

    # Baseline (첫 1.5초)
    bl_mask = time < 1.5
    left_bl_mean = np.mean(left[bl_mask])
    left_bl_std = np.std(left[bl_mask])
    right_bl_mean = np.mean(right[bl_mask])
    right_bl_std = np.std(right[bl_mask])

    # Left rise 또는 Right rise 중 먼저 오는 것
    left_threshold = left_bl_mean + 3 * max(left_bl_std, 0.5)
    right_threshold = right_bl_mean + 3 * max(right_bl_std, 0.5)

    onset_l = None
    onset_r = None

    left_onset_mask = (time > search_start) & (left > left_threshold)
    if np.any(left_onset_mask):
        onset_l = time[np.argmax(left_onset_mask)]

    right_onset_mask = (time > search_start) & (right > right_threshold)
    if np.any(right_onset_mask):
        onset_r = time[np.argmax(right_onset_mask)]

    # imbalance 기반 fallback
    total = left + right
    safe_total = np.maximum(total, 0.1)
    imbalance = np.abs(left - right) / safe_total
    imb_mask = (time > search_start) & (time < search_end) & (imbalance > 0.18)
    if np.any(imb_mask):
        onset_imb = time[np.argmax(imb_mask)]
    else:
        onset_imb = None

    # 가장 빠른 onset 선택
    candidates = [t for t in [onset_l, onset_r, onset_imb] if t is not None]
    if candidates:
        return min(candidates)

    return None


def compute_grf_peak_time(grf_df, onset_time, window=3.0):
    """GRF onset 이후 peak imbalance 시점."""
    time = grf_df['time'].values - grf_df['time'].values[0]
    left = grf_df['left_grf'].values

    mask = (time >= onset_time) & (time <= onset_time + window)
    if not np.any(mask):
        return None
    peak_idx = np.argmax(left[mask])
    return time[mask][peak_idx]


def plot_alignment(pair, save_dir='result/alignment_check'):
    """단일 세션의 정렬 검증 4패널 그래프."""
    setup_korean_font()

    config = FilterConfig()
    config.lowpass_cutoff = 5.0
    config.start_time = 0.0

    # --- 데이터 로드 ---
    try:
        eag_analyzer = EAGAnalyzer(pair.eag_filepath, config=config)
        grf_df = load_grf_data(pair.grf_filepath)
        body_weight = extract_body_weight(pair.grf_filepath)
    except Exception as e:
        print(f"  [SKIP] 데이터 로드 실패: {e}")
        return

    # EAG filtered + time axis
    eag_filtered = eag_analyzer.get_filtered_data()
    eag_time = np.arange(eag_filtered.shape[0]) / SAMPLE_RATE

    # GRF time (relative from 0)
    grf_time = grf_df['time'].values - grf_df['time'].values[0]
    grf_left = grf_df['left_grf'].values
    grf_right = grf_df['right_grf'].values

    # --- 이벤트 시점 계산 ---
    eag_event_time, ch_times = compute_eag_event_time(eag_filtered, eag_time)
    grf_onset_time = compute_grf_onset_time(grf_df, body_weight)
    if grf_onset_time is None:
        print(f"  [SKIP] GRF onset 감지 실패")
        return
    grf_peak_time = compute_grf_peak_time(grf_df, grf_onset_time)

    # --- Offset 계산 ---
    # Timestamp-based
    eag_data_raw = pd.read_csv(pair.eag_filepath, header=None, sep='\t', quotechar='"')
    if len(eag_data_raw.columns) == 1:
        eag_data_raw = eag_data_raw.iloc[:, 0].str.split('\t', expand=True).astype(float)
    eag_timestamps = eag_data_raw.iloc[:, 22].values.astype(float)
    eag_start_sod = parse_eag_start_sod(eag_timestamps)
    grf_start_sod = parse_grf_start_sod(pair.grf_filepath)
    ts_offset = grf_start_sod - eag_start_sod

    # Event-based
    event_offset = eag_event_time - grf_onset_time

    # --- SyncAnalyzer 실제 사용 offset ---
    try:
        analyzer = SyncAnalyzer(SessionPair(
            session_dir=pair.session_dir,
            eag_filepath=pair.eag_filepath,
            grf_filepath=pair.grf_filepath,
            session_name=pair.session_name,
            subject_name=pair.subject_name,
        ), config=config)
        sync_method = "event" if hasattr(analyzer, 'sync_method') else "auto"
        actual_offset = analyzer.time_offset
    except Exception:
        actual_offset = event_offset
        sync_method = "fallback"

    # --- 5초 윈도우 (0.5초부터: filter ringing 제외) ---
    view_start = 0.5
    view_end = 5.0
    eag_5s_mask = (eag_time >= view_start) & (eag_time <= view_end)
    eag_time_5s = eag_time[eag_5s_mask]
    eag_5s = eag_filtered[eag_5s_mask]

    grf_5s_mask = grf_time <= view_end
    grf_time_5s = grf_time[grf_5s_mask]
    grf_left_5s = grf_left[grf_5s_mask]
    grf_right_5s = grf_right[grf_5s_mask]

    # --- 그래프 ---
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))

    # ===== 좌상: EAG filtered (mean-subtracted) =====
    ax = axes[0, 0]
    for ch in range(EEG_CHANNELS):
        data_ch = eag_5s[:, ch] - np.nanmean(eag_5s[:, ch])
        ax.plot(eag_time_5s, data_ch, linewidth=0.6,
                color=CHANNEL_COLORS[ch], label=CHANNEL_NAMES[ch], alpha=0.8)
    ax.axvline(x=eag_event_time, color='red', linewidth=2, linestyle='--',
               label=f'EAG event: {eag_event_time:.3f}s')
    ax.set_title('EAG Filtered (LP 5Hz + Detrend)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude (µV, mean-subtracted)', fontsize=10)
    ax.set_xlabel('EAG time (seconds)', fontsize=10)
    # Y축: percentile 기반 auto-scale (artifact 제외)
    all_vals = np.concatenate([eag_5s[:, ch] - np.nanmean(eag_5s[:, ch])
                               for ch in range(EEG_CHANNELS)])
    p2, p98 = np.nanpercentile(all_vals, [2, 98])
    margin = (p98 - p2) * 0.2
    ax.set_ylim(p2 - margin, p98 + margin)
    ax.legend(fontsize=7, ncol=5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(view_start, view_end)

    # ===== 우상: GRF =====
    ax = axes[0, 1]
    ax.plot(grf_time_5s, grf_left_5s, color='#2196F3', linewidth=1.2, label='Left GRF')
    ax.plot(grf_time_5s, grf_right_5s, color='#F44336', linewidth=1.2, label='Right GRF')
    ax.axvline(x=grf_onset_time, color='green', linewidth=2, linestyle='--',
               label=f'GRF onset: {grf_onset_time:.3f}s')
    if grf_peak_time:
        ax.axvline(x=grf_peak_time, color='darkgreen', linewidth=2, linestyle=':',
                   label=f'GRF peak: {grf_peak_time:.3f}s')
    if body_weight > 0:
        ax.axhline(y=body_weight / 2, color='gray', linestyle=':', alpha=0.4,
                   label=f'BW/2={body_weight/2:.1f}')
    ax.set_title('GRF (from GRF t=0)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GRF (kg)', fontsize=10)
    ax.set_xlabel('GRF time (seconds)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(view_start, view_end)

    # ===== 좌하: TIMESTAMP-ALIGNED =====
    ax = axes[1, 0]
    grf_time_ts = grf_time + ts_offset
    grf_ts_mask = (grf_time_ts >= -1) & (grf_time_ts <= view_end)

    for ch in range(EEG_CHANNELS):
        data_ch = eag_5s[:, ch] - np.nanmean(eag_5s[:, ch])
        std = np.nanstd(data_ch)
        if std > 0:
            ax.plot(eag_time_5s, data_ch / std * 10, linewidth=0.3,
                    color=CHANNEL_COLORS[ch], alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(grf_time_ts[grf_ts_mask], grf_left[grf_ts_mask],
             color='blue', linewidth=1.5, label='Left GRF')
    ax2.plot(grf_time_ts[grf_ts_mask], grf_right[grf_ts_mask],
             color='red', linewidth=1.5, label='Right GRF')
    ax2.set_ylabel('GRF (kg)', fontsize=10, color='blue')
    ax2.legend(loc='upper right', fontsize=8)

    ax.axvline(x=eag_event_time, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax.set_title(f'TIMESTAMP-ALIGNED (offset={ts_offset:.3f}s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('EAG (normalized)', fontsize=10)
    ax.set_xlabel('EAG time (seconds)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, view_end)

    # ===== 우하: EVENT-ALIGNED =====
    ax = axes[1, 1]
    grf_time_evt = grf_time + event_offset
    grf_evt_mask = (grf_time_evt >= -1) & (grf_time_evt <= view_end)

    for ch in range(EEG_CHANNELS):
        data_ch = eag_5s[:, ch] - np.nanmean(eag_5s[:, ch])
        std = np.nanstd(data_ch)
        if std > 0:
            ax.plot(eag_time_5s, data_ch / std * 10, linewidth=0.3,
                    color=CHANNEL_COLORS[ch], alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(grf_time_evt[grf_evt_mask], grf_left[grf_evt_mask],
             color='blue', linewidth=1.5, label='Left GRF')
    ax2.plot(grf_time_evt[grf_evt_mask], grf_right[grf_evt_mask],
             color='red', linewidth=1.5, label='Right GRF')
    ax2.set_ylabel('GRF (kg)', fontsize=10, color='blue')
    ax2.legend(loc='upper right', fontsize=8)

    ax.axvline(x=eag_event_time, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax.set_title(f'EVENT-ALIGNED (offset={event_offset:.3f}s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('EAG (normalized)', fontsize=10)
    ax.set_xlabel('EAG time (seconds)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, view_end)

    # --- 전체 제목 ---
    discrepancy = event_offset - ts_offset
    fig.suptitle(
        f'Alignment Verification — {pair.subject_name}, Session: {pair.session_name}\n'
        f'TS offset: {ts_offset:.3f}s | Event offset: {event_offset:.3f}s | '
        f'Discrepancy: {discrepancy:.3f}s | Sync used: {actual_offset:.3f}s',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             f'{pair.subject_name}_{pair.session_name}_alignment.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  저장: {save_path}")
    plt.close(fig)

    return {
        'subject': pair.subject_name,
        'session': pair.session_name,
        'ts_offset': ts_offset,
        'event_offset': event_offset,
        'discrepancy': discrepancy,
        'actual_sync_offset': actual_offset,
    }


def main():
    parser = argparse.ArgumentParser(description='EAG-GRF 정렬 검증')
    parser.add_argument('--subject', type=str, default=None)
    parser.add_argument('--session', type=str, default=None,
                        help='세션명 (기본: s1만)')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--all-sessions', action='store_true',
                        help='모든 세션 (기본: s1만)')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--save-dir', type=str, default='result/alignment_check')
    args = parser.parse_args()

    pairs = find_all_pairs(args.data_dir)
    if not pairs:
        print(f"데이터를 찾을 수 없습니다: {args.data_dir}")
        sys.exit(1)

    if args.subject:
        pairs = [p for p in pairs if args.subject in p.subject_name]
    if args.session:
        pairs = [p for p in pairs if p.session_name == args.session]
    elif not args.all_sessions:
        # 기본: 각 피험자의 s1만
        pairs = [p for p in pairs if p.session_name == 's1']

    if not pairs:
        print("매칭되는 세션이 없습니다.")
        sys.exit(1)

    print(f"처리 대상: {len(pairs)}개 세션")
    results = []
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair.subject_name} / {pair.session_name}")
        r = plot_alignment(pair, save_dir=args.save_dir)
        if r:
            results.append(r)

    # 요약 CSV
    if results:
        import csv
        csv_path = os.path.join(args.save_dir, 'alignment_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n요약 CSV: {csv_path}")

    print("완료.")


if __name__ == '__main__':
    main()
