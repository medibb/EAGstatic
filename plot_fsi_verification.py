"""
FSI/APA 검증 시각화 스크립트
- EAG 8ch + GRF 동시 표시
- GRF onset 수직선 (빨간 점선)
- FSI 마커 (★) + onset 대비 latency 표시
- GRF onset 전후 2초 윈도우 음영 (APA 검출 구간)

Usage:
    # 특정 피험자 1세션
    python3 plot_fsi_verification.py --subject 주창민 --session s1

    # 특정 피험자 전체 세션
    python3 plot_fsi_verification.py --subject 주창민

    # 전체 피험자 (batch)
    python3 plot_fsi_verification.py --batch
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sync_analyzer import (
    SyncAnalyzer, find_all_pairs, SessionPair,
    GRFEvent, EAGInflection,
)
from eag_analyzer import (
    FilterConfig, setup_korean_font,
    SAMPLE_RATE, EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS,
)
from parameter_extractor import extract_grf_event_params, extract_eag_grf_cross_params


def find_fsi_for_event(analyzer, event, ch):
    """해당 이벤트 × 채널의 FSI를 찾는다 (onset - 0.5초 이후 첫 변곡점)."""
    ch_infl = [ip for ip in analyzer.eag_inflections
               if ip.event_id == event.event_id and ip.channel == ch]
    ch_infl_sorted = sorted(ch_infl, key=lambda x: x.time)
    post_onset = [ip for ip in ch_infl_sorted
                  if ip.time >= event.onset_time - 0.5]
    return post_onset[0] if post_onset else None


def plot_fsi_verification(analyzer, save_path=None):
    """GRF onset 기준 FSI/APA 검증 그래프."""
    setup_korean_font()

    n_rows = EEG_CHANNELS + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(20, 2.5 * EEG_CHANNELS + 3),
        sharex=True,
        gridspec_kw={'height_ratios': [1] * EEG_CHANNELS + [1.3]}
    )

    # Y축 통일 range
    ranges = []
    centers = []
    for ch in range(EEG_CHANNELS):
        d = analyzer.eag_filtered[:, ch]
        ch_range = np.nanmax(d) - np.nanmin(d)
        ranges.append(ch_range)
        centers.append(np.nanmean(d))
    max_range = max(ranges)
    half = max_range * 1.1 / 2

    # --- EAG 채널들 ---
    for ch in range(EEG_CHANNELS):
        ax = axes[ch]

        # EAG 신호
        ax.plot(analyzer.unified_time_eag, analyzer.eag_filtered[:, ch],
                linewidth=0.6, color=CHANNEL_COLORS[ch], alpha=0.8)
        ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(µV)', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(centers[ch] - half, centers[ch] + half)
        ax.set_xlim(0, analyzer.overlap_duration)

        for event in analyzer.grf_events:
            # GRF onset 수직선 (빨간 점선)
            ax.axvline(event.onset_time, color='red', linestyle='--',
                       linewidth=1.0, alpha=0.7)

            # APA 윈도우: onset 전 2초 (하늘색 음영)
            apa_start = max(0, event.onset_time - 2.0)
            ax.axvspan(apa_start, event.onset_time, alpha=0.08,
                       color='#2196F3', zorder=0)

            # 이벤트 구간 (연한 주황 음영)
            ax.axvspan(event.onset_time, event.offset_time, alpha=0.12,
                       color='orange', zorder=0)

            # FSI 마커
            fsi = find_fsi_for_event(analyzer, event, ch)
            if fsi:
                latency_ms = (fsi.time - event.onset_time) * 1000
                marker_color = '#E91E63' if latency_ms < 0 else '#FF9800'
                ax.plot(fsi.time, fsi.amplitude, marker='*',
                        color=marker_color, markersize=10,
                        markeredgewidth=0.5, markeredgecolor='black',
                        zorder=5)
                # latency 텍스트 (ms)
                ax.annotate(f'{latency_ms:+.0f}ms',
                            xy=(fsi.time, fsi.amplitude),
                            xytext=(0, 8), textcoords='offset points',
                            fontsize=6, ha='center', color=marker_color,
                            fontweight='bold')

    # --- GRF ---
    ax_grf = axes[EEG_CHANNELS]
    ax_grf.plot(analyzer.unified_time_grf, analyzer.grf_left,
                color='#2196F3', linewidth=0.8, label='Left GRF')
    ax_grf.plot(analyzer.unified_time_grf, analyzer.grf_right,
                color='#F44336', linewidth=0.8, label='Right GRF')
    ax_grf.set_ylabel('GRF (kg)', fontsize=9)
    ax_grf.set_xlabel('Time (sec)', fontsize=10)
    ax_grf.grid(True, alpha=0.3)
    ax_grf.set_xlim(0, analyzer.overlap_duration)

    if analyzer.body_weight > 0:
        ax_grf.axhline(y=analyzer.body_weight / 2, color='gray',
                       linestyle=':', alpha=0.4, linewidth=0.8)

    # GRF 이벤트 onset 수직선 + 번호
    for event in analyzer.grf_events:
        ax_grf.axvline(event.onset_time, color='red', linestyle='--',
                       linewidth=1.0, alpha=0.7)
        ax_grf.axvspan(event.onset_time, event.offset_time,
                       alpha=0.12, color='orange')
        ax_grf.text(event.onset_time, ax_grf.get_ylim()[1] * 0.95,
                    f'E{event.event_id}', ha='center', va='top',
                    fontsize=8, fontweight='bold', color='red', alpha=0.8)

    ax_grf.legend(loc='upper right', fontsize=8)

    # --- 범례 ---
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.0,
               label='GRF Onset'),
        Patch(facecolor='#2196F3', alpha=0.15,
              label='APA Window (onset - 2s)'),
        Patch(facecolor='orange', alpha=0.2,
              label='Event Duration'),
        Line2D([0], [0], marker='*', color='#E91E63', linestyle='None',
               markersize=10, label='FSI (pre-onset, APA)'),
        Line2D([0], [0], marker='*', color='#FF9800', linestyle='None',
               markersize=10, label='FSI (post-onset)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right',
                   fontsize=7, framealpha=0.9, ncol=2)

    fig.suptitle(
        f'FSI / APA Verification — {analyzer.pair.subject_name}, '
        f'Session: {analyzer.pair.session_name}\n'
        f'(★ = FSI, ms = latency vs GRF onset, '
        f'pink = pre-onset APA, orange = post-onset)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    plt.close(fig)


def collect_fsi_data(analyzer):
    """세션의 FSI 데이터를 수집하여 리스트로 반환."""
    rows = []
    for event in analyzer.grf_events:
        for ch in range(EEG_CHANNELS):
            fsi = find_fsi_for_event(analyzer, event, ch)
            latency_ms = (fsi.time - event.onset_time) * 1000 if fsi else float('nan')
            rows.append({
                'subject': analyzer.pair.subject_name,
                'session': analyzer.pair.session_name,
                'event_id': event.event_id,
                'event_onset': round(event.onset_time, 4),
                'event_intensity': event.intensity,
                'channel': CHANNEL_NAMES[ch],
                'fsi_time': round(fsi.time, 4) if fsi else float('nan'),
                'fsi_latency_ms': round(latency_ms, 1) if fsi else float('nan'),
                'fsi_amplitude': round(fsi.amplitude, 2) if fsi else float('nan'),
                'fsi_direction': fsi.direction if fsi else '',
                'is_apa': latency_ms < 0 if fsi else False,
            })
    return rows


def process_session(pair, result_base='result'):
    """단일 세션 처리. FSI 데이터 리스트를 반환."""
    config = FilterConfig()
    config.lowpass_cutoff = 5.0
    config.start_time = 0.0

    try:
        analyzer = SyncAnalyzer(pair, config=config)
        analyzer.run_analysis()
    except Exception as e:
        print(f"[SKIP] {pair.subject_name}/{pair.session_name}: {e}")
        return []

    if not analyzer.grf_events:
        print(f"[SKIP] {pair.subject_name}/{pair.session_name}: 이벤트 없음")
        return []

    save_dir = os.path.join(result_base, pair.subject_name)
    save_path = os.path.join(save_dir, f'{pair.session_name}_fsi_verify.png')
    plot_fsi_verification(analyzer, save_path=save_path)

    return collect_fsi_data(analyzer)


def main():
    parser = argparse.ArgumentParser(description='FSI/APA 검증 시각화')
    parser.add_argument('--subject', type=str, default=None,
                        help='피험자 이름 (부분 매칭)')
    parser.add_argument('--session', type=str, default=None,
                        help='세션명 (e.g. s1, f2)')
    parser.add_argument('--batch', action='store_true',
                        help='전체 피험자 batch')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='데이터 디렉토리')
    parser.add_argument('--result-dir', type=str, default='result',
                        help='결과 디렉토리')
    args = parser.parse_args()

    pairs = find_all_pairs(args.data_dir)
    if not pairs:
        print(f"데이터를 찾을 수 없습니다: {args.data_dir}")
        sys.exit(1)

    # 필터링
    if args.subject:
        pairs = [p for p in pairs if args.subject in p.subject_name]
    if args.session:
        pairs = [p for p in pairs if p.session_name == args.session]

    if not pairs:
        print("매칭되는 세션이 없습니다.")
        sys.exit(1)

    print(f"처리 대상: {len(pairs)}개 세션")
    all_fsi_rows = []
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair.subject_name} / {pair.session_name}")
        rows = process_session(pair, result_base=args.result_dir)
        all_fsi_rows.extend(rows)

    # CSV 저장
    if all_fsi_rows:
        import csv
        csv_dir = os.path.join(args.result_dir, 'fsi_verification')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'fsi_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_fsi_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_fsi_rows)
        print(f"\nFSI 요약 CSV: {csv_path}")

        # 간단 통계 출력
        import numpy as np
        latencies = [r['fsi_latency_ms'] for r in all_fsi_rows
                     if not np.isnan(r['fsi_latency_ms'])]
        apa_count = sum(1 for r in all_fsi_rows if r['is_apa'])
        total = len([r for r in all_fsi_rows if not np.isnan(r['fsi_latency_ms'])])
        print(f"총 FSI: {total}개, APA(pre-onset): {apa_count}개 ({apa_count/total*100:.1f}%)")
        print(f"FSI latency: median {np.median(latencies):.1f}ms, "
              f"mean {np.mean(latencies):.1f}ms")

    print("완료.")


if __name__ == '__main__':
    main()
