"""
EAG-GRF Manual Offset Adjustment CLI

Usage:
    python adjust_offset.py review                          # 전체 피험자 offset 상태
    python adjust_offset.py review --subject 김은혜         # 특정 피험자

    python adjust_offset.py explore --subject 김은혜 --session s1   # offset 탐색 PNG
    python adjust_offset.py explore --subject 김은혜 --session s1 --range -2 2 --step 0.4

    python adjust_offset.py set --subject 김은혜 --session s1 --offset -0.35
    python adjust_offset.py set --subject 김은혜 --session s1 --offset -0.35 --note "visual ok"

    python adjust_offset.py clear --subject 김은혜 --session s1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sync_analyzer import (
    SyncAnalyzer, find_all_pairs, SessionPair,
    parse_eag_start_sod, parse_grf_start_sod,
)
from eag_analyzer import (
    EAGAnalyzer, FilterConfig, EMGFilter, setup_korean_font,
    SAMPLE_RATE, EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS,
)
from grf_viewer import load_grf_data, extract_body_weight
from offset_manager import (
    get_manual_offset, set_manual_offset, clear_manual_offset,
    load_manual_offsets, list_all_offsets,
)


# ==================== review ====================

def cmd_review(args):
    """피험자별 offset 상태 테이블 출력."""
    pairs = find_all_pairs(args.data_dir)
    if args.subject:
        pairs = [p for p in pairs if args.subject in p.subject_name]

    # s1만
    pairs_s1 = [p for p in pairs if p.session_name == 's1']
    if not pairs_s1:
        print("매칭되는 피험자가 없습니다.")
        return

    manual_offsets = load_manual_offsets()

    print(f"\n{'Subject':<30} {'Method':<8} {'Auto Offset':>12} {'Manual':>8} {'Status'}")
    print('-' * 80)

    for pair in pairs_s1:
        config = FilterConfig()
        config.lowpass_cutoff = 5.0
        config.start_time = 0.0

        # auto offset 계산 (manual 무시)
        try:
            analyzer = SyncAnalyzer(pair, config=config, manual_offset=float('inf'))
        except:
            # manual_offset=inf로는 안되므로, 직접 계산
            pass

        # auto alignment 실행
        try:
            analyzer = SyncAnalyzer.__new__(SyncAnalyzer)
            analyzer.pair = pair
            analyzer.config = config
            analyzer.config.start_time = 0.0
            analyzer.utc_offset = 9
            analyzer._manual_offset = None  # auto만
            analyzer.eag = EAGAnalyzer(pair.eag_filepath, config=config)
            analyzer.grf_df = load_grf_data(pair.grf_filepath)
            analyzer.body_weight = extract_body_weight(pair.grf_filepath)
            analyzer.time_offset = 0.0
            analyzer.overlap_start = 0.0
            analyzer.overlap_duration = 0.0
            analyzer.unified_time_eag = None
            analyzer.unified_time_grf = None
            analyzer.eag_filtered = None
            analyzer.grf_left = None
            analyzer.grf_right = None

            # auto alignment만 실행
            analyzer.sync_method = "unknown"
            analyzer._xcorr_confidence = 0.0
            event_ok = False
            try:
                eag_event_t = analyzer._detect_initial_event_eag(analyzer.eag.eeg_data)
                grf_event_t = analyzer._detect_initial_event_grf(analyzer.grf_df, analyzer.body_weight)
                offset_event = eag_event_t - grf_event_t
                if abs(offset_event) <= 2.0:
                    analyzer.time_offset = offset_event
                    analyzer.sync_method = "event"
                    event_ok = True
            except ValueError:
                pass
            if not event_ok:
                offset_xcorr = analyzer._sync_by_xcorr()
                analyzer.time_offset = offset_xcorr
                analyzer.sync_method = "xcorr"

            auto_offset = analyzer.time_offset
            auto_method = analyzer.sync_method
        except Exception as e:
            auto_offset = float('nan')
            auto_method = "error"

        # manual offset 확인
        manual = get_manual_offset(pair.subject_name, 's1')
        manual_str = f"{manual:+.3f}" if manual is not None else "-"

        # 상태 판정
        flags = []
        if auto_method == "xcorr":
            flags.append("⚠ xcorr")
        if abs(auto_offset) > 2.0:
            flags.append("⚠ large")
        if manual is not None:
            flags.append("✓ manual")
        status = " | ".join(flags) if flags else "ok"

        print(f"{pair.subject_name:<30} {auto_method:<8} {auto_offset:>+12.3f} {manual_str:>8} {status}")

    print()


# ==================== explore ====================

def cmd_explore(args):
    """Multi-panel offset 탐색 PNG 생성."""
    setup_korean_font()

    pairs = find_all_pairs(args.data_dir)
    matching = [p for p in pairs
                if args.subject in p.subject_name and p.session_name == args.session]
    if not matching:
        print(f"세션을 찾을 수 없습니다: {args.subject} / {args.session}")
        return
    pair = matching[0]

    config = FilterConfig()
    config.lowpass_cutoff = 5.0
    config.start_time = 0.0

    # EAG + GRF 독립 로드
    eag_analyzer = EAGAnalyzer(pair.eag_filepath, config=config)
    eag_filtered = eag_analyzer.get_filtered_data()
    eag_time = np.arange(eag_filtered.shape[0]) / SAMPLE_RATE

    grf_df = load_grf_data(pair.grf_filepath)
    grf_time = grf_df['time'].values - grf_df['time'].values[0]
    grf_left = grf_df['left_grf'].values
    grf_right = grf_df['right_grf'].values

    # Auto offset 계산 (기준점)
    try:
        analyzer = SyncAnalyzer(pair, config=config, manual_offset=None)
        center_offset = analyzer.time_offset
        auto_method = analyzer.sync_method
    except:
        center_offset = 0.0
        auto_method = "unknown"

    # Offset 범위
    if args.range:
        off_min, off_max = args.range
    else:
        # xcorr이면 넓은 범위
        if auto_method == "xcorr" or abs(center_offset) > 2.0:
            off_min, off_max = center_offset - 3.0, center_offset + 3.0
        else:
            off_min, off_max = center_offset - 1.0, center_offset + 1.0

    step = args.step or 0.2
    offsets = np.arange(off_min, off_max + step / 2, step)
    n_panels = len(offsets)

    # 표시 구간 (0.5 ~ 6초 of EAG time)
    view_start, view_end = 0.5, 6.0

    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 2.5 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    labels = [chr(65 + i) if i < 26 else str(i) for i in range(n_panels)]

    for i, (offset, label) in enumerate(zip(offsets, labels)):
        ax = axes[i]

        # EAG (normalized, all 8ch overlaid)
        eag_mask = (eag_time >= view_start) & (eag_time <= view_end)
        for ch in range(EEG_CHANNELS):
            data = eag_filtered[eag_mask, ch]
            data_norm = data - np.nanmean(data)
            std = np.nanstd(data_norm)
            if std > 0:
                data_norm = data_norm / std * 10
            ax.plot(eag_time[eag_mask], data_norm, linewidth=0.4,
                    color=CHANNEL_COLORS[ch], alpha=0.6)

        # GRF (twin axis, aligned with this offset)
        grf_aligned = grf_time + offset
        grf_mask = (grf_aligned >= view_start) & (grf_aligned <= view_end)

        ax2 = ax.twinx()
        if np.any(grf_mask):
            ax2.plot(grf_aligned[grf_mask], grf_left[grf_mask],
                     color='blue', linewidth=1.5, alpha=0.8)
            ax2.plot(grf_aligned[grf_mask], grf_right[grf_mask],
                     color='red', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('GRF', fontsize=8, color='blue')

        # 라벨
        is_auto = abs(offset - center_offset) < step / 4
        title_color = 'green' if is_auto else 'black'
        marker = " ← AUTO" if is_auto else ""
        ax.set_ylabel(f'[{label}]\n{offset:+.2f}s', fontsize=10,
                      fontweight='bold', color=title_color)
        if is_auto:
            ax.set_facecolor('#f0fff0')

        ax.grid(True, alpha=0.2)
        ax.set_xlim(view_start, view_end)

    axes[-1].set_xlabel('EAG time (seconds)', fontsize=10)

    # 범례 (첫 패널에)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1.5, label='Left GRF'),
        Line2D([0], [0], color='red', linewidth=1.5, label='Right GRF'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.suptitle(
        f'Offset Exploration — {pair.subject_name}, Session: {pair.session_name}\n'
        f'Auto method: {auto_method}, Auto offset: {center_offset:+.3f}s\n'
        f'Select the panel where EAG and GRF events align best',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    save_dir = os.path.join(args.save_dir, 'alignment_check')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             f'{pair.subject_name}_{pair.session_name}_offset_explore.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"저장: {save_path}")
    print(f"\n최적 패널을 확인 후 실행:")
    print(f"  python adjust_offset.py set --subject {args.subject} "
          f"--session {args.session} --offset <선택한 offset 값>")


# ==================== set ====================

def cmd_set(args):
    """Manual offset을 JSON에 저장."""
    pairs = find_all_pairs(args.data_dir)
    matching = [p for p in pairs
                if args.subject in p.subject_name and p.session_name == args.session]
    if not matching:
        print(f"세션을 찾을 수 없습니다: {args.subject} / {args.session}")
        return
    pair = matching[0]

    # Auto offset 조회 (기록용)
    config = FilterConfig()
    config.lowpass_cutoff = 5.0
    config.start_time = 0.0
    try:
        # auto로 한 번 실행 (manual 무시)
        temp = SyncAnalyzer.__new__(SyncAnalyzer)
        temp.pair = pair
        temp.config = config
        temp.config.start_time = 0.0
        temp.utc_offset = 9
        temp._manual_offset = None
        temp.eag = EAGAnalyzer(pair.eag_filepath, config=config)
        temp.grf_df = load_grf_data(pair.grf_filepath)
        temp.body_weight = extract_body_weight(pair.grf_filepath)
        temp.time_offset = 0.0
        temp.overlap_start = 0.0
        temp.overlap_duration = 0.0
        temp.unified_time_eag = None
        temp.unified_time_grf = None
        temp.eag_filtered = None
        temp.grf_left = None
        temp.grf_right = None
        temp.sync_method = "unknown"
        temp._xcorr_confidence = 0.0
        event_ok = False
        try:
            eag_t = temp._detect_initial_event_eag(temp.eag.eeg_data)
            grf_t = temp._detect_initial_event_grf(temp.grf_df, temp.body_weight)
            off = eag_t - grf_t
            if abs(off) <= 2.0:
                temp.time_offset = off
                temp.sync_method = "event"
                event_ok = True
        except ValueError:
            pass
        if not event_ok:
            temp.time_offset = temp._sync_by_xcorr()
            temp.sync_method = "xcorr"
        auto_offset = temp.time_offset
        auto_method = temp.sync_method
    except:
        auto_offset = 0.0
        auto_method = "unknown"

    # 검증: 제안된 offset으로 overlap 확인
    try:
        test = SyncAnalyzer(pair, config=config, manual_offset=args.offset)
        overlap = test.overlap_duration
    except ValueError as e:
        print(f"[ERROR] offset {args.offset:+.3f}s → 중첩 구간 없음. 다른 값을 시도하세요.")
        return

    set_manual_offset(
        subject=pair.subject_name,
        session=args.session,
        offset=args.offset,
        auto_offset=auto_offset,
        auto_method=auto_method,
        note=args.note or "",
    )

    print(f"\n✓ Manual offset 저장 완료:")
    print(f"  피험자: {pair.subject_name}")
    print(f"  세션: {args.session}")
    print(f"  Manual offset: {args.offset:+.3f}s")
    print(f"  Auto offset: {auto_offset:+.3f}s ({auto_method})")
    print(f"  Overlap: {overlap:.1f}s")
    if args.note:
        print(f"  Note: {args.note}")
    print(f"\n검증 명령:")
    print(f"  python plot_alignment_verification.py --subject {args.subject} --session {args.session}")


# ==================== clear ====================

def cmd_clear(args):
    """Manual offset 제거."""
    pairs = find_all_pairs(args.data_dir)
    matching = [p for p in pairs
                if args.subject in p.subject_name and p.session_name == args.session]
    if not matching:
        print(f"세션을 찾을 수 없습니다: {args.subject} / {args.session}")
        return
    pair = matching[0]

    if clear_manual_offset(pair.subject_name, args.session):
        print(f"✓ Manual offset 제거: {pair.subject_name} / {args.session}")
        print(f"  Auto alignment으로 복귀합니다.")
    else:
        print(f"해당 세션에 manual offset이 없습니다.")


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(description='EAG-GRF Manual Offset Adjustment')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--save-dir', type=str, default='result')
    sub = parser.add_subparsers(dest='command')

    # review
    p_review = sub.add_parser('review', help='Offset 상태 테이블')
    p_review.add_argument('--subject', type=str, default=None)

    # explore
    p_explore = sub.add_parser('explore', help='Multi-panel offset 탐색 PNG')
    p_explore.add_argument('--subject', type=str, required=True)
    p_explore.add_argument('--session', type=str, required=True)
    p_explore.add_argument('--range', type=float, nargs=2, default=None,
                           metavar=('MIN', 'MAX'))
    p_explore.add_argument('--step', type=float, default=None)

    # set
    p_set = sub.add_parser('set', help='Manual offset 저장')
    p_set.add_argument('--subject', type=str, required=True)
    p_set.add_argument('--session', type=str, required=True)
    p_set.add_argument('--offset', type=float, required=True)
    p_set.add_argument('--note', type=str, default='')

    # clear
    p_clear = sub.add_parser('clear', help='Manual offset 제거')
    p_clear.add_argument('--subject', type=str, required=True)
    p_clear.add_argument('--session', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'review':
        cmd_review(args)
    elif args.command == 'explore':
        cmd_explore(args)
    elif args.command == 'set':
        cmd_set(args)
    elif args.command == 'clear':
        cmd_clear(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
