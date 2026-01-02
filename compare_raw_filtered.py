"""
원시 데이터와 필터링 데이터 비교 시각화 도구
eag_analyzer.py와 함께 사용
"""

import argparse
from eag_analyzer import (
    EAGAnalyzer, FilterConfig, setup_korean_font,
    CHANNEL_NAMES, CHANNEL_COLORS, EEG_CHANNELS,
    find_csv_files,
    interactive_select_data, interactive_select_drift
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def plot_comparison_all_channels(analyzer: EAGAnalyzer):
    """8채널 모두 원시 데이터와 필터링 데이터 비교 플롯

    Args:
        analyzer: EAGAnalyzer 인스턴스
    """
    setup_korean_font()

    time = analyzer.get_time_axis()
    config = analyzer.config
    raw_data = analyzer.eeg_data
    filtered_data = analyzer.get_filtered_data()

    # 시작 시간 이후 데이터만 선택
    time_mask = time >= config.start_time
    time = time[time_mask]
    raw_data = raw_data[time_mask, :]
    filtered_data = filtered_data[time_mask, :]

    # 필터 설정 표시
    parts = []
    if config.lowpass_enabled:
        parts.append(f"LP {config.lowpass_cutoff}Hz")
    if config.drift_enabled:
        parts.append(f"Drift:{config.drift_method}")
    filter_info = ', '.join(parts) if parts else "없음"

    # 8채널 x 2행 (원시/필터링) 그래프
    fig, axes = plt.subplots(EEG_CHANNELS, 2, figsize=(18, 2.5 * EEG_CHANNELS), sharex=True)

    # Y축 범위 계산: 각 채널별로 원시/필터링 통일
    for ch in range(EEG_CHANNELS):
        # 해당 채널의 원시/필터링 데이터 범위
        raw_min = np.nanmin(raw_data[:, ch])
        raw_max = np.nanmax(raw_data[:, ch])
        filt_min = np.nanmin(filtered_data[:, ch])
        filt_max = np.nanmax(filtered_data[:, ch])

        # 각 채널별 통일된 범위 (10% 마진)
        ch_min = min(raw_min, filt_min)
        ch_max = max(raw_max, filt_max)
        ch_range = ch_max - ch_min
        ylim = (ch_min - ch_range * 0.1, ch_max + ch_range * 0.1)

        # 원시 데이터 (왼쪽)
        axes[ch, 0].plot(time, raw_data[:, ch], linewidth=0.8, color='gray', alpha=0.8)
        axes[ch, 0].set_ylabel(f'{CHANNEL_NAMES[ch]}\n(µV)', fontsize=9)
        axes[ch, 0].grid(True, alpha=0.3)
        axes[ch, 0].xaxis.set_major_locator(MultipleLocator(5.0))
        axes[ch, 0].set_ylim(ylim)
        axes[ch, 0].set_xlim(time[0], time[-1])

        # 필터링 데이터 (오른쪽)
        axes[ch, 1].plot(time, filtered_data[:, ch], linewidth=0.8,
                         color=CHANNEL_COLORS[ch])
        axes[ch, 1].grid(True, alpha=0.3)
        axes[ch, 1].xaxis.set_major_locator(MultipleLocator(5.0))
        axes[ch, 1].set_ylim(ylim)
        axes[ch, 1].set_xlim(time[0], time[-1])

    # 컬럼 타이틀
    axes[0, 0].set_title('원시 데이터', fontsize=11)
    axes[0, 1].set_title(f'필터링됨 ({filter_info})', fontsize=11)

    # X축 라벨
    axes[-1, 0].set_xlabel('시간 (초)', fontsize=10)
    axes[-1, 1].set_xlabel('시간 (초)', fontsize=10)

    fig.suptitle(f'원시 vs 필터링 비교 - {analyzer.filename}', fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='원시 데이터와 필터링 데이터 비교 시각화 (8채널)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 인터랙티브 모드
  python compare_raw_filtered.py

  # 특정 파일 분석
  python compare_raw_filtered.py --file ./data/피험자/세션/BrainFlow-RAW.csv

  # 저역통과 필터 변경
  python compare_raw_filtered.py --lowpass 10

  # 드리프트 보정 OFF
  python compare_raw_filtered.py --no-drift
        """
    )

    parser.add_argument('--file', '-f', type=str, help='분석할 CSV 파일 경로')

    # Lowpass Filter
    parser.add_argument('--lowpass', '-lp', type=float, default=5.0,
                        help='저역통과 필터 차단 주파수 (기본: 5Hz)')
    parser.add_argument('--no-lowpass', action='store_true',
                        help='저역통과 필터 비활성화')

    # Drift 보정
    parser.add_argument('--no-drift', action='store_true',
                        help='드리프트 보정 비활성화')
    parser.add_argument('--drift-method', type=str, default='detrend',
                        choices=['detrend', 'moving', 'none'],
                        help='드리프트 보정 방법 (기본: detrend)')
    parser.add_argument('--drift-window', type=float, default=1.0,
                        help='드리프트 이동평균 윈도우 (초, 기본: 1.0)')

    # 표시 설정
    parser.add_argument('--start-time', '-st', type=float, default=5.0,
                        help='표시 시작 시간 (초, 기본: 5)')

    args = parser.parse_args()

    # FilterConfig 생성
    config = FilterConfig()
    config.lowpass_enabled = not args.no_lowpass
    config.lowpass_cutoff = args.lowpass
    config.drift_enabled = not args.no_drift
    config.drift_method = args.drift_method if not args.no_drift else "none"
    config.drift_window_sec = args.drift_window
    config.start_time = args.start_time

    # 파일 선택
    if args.file:
        filepath = args.file
    else:
        # 인터랙티브 모드
        filepath = interactive_select_data()
        if filepath is None:
            print("선택이 취소되었습니다.")
            return

        if not filepath.endswith('.csv'):
            # 폴더가 선택된 경우 첫 번째 CSV 파일 사용
            csv_files = find_csv_files(filepath)
            if not csv_files:
                print("CSV 파일을 찾을 수 없습니다.")
                return
            filepath = csv_files[0]
            print(f"선택된 파일: {filepath}")

        # 드리프트 보정 설정 선택
        config = interactive_select_drift(config)

    # 분석 실행
    analyzer = EAGAnalyzer(filepath, config=config)
    plot_comparison_all_channels(analyzer)


if __name__ == '__main__':
    main()
