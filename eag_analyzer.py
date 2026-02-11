"""
OpenBCI EAG (Electrodermal Activity Graph) Data Analyzer
OpenBCI BrainFlow RAW CSV 데이터를 분석하고 시각화하는 도구
- EMG(근전도) 노이즈 제거 최적화
- 8채널 개별/통합 시각화

BrainFlow RAW 데이터 형식 (24 columns):
- Column 0: Sample Index
- Columns 1-8: EEG Channels (8 channels) - µV
- Columns 9-11: Accelerometer (X, Y, Z)
- Column 12: Package Counter
- Columns 13-20: Digital/Analog Aux channels
- Column 22: Timestamp (Unix timestamp)
- Column 23: Other marker
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt, detrend
from pathlib import Path
from typing import Optional, List, Tuple
import argparse


# ==================== 한글 폰트 설정 ====================
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        font_path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    except Exception:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


# ==================== 설정 상수 ====================
SAMPLE_RATE = 250  # Hz (OpenBCI Cyton 보드 기본 샘플링 레이트)
EEG_CHANNELS = 8   # EEG 채널 수
CHANNEL_NAMES = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
CHANNEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# ==================== 필터 설정 (사용자 조절 가능) ====================
class FilterConfig:
    """필터 설정 클래스 - 파라미터를 쉽게 조절할 수 있음"""

    def __init__(self):
        # ---- Lowpass Filter (EMG 제거) ----
        self.lowpass_enabled: bool = True
        self.lowpass_cutoff: float = 5.0  # Hz (EMG 제거용)
        self.lowpass_order: int = 5

        # ---- Drift 보정 ----
        self.drift_enabled: bool = True  # ON/OFF 선택
        self.drift_method: str = "detrend"  # "detrend", "moving", "none"
        self.drift_window_sec: float = 1.0  # 이동평균 윈도우 크기 (초) - moving 방식에서 사용

        # ---- 표시 설정 ----
        self.start_time: float = 5.0  # 표시 시작 시간 (초) - 초기 불안정 구간 제외
        self.y_tick_interval: float = 1500  # Y축 눈금 간격

    def __str__(self):
        """현재 설정 출력"""
        lines = [
            "=== 필터 설정 ===",
            f"Lowpass Filter: {'ON' if self.lowpass_enabled else 'OFF'}",
            f"  - Cutoff: {self.lowpass_cutoff} Hz",
            f"  - Order: {self.lowpass_order}",
            f"Drift 보정: {'ON' if self.drift_enabled else 'OFF'}",
            f"  - Method: {self.drift_method}",
            f"  - Window (moving): {self.drift_window_sec} sec",
            f"Start Time: {self.start_time} sec",
        ]
        return "\n".join(lines)

    def copy(self):
        """설정 복사"""
        new_config = FilterConfig()
        new_config.lowpass_enabled = self.lowpass_enabled
        new_config.lowpass_cutoff = self.lowpass_cutoff
        new_config.lowpass_order = self.lowpass_order
        new_config.drift_enabled = self.drift_enabled
        new_config.drift_method = self.drift_method
        new_config.drift_window_sec = self.drift_window_sec
        new_config.start_time = self.start_time
        new_config.y_tick_interval = self.y_tick_interval
        return new_config


# 기본 필터 설정 (전역)
DEFAULT_FILTER_CONFIG = FilterConfig()


# ==================== 필터 함수들 ====================
class EMGFilter:
    """EMG(근전도) 노이즈 제거를 위한 필터 클래스"""

    @staticmethod
    def butter_lowpass(cutoff: float, fs: int, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """버터워스 저역통과 필터 계수 생성

        Args:
            cutoff: 차단 주파수 (Hz) - EMG 제거를 위해 보통 10-20Hz 사용
            fs: 샘플링 레이트 (Hz)
            order: 필터 차수
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    @staticmethod
    def apply_lowpass(data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
        """저역통과 필터 적용 - EMG 고주파 성분 제거

        EMG 신호는 주로 20-500Hz 대역에 존재하므로,
        저역통과 필터로 10-20Hz 이상의 신호를 제거하면 EMG를 효과적으로 제거
        """
        b, a = EMGFilter.butter_lowpass(cutoff, fs, order)

        # 데이터 전처리
        if isinstance(data, pd.Series):
            data = pd.to_numeric(data, errors='coerce')
            data = data.ffill().bfill().to_numpy(dtype=float)
        else:
            data = np.array(data, dtype=float)

        # NaN 처리
        if np.any(np.isnan(data)):
            data = pd.Series(data).ffill().bfill().to_numpy(dtype=float)

        # 양방향 필터링 (위상 왜곡 방지)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def remove_drift(data: np.ndarray, method: str = "detrend",
                     window_sec: float = 1.0, fs: int = 250) -> np.ndarray:
        """드리프트(기저선 변동) 제거

        Args:
            data: 입력 신호
            method: 'detrend' (선형 추세 제거) 또는 'moving' (이동평균 기저선 제거)
            window_sec: 이동평균 윈도우 크기 (초)
            fs: 샘플링 레이트
        """
        data = np.array(data, dtype=float)

        if method == "detrend":
            return detrend(data)
        elif method == "moving":
            window = max(int(window_sec * fs), 1)
            baseline = pd.Series(data).rolling(window, center=True, min_periods=1).mean()
            return data - baseline.to_numpy()
        else:
            return data

    @staticmethod
    def apply_bandpass(data: np.ndarray, lowcut: float, highcut: float,
                       fs: int, order: int = 4) -> np.ndarray:
        """대역통과 필터 적용

        Args:
            lowcut: 하한 주파수
            highcut: 상한 주파수
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)


class EAGAnalyzer:
    """OpenBCI EAG 데이터 분석기 - EMG 노이즈 제거 최적화"""

    def __init__(self, filepath: str, sample_rate: int = SAMPLE_RATE,
                 config: Optional[FilterConfig] = None):
        """
        Args:
            filepath: CSV 파일 경로
            sample_rate: 샘플링 레이트 (Hz)
            config: 필터 설정 (None이면 기본값 사용)
        """
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.data = None
        self.eeg_data = None
        self.timestamps = None
        self.accel_data = None
        self.filename = Path(filepath).stem
        self.emg_filter = EMGFilter()

        # 필터 설정
        self.config = config if config is not None else DEFAULT_FILTER_CONFIG.copy()

        # 필터링된 데이터 캐시
        self._filtered_cache = None
        self._filter_params = None

        self._load_data()

    def _load_data(self):
        """CSV 데이터 로드"""
        # 파일 읽기 시도 - 다양한 형식 지원
        try:
            # 먼저 일반적인 방식 시도
            self.data = pd.read_csv(
                self.filepath,
                header=None,
                sep='\t',
                quotechar='"'
            )

            # 컬럼이 1개만 있으면 따옴표 안에 전체가 들어있는 형식
            if len(self.data.columns) == 1:
                # 따옴표 제거 후 탭으로 분리
                self.data = self.data.iloc[:, 0].str.split('\t', expand=True)
                self.data = self.data.astype(float)
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            raise

        # EEG 데이터 추출 (컬럼 1-8)
        self.eeg_data = self.data.iloc[:, 1:9].values.astype(float)

        # 가속도계 데이터 추출 (컬럼 9-11)
        self.accel_data = self.data.iloc[:, 9:12].values.astype(float)

        # 타임스탬프 추출 (컬럼 22)
        self.timestamps = self.data.iloc[:, 22].values

        # 샘플 인덱스 (컬럼 0)
        self.sample_index = self.data.iloc[:, 0].values

        print(f"데이터 로드 완료: {len(self.data)} 샘플")
        print(f"녹화 시간: {len(self.data) / self.sample_rate:.2f} 초")

    def get_time_axis(self) -> np.ndarray:
        """시간 축 생성 (초 단위)"""
        return np.arange(len(self.eeg_data)) / self.sample_rate

    def get_filtered_data(self, config: Optional[FilterConfig] = None) -> np.ndarray:
        """EMG 노이즈가 제거된 필터링 데이터 반환

        Args:
            config: 필터 설정 (None이면 self.config 사용)

        Returns:
            필터링된 8채널 데이터 (samples x channels)
        """
        if config is None:
            config = self.config

        # 캐시 확인용 파라미터 튜플
        params = (
            config.lowpass_enabled, config.lowpass_cutoff, config.lowpass_order,
            config.drift_enabled, config.drift_method, config.drift_window_sec
        )

        if self._filtered_cache is not None and self._filter_params == params:
            return self._filtered_cache

        filtered_data = np.zeros_like(self.eeg_data)

        for ch in range(EEG_CHANNELS):
            data = self.eeg_data[:, ch].copy()

            # 1. 저역통과 필터 (EMG 고주파 성분 제거)
            if config.lowpass_enabled:
                data = self.emg_filter.apply_lowpass(
                    data, config.lowpass_cutoff, self.sample_rate, config.lowpass_order
                )

            # 2. 드리프트 제거
            if config.drift_enabled and config.drift_method != "none":
                data = self.emg_filter.remove_drift(
                    data, method=config.drift_method,
                    window_sec=config.drift_window_sec, fs=self.sample_rate
                )

            filtered_data[:, ch] = data

        # 캐시 저장
        self._filtered_cache = filtered_data
        self._filter_params = params

        return filtered_data

    def _get_unified_ylim(self, data: np.ndarray, margin: float = 0.1) -> Tuple[float, float]:
        """모든 채널에 대해 통일된 Y축 범위 계산"""
        y_min = np.nanmin(data)
        y_max = np.nanmax(data)
        y_range = y_max - y_min
        return (y_min - y_range * margin, y_max + y_range * margin)

    def _get_filter_title_suffix(self, config: FilterConfig) -> str:
        """필터 설정을 기반으로 제목 접미사 생성"""
        parts = []
        if config.lowpass_enabled:
            parts.append(f"LP {config.lowpass_cutoff}Hz")
        if config.drift_enabled:
            parts.append(f"Drift:{config.drift_method}")

        if parts:
            return f"(필터링됨: {', '.join(parts)})"
        return "(원시 데이터)"

    def plot_individual_channels(self, use_filtered: bool = True,
                                  save_path: Optional[str] = None):
        """8개 채널을 각각 개별 그래프로 시각화

        Args:
            use_filtered: 필터링된 데이터 사용 여부
            save_path: 저장 경로
        """
        setup_korean_font()

        time = self.get_time_axis()
        config = self.config

        if use_filtered:
            data = self.get_filtered_data()
            title_suffix = self._get_filter_title_suffix(config)
        else:
            data = self.eeg_data
            title_suffix = "(원시 데이터)"

        # 시작 시간 이후 데이터만 선택
        time_mask = time >= config.start_time
        time = time[time_mask]
        data = data[time_mask, :]

        # Y축 범위 계산: 각 채널의 range 중 최대값 + 10% 마진을 통일된 범위로 사용
        channel_ranges = []
        channel_centers = []
        for ch in range(EEG_CHANNELS):
            ch_min = np.nanmin(data[:, ch])
            ch_max = np.nanmax(data[:, ch])
            channel_ranges.append(ch_max - ch_min)
            channel_centers.append((ch_max + ch_min) / 2)

        max_range = max(channel_ranges)
        unified_half_range = max_range * 1.1 / 2  # 10% 마진 포함

        fig, axes = plt.subplots(EEG_CHANNELS, 1, figsize=(14, 2.5 * EEG_CHANNELS), sharex=True)

        for ch in range(EEG_CHANNELS):
            ax = axes[ch]
            ax.plot(time, data[:, ch], linewidth=1.0, color=CHANNEL_COLORS[ch],
                    label=CHANNEL_NAMES[ch])

            ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(µV)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            # X축 눈금
            ax.xaxis.set_major_locator(MultipleLocator(5.0))

            # Y축 범위: 각 채널의 중심값 기준으로 통일된 범위 적용
            center = channel_centers[ch]
            ax.set_ylim(center - unified_half_range, center + unified_half_range)

            ax.set_xlim(time[0], time[-1])

        axes[-1].set_xlabel('시간 (초)', fontsize=11)
        fig.suptitle(f'EAG 8채널 개별 신호 {title_suffix}\n{self.filename}', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프 저장: {save_path}")

        plt.show()

    def plot_all_channels_overlay(self, use_filtered: bool = True,
                                   save_path: Optional[str] = None):
        """8개 채널을 하나의 그래프에 오버레이하여 시각화

        Args:
            use_filtered: 필터링된 데이터 사용 여부
            save_path: 저장 경로
        """
        setup_korean_font()

        time = self.get_time_axis()
        config = self.config

        if use_filtered:
            data = self.get_filtered_data()
            title_suffix = self._get_filter_title_suffix(config)
        else:
            data = self.eeg_data
            title_suffix = "(원시 데이터)"

        # 시작 시간 이후 데이터만 선택
        time_mask = time >= config.start_time
        time = time[time_mask]
        data = data[time_mask, :]

        fig, ax = plt.subplots(figsize=(14, 8))

        for ch in range(EEG_CHANNELS):
            ax.plot(time, data[:, ch], linewidth=0.8, color=CHANNEL_COLORS[ch],
                    label=CHANNEL_NAMES[ch], alpha=0.8)

        ax.set_xlabel('시간 (초)', fontsize=11)
        ax.set_ylabel('진폭 (µV)', fontsize=11)
        ax.set_title(f'EAG 8채널 통합 신호 {title_suffix}\n{self.filename}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=4)

        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_major_locator(MultipleLocator(config.y_tick_interval))
        ax.set_xlim(time[0], time[-1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프 저장: {save_path}")

        plt.show()

    def generate_report(self, output_dir: Optional[str] = None):
        """분석 리포트 생성 및 모든 그래프 저장

        Args:
            output_dir: 출력 디렉토리
        """
        if output_dir is None:
            # result/피험자/ 구조로 저장
            # OpenBCISession 폴더의 바로 상위를 피험자 폴더로 인식
            result_base = Path(get_result_dir())
            filepath = Path(self.filepath)
            parts = filepath.parts

            subject_name = None
            for i, part in enumerate(parts):
                if part.startswith('OpenBCISession'):
                    if i >= 1:
                        subject_name = parts[i - 1]
                    break

            if subject_name:
                output_dir = result_base / subject_name
            else:
                output_dir = result_base / 'unknown'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.filename
        config = self.config

        print(f"\n{'='*60}")
        print(f"분석 리포트 생성: {base_name}")
        print(f"{'='*60}\n")

        # 1. 기본 정보
        print("1. 데이터 기본 정보:")
        print(f"   - 샘플 수: {len(self.data)}")
        print(f"   - 녹화 시간: {len(self.data) / self.sample_rate:.2f} 초")
        print(f"   - 샘플링 레이트: {self.sample_rate} Hz")
        print(f"   - EEG 채널 수: {EEG_CHANNELS}")

        # 2. 필터 설정
        print(f"\n2. 필터 설정:")
        print(str(config))

        # 3. 그래프 생성
        print("\n3. 그래프 생성 중...")

        # 개별 채널 그래프
        self.plot_individual_channels(
            use_filtered=True,
            save_path=str(output_dir / f'{base_name}_individual_channels.png')
        )

        # 통합 오버레이 그래프
        self.plot_all_channels_overlay(
            use_filtered=True,
            save_path=str(output_dir / f'{base_name}_overlay.png')
        )

        print(f"\n{'='*60}")
        print(f"분석 완료. 결과 저장 위치: {output_dir}")
        print(f"{'='*60}\n")


def find_csv_files(base_dir: str) -> List[str]:
    """하위 폴더에서 모든 OpenBCI CSV 파일 찾기"""
    pattern = os.path.join(base_dir, '**', 'BrainFlow-RAW*.csv')
    files = glob.glob(pattern, recursive=True)
    return files


def get_data_dir() -> str:
    """data 폴더 경로 반환"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'data')


def get_result_dir() -> str:
    """result 폴더 경로 반환"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'result')


def list_subjects(data_dir: str) -> List[Tuple[str, str]]:
    """data 폴더 내 피험자 폴더 목록 반환 (하위 폴더까지 재귀 검색)

    Returns:
        List of tuples: (표시명, 실제 경로)
    """
    if not os.path.exists(data_dir):
        return []

    subjects = []

    def find_subjects_recursive(current_dir: str, prefix: str = ""):
        """재귀적으로 피험자 폴더 찾기 (OpenBCISession 폴더를 포함하는 폴더)"""
        try:
            items = os.listdir(current_dir)
        except PermissionError:
            return

        for item in items:
            if item.startswith('.'):
                continue
            item_path = os.path.join(current_dir, item)
            if not os.path.isdir(item_path):
                continue

            # OpenBCISession 폴더가 있는지 확인 (피험자 폴더)
            has_session = any(
                sub.startswith('OpenBCISession')
                for sub in os.listdir(item_path)
                if os.path.isdir(os.path.join(item_path, sub))
            )

            if has_session:
                # 피험자 폴더 발견
                display_name = f"{prefix}{item}" if prefix else item
                subjects.append((display_name, item_path))
            else:
                # 하위 폴더 더 검색
                new_prefix = f"{prefix}{item}/" if prefix else f"{item}/"
                find_subjects_recursive(item_path, new_prefix)

    find_subjects_recursive(data_dir)
    return sorted(subjects, key=lambda x: x[0])


def list_sessions(subject_dir: str) -> List[str]:
    """피험자 폴더 내 세션(녹화) 폴더 목록 반환"""
    if not os.path.exists(subject_dir):
        return []

    sessions = []
    for item in os.listdir(subject_dir):
        item_path = os.path.join(subject_dir, item)
        if os.path.isdir(item_path) and item.startswith('OpenBCISession'):
            sessions.append(item)

    return sorted(sessions)


def interactive_select_drift(config: FilterConfig) -> FilterConfig:
    """인터랙티브하게 드리프트 보정 설정 선택

    Args:
        config: 현재 필터 설정

    Returns:
        업데이트된 필터 설정
    """
    print("\n" + "="*60)
    print("=== 드리프트 보정 설정 ===")
    print("="*60)
    print("  0: OFF (드리프트 보정 비활성화)")
    print("  1: ON - Detrend (선형 추세 제거, 권장)")
    print("  2: ON - Moving Average (이동평균 기저선 제거)")
    print("="*60)

    while True:
        choice = input("드리프트 보정 옵션을 선택하세요 (0-2) [기본: 1]: ").strip()
        if choice == '':
            choice = '1'  # 기본값: detrend

        if choice == '0':
            config.drift_enabled = False
            config.drift_method = "none"
            print("→ 드리프트 보정: OFF")
            break
        elif choice == '1':
            config.drift_enabled = True
            config.drift_method = "detrend"
            print("→ 드리프트 보정: ON (Detrend)")
            break
        elif choice == '2':
            config.drift_enabled = True
            config.drift_method = "moving"
            # 윈도우 크기 추가 입력
            window_input = input(f"이동평균 윈도우 크기 (초) [기본: {config.drift_window_sec}]: ").strip()
            if window_input:
                try:
                    config.drift_window_sec = float(window_input)
                except ValueError:
                    print("잘못된 입력. 기본값 사용.")
            print(f"→ 드리프트 보정: ON (Moving Average, 윈도우: {config.drift_window_sec}초)")
            break
        else:
            print("0, 1, 2 중에서 선택하세요.")

    return config


def interactive_select_data() -> Optional[str]:
    """인터랙티브하게 분석할 데이터 선택

    Returns:
        선택된 CSV 파일 경로 또는 폴더 경로 (None이면 취소)
    """
    data_dir = get_data_dir()

    if not os.path.exists(data_dir):
        print(f"data 폴더가 없습니다: {data_dir}")
        return None

    # 1. 피험자 선택
    subjects = list_subjects(data_dir)  # List of (display_name, path)
    if not subjects:
        print("data 폴더에 피험자 데이터가 없습니다.")
        return None

    print("\n" + "="*60)
    print("=== 피험자 선택 ===")
    print("="*60)
    for i, (display_name, _) in enumerate(subjects):
        print(f"  {i}: {display_name}")
    print(f"  a: 전체 분석 (모든 피험자)")
    print("="*60)

    while True:
        choice = input(f"피험자 번호를 선택하세요 (0-{len(subjects)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return data_dir
        try:
            idx = int(choice)
            if 0 <= idx < len(subjects):
                break
            print(f"0-{len(subjects)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    subject_name, subject_dir = subjects[idx]

    # 2. 세션 선택
    sessions = list_sessions(subject_dir)
    if not sessions:
        print(f"{os.path.basename(subject_dir)}에 세션 데이터가 없습니다.")
        return None

    print("\n" + "="*60)
    print(f"=== 세션 선택 ({os.path.basename(subject_dir)}) ===")
    print("="*60)
    for i, session in enumerate(sessions):
        # 세션 폴더 내 CSV 파일 확인
        csv_files = find_csv_files(os.path.join(subject_dir, session))
        csv_count = len(csv_files)
        print(f"  {i}: {session} ({csv_count}개 파일)")
    print(f"  a: 전체 분석 (이 피험자의 모든 세션)")
    print("="*60)

    while True:
        choice = input(f"세션 번호를 선택하세요 (0-{len(sessions)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return subject_dir
        try:
            idx = int(choice)
            if 0 <= idx < len(sessions):
                break
            print(f"0-{len(sessions)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    session_name = sessions[idx]
    session_dir = os.path.join(subject_dir, session_name)

    # 3. CSV 파일 선택
    csv_files = find_csv_files(session_dir)
    if not csv_files:
        print(f"{session_name}에 CSV 파일이 없습니다.")
        return None

    if len(csv_files) == 1:
        # 파일이 하나면 바로 선택
        print(f"\n선택된 파일: {csv_files[0]}")
        return csv_files[0]

    print("\n" + "="*60)
    print(f"=== CSV 파일 선택 ({session_name}) ===")
    print("="*60)
    for i, csv_file in enumerate(csv_files):
        filename = os.path.basename(csv_file)
        print(f"  {i}: {filename}")
    print(f"  a: 전체 분석 (이 세션의 모든 파일)")
    print("="*60)

    while True:
        choice = input(f"파일 번호를 선택하세요 (0-{len(csv_files)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return session_dir
        try:
            idx = int(choice)
            if 0 <= idx < len(csv_files):
                break
            print(f"0-{len(csv_files)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    return csv_files[idx]


def analyze_all_files(base_dir: str, config: Optional[FilterConfig] = None):
    """모든 CSV 파일 분석

    Args:
        base_dir: CSV 파일이 있는 디렉토리
        config: 필터 설정 (None이면 기본값 사용)
    """
    csv_files = find_csv_files(base_dir)

    if not csv_files:
        print(f"CSV 파일을 찾을 수 없습니다: {base_dir}")
        return

    print(f"\n총 {len(csv_files)}개의 CSV 파일을 찾았습니다.\n")

    for i, filepath in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] 분석 중: {filepath}")

        try:
            analyzer = EAGAnalyzer(filepath, config=config)
            analyzer.generate_report()
        except Exception as e:
            print(f"오류 발생: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='OpenBCI EAG 데이터 분석기 (EMG 노이즈 제거)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (data 폴더가 있으면 자동으로 인터랙티브 모드)
  python eag_analyzer.py

  # 인터랙티브 모드 (data 폴더에서 분석할 데이터 선택)
  python eag_analyzer.py -i
  python eag_analyzer.py --interactive

  # 드리프트 보정 방법 변경 (moving average)
  python eag_analyzer.py --drift-method moving --drift-window 2.0

  # 드리프트 보정 OFF
  python eag_analyzer.py --no-drift

  # 저역통과 필터 20Hz로 변경
  python eag_analyzer.py --lowpass 20

  # 모든 필터 설정 조합
  python eag_analyzer.py --lowpass 15 --drift-method moving --start-time 3

  # 특정 디렉토리 직접 분석 (인터랙티브 모드 건너뛰기)
  python eag_analyzer.py --dir ./data/김O진(F23)_25.12.23
        """
    )

    # 파일/디렉토리
    parser.add_argument('--file', '-f', type=str, help='분석할 CSV 파일 경로')
    parser.add_argument('--dir', '-d', type=str, help='CSV 파일이 있는 디렉토리')
    parser.add_argument('--output', '-o', type=str, help='결과 저장 디렉토리')

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
                        help='드리프트 이동평균 윈도우 (초, moving 방식에서 사용, 기본: 1.0)')

    # 표시 설정
    parser.add_argument('--start-time', '-st', type=float, default=5.0,
                        help='표시 시작 시간 (초, 기본: 5)')
    parser.add_argument('--y-tick', type=float, default=1500,
                        help='Y축 눈금 간격 (기본: 1500)')

    # 설정 확인
    parser.add_argument('--show-config', action='store_true',
                        help='현재 필터 설정만 출력하고 종료')

    # 인터랙티브 모드
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='인터랙티브 모드 (data 폴더에서 분석할 데이터 선택)')

    args = parser.parse_args()

    # FilterConfig 생성
    config = FilterConfig()

    # Lowpass
    config.lowpass_enabled = not args.no_lowpass
    config.lowpass_cutoff = args.lowpass

    # Drift
    config.drift_enabled = not args.no_drift
    config.drift_method = args.drift_method if not args.no_drift else "none"
    config.drift_window_sec = args.drift_window

    # 표시 설정
    config.start_time = args.start_time
    config.y_tick_interval = args.y_tick

    # 설정 확인 모드
    if args.show_config:
        print(config)
        return

    # 분석 실행
    if args.interactive:
        # 인터랙티브 모드: data 폴더에서 분석할 데이터 선택
        selected_path = interactive_select_data()
        if selected_path is None:
            print("선택이 취소되었습니다.")
            return

        # 드리프트 보정 설정 선택
        config = interactive_select_drift(config)

        if selected_path.endswith('.csv'):
            # 단일 파일 선택
            analyzer = EAGAnalyzer(selected_path, config=config)
            analyzer.generate_report(output_dir=args.output)
        else:
            # 폴더 선택 (전체 분석)
            analyze_all_files(selected_path, config=config)
    elif args.file:
        analyzer = EAGAnalyzer(args.file, config=config)
        analyzer.generate_report(output_dir=args.output)
    elif args.dir:
        analyze_all_files(args.dir, config=config)
    else:
        # 기본: data 폴더가 있으면 인터랙티브 모드, 없으면 현재 폴더 분석
        data_dir = get_data_dir()
        if os.path.exists(data_dir) and list_subjects(data_dir):
            print("data 폴더가 감지되었습니다. 인터랙티브 모드로 전환합니다.")
            print("(인터랙티브 모드를 건너뛰려면 --dir 옵션을 사용하세요)\n")
            selected_path = interactive_select_data()
            if selected_path is None:
                print("선택이 취소되었습니다.")
                return

            # 드리프트 보정 설정 선택
            config = interactive_select_drift(config)

            if selected_path.endswith('.csv'):
                analyzer = EAGAnalyzer(selected_path, config=config)
                analyzer.generate_report(output_dir=args.output)
            else:
                analyze_all_files(selected_path, config=config)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            analyze_all_files(current_dir, config=config)


if __name__ == '__main__':
    main()
