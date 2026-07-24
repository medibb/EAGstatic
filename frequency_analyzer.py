"""
EAG Frequency Analyzer — 노이즈 구간의 주파수 성분 시각화
- Spectrogram (STFT): 시간-주파수 히트맵
- PSD 비교: 노이즈 구간 vs 정상 구간
- 채널별 분석 지원

사용법:
  # 인터랙티브 선택
  python3 frequency_analyzer.py

  # 특정 채널만 분석
  python3 frequency_analyzer.py --channels 1 3 5

  # 특정 시간 구간 지정 (노이즈 구간 vs 정상 구간)
  python3 frequency_analyzer.py --noise-range 0 5 --clean-range 10 20

  # 전체 시간 spectrogram만
  python3 frequency_analyzer.py --spectrogram-only
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram, welch, butter, sosfiltfilt
from pathlib import Path
from typing import Optional, List, Tuple
import argparse

from eag_analyzer import (
    EAGAnalyzer, FilterConfig, SAMPLE_RATE, EEG_CHANNELS,
    CHANNEL_NAMES, CHANNEL_COLORS, setup_korean_font
)
from grf_viewer import list_subjects, list_sessions, get_output_dir

setup_korean_font()


def compute_spectrogram(data: np.ndarray, fs: int = SAMPLE_RATE,
                        nperseg: int = 512, noverlap: int = 448
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT 기반 spectrogram 계산

    Args:
        data: 1D 신호 배열
        fs: 샘플링 레이트
        nperseg: FFT 윈도우 크기 (기본 512 → 주파수 해상도 ~0.49Hz)
        noverlap: 오버랩 샘플 수 (기본 448 → 시간 해상도 ~0.26s)

    Returns:
        f: 주파수 배열
        t: 시간 배열
        Sxx: 파워 스펙트럼 (주파수 x 시간)
    """
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            window='hann', scaling='density')
    return f, t, Sxx


def compute_psd(data: np.ndarray, fs: int = SAMPLE_RATE,
                nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD 계산

    Args:
        data: 1D 신호 배열
        fs: 샘플링 레이트
        nperseg: FFT 윈도우 크기

    Returns:
        f: 주파수 배열
        psd: 파워 스펙트럼 밀도
    """
    f, psd = welch(data, fs=fs, nperseg=nperseg, window='hann', scaling='density')
    return f, psd


def plot_spectrogram_all_channels(analyzer: EAGAnalyzer,
                                  channels: Optional[List[int]] = None,
                                  freq_max: float = 60.0,
                                  save_path: Optional[str] = None):
    """전 채널 spectrogram — Raw vs Filtered 비교

    Args:
        analyzer: EAGAnalyzer 인스턴스
        channels: 분석할 채널 리스트 (0-indexed, None이면 전체)
        freq_max: 표시할 최대 주파수 (Hz)
        save_path: 저장 경로
    """
    if channels is None:
        channels = list(range(EEG_CHANNELS))

    n_ch = len(channels)
    raw_data = analyzer.eeg_data
    filtered_data = analyzer.get_filtered_data()

    fig, axes = plt.subplots(n_ch, 2, figsize=(18, 3 * n_ch), squeeze=False)
    fig.suptitle(f'Spectrogram: Raw vs Filtered (Lowpass {analyzer.config.lowpass_cutoff}Hz)\n'
                 f'{analyzer.filename}', fontsize=14, fontweight='bold')

    for i, ch in enumerate(channels):
        # Raw spectrogram
        f_raw, t_raw, Sxx_raw = compute_spectrogram(raw_data[:, ch])
        freq_mask = f_raw <= freq_max

        im0 = axes[i, 0].pcolormesh(t_raw, f_raw[freq_mask], Sxx_raw[freq_mask, :],
                                      shading='gouraud', cmap='inferno',
                                      norm=LogNorm(vmin=max(Sxx_raw[freq_mask, :].min(), 1e-2),
                                                   vmax=Sxx_raw[freq_mask, :].max()))
        axes[i, 0].set_ylabel(f'{CHANNEL_NAMES[ch]}\nFreq (Hz)')
        if i == 0:
            axes[i, 0].set_title('Raw Signal')
        axes[i, 0].set_ylim(0, freq_max)
        plt.colorbar(im0, ax=axes[i, 0], label='PSD (µV²/Hz)', pad=0.01, aspect=20)

        # Filtered spectrogram
        f_filt, t_filt, Sxx_filt = compute_spectrogram(filtered_data[:, ch])

        im1 = axes[i, 1].pcolormesh(t_filt, f_filt[freq_mask], Sxx_filt[freq_mask, :],
                                      shading='gouraud', cmap='inferno',
                                      norm=LogNorm(vmin=max(Sxx_filt[freq_mask, :].min(), 1e-2),
                                                   vmax=Sxx_filt[freq_mask, :].max()))
        if i == 0:
            axes[i, 1].set_title(f'Filtered (LP {analyzer.config.lowpass_cutoff}Hz)')
        axes[i, 1].set_ylim(0, freq_max)
        plt.colorbar(im1, ax=axes[i, 1], label='PSD (µV²/Hz)', pad=0.01, aspect=20)

    # x축 라벨은 마지막 행에만
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectrogram 저장: {save_path}")
    plt.close(fig)
    return fig


def plot_psd_comparison(analyzer: EAGAnalyzer,
                        noise_range: Tuple[float, float] = (0, 5),
                        clean_range: Tuple[float, float] = (10, 20),
                        channels: Optional[List[int]] = None,
                        freq_max: float = 125.0,
                        save_path: Optional[str] = None):
    """노이즈 구간 vs 정상 구간 PSD 비교

    Args:
        analyzer: EAGAnalyzer 인스턴스
        noise_range: 노이즈 구간 (시작, 끝) 초
        clean_range: 정상 구간 (시작, 끝) 초
        channels: 분석할 채널 (None이면 전체)
        freq_max: 최대 주파수
        save_path: 저장 경로
    """
    if channels is None:
        channels = list(range(EEG_CHANNELS))

    fs = analyzer.sample_rate
    raw_data = analyzer.eeg_data

    noise_start = int(noise_range[0] * fs)
    noise_end = int(noise_range[1] * fs)
    clean_start = int(clean_range[0] * fs)
    clean_end = int(clean_range[1] * fs)

    n_ch = len(channels)
    n_cols = min(4, n_ch)
    n_rows = (n_ch + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f'PSD Comparison: Noise ({noise_range[0]}-{noise_range[1]}s) vs '
                 f'Clean ({clean_range[0]}-{clean_range[1]}s)\n{analyzer.filename}',
                 fontsize=13, fontweight='bold')

    for idx, ch in enumerate(channels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # 노이즈 구간 PSD
        noise_seg = raw_data[noise_start:noise_end, ch]
        f_noise, psd_noise = compute_psd(noise_seg, fs)

        # 정상 구간 PSD
        clean_seg = raw_data[clean_start:clean_end, ch]
        f_clean, psd_clean = compute_psd(clean_seg, fs)

        freq_mask_n = f_noise <= freq_max
        freq_mask_c = f_clean <= freq_max

        ax.semilogy(f_noise[freq_mask_n], psd_noise[freq_mask_n],
                     color='red', alpha=0.8, linewidth=1.2,
                     label=f'Noise ({noise_range[0]}-{noise_range[1]}s)')
        ax.semilogy(f_clean[freq_mask_c], psd_clean[freq_mask_c],
                     color='blue', alpha=0.8, linewidth=1.2,
                     label=f'Clean ({clean_range[0]}-{clean_range[1]}s)')

        # 주요 주파수 대역 표시
        ax.axvspan(0, 5, alpha=0.1, color='green', label='EAG band (0-5Hz)')
        ax.axvspan(20, 60, alpha=0.08, color='orange', label='EMG band (20-60Hz)')
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50Hz (power line)')

        ax.set_title(CHANNEL_NAMES[ch], fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (µV²/Hz)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # 빈 subplot 숨기기
    for idx in range(n_ch, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PSD 비교 저장: {save_path}")
    plt.close(fig)
    return fig


def plot_single_channel_detail(analyzer: EAGAnalyzer, ch: int = 0,
                                freq_max: float = 60.0,
                                save_path: Optional[str] = None):
    """단일 채널 상세 분석: 시계열 + Spectrogram + PSD (3행 레이아웃)

    Args:
        analyzer: EAGAnalyzer 인스턴스
        ch: 채널 번호 (0-indexed)
        freq_max: 최대 주파수
        save_path: 저장 경로
    """
    raw = analyzer.eeg_data[:, ch]
    filtered = analyzer.get_filtered_data()[:, ch]
    time = analyzer.get_time_axis()
    fs = analyzer.sample_rate

    fig, axes = plt.subplots(3, 2, figsize=(18, 12),
                              gridspec_kw={'height_ratios': [1, 1.5, 1]})
    fig.suptitle(f'{CHANNEL_NAMES[ch]} Frequency Analysis — {analyzer.filename}',
                 fontsize=14, fontweight='bold')

    # Row 1: Time-domain signal (Raw / Filtered)
    axes[0, 0].plot(time, raw, color=CHANNEL_COLORS[ch], linewidth=0.3, alpha=0.8)
    axes[0, 0].set_title('Raw Signal')
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, filtered, color=CHANNEL_COLORS[ch], linewidth=0.5, alpha=0.8)
    axes[0, 1].set_title(f'Filtered (LP {analyzer.config.lowpass_cutoff}Hz + {analyzer.config.drift_method})')
    axes[0, 1].set_ylabel('Amplitude (µV)')
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2: Spectrogram (Raw / Filtered)
    f_raw, t_raw, Sxx_raw = compute_spectrogram(raw)
    f_filt, t_filt, Sxx_filt = compute_spectrogram(filtered)
    freq_mask = f_raw <= freq_max

    im0 = axes[1, 0].pcolormesh(t_raw, f_raw[freq_mask], Sxx_raw[freq_mask, :],
                                  shading='gouraud', cmap='inferno',
                                  norm=LogNorm(vmin=max(Sxx_raw[freq_mask, :].min(), 1e-2),
                                               vmax=Sxx_raw[freq_mask, :].max()))
    axes[1, 0].set_title('Raw Spectrogram')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].axhline(y=analyzer.config.lowpass_cutoff, color='cyan',
                        linestyle='--', alpha=0.7, label=f'LP cutoff ({analyzer.config.lowpass_cutoff}Hz)')
    axes[1, 0].axhline(y=50, color='white', linestyle=':', alpha=0.5, label='50Hz power line')
    axes[1, 0].legend(fontsize=8, loc='upper right')
    plt.colorbar(im0, ax=axes[1, 0], label='PSD (µV²/Hz)', pad=0.01)

    im1 = axes[1, 1].pcolormesh(t_filt, f_filt[freq_mask], Sxx_filt[freq_mask, :],
                                  shading='gouraud', cmap='inferno',
                                  norm=LogNorm(vmin=max(Sxx_filt[freq_mask, :].min(), 1e-2),
                                               vmax=Sxx_filt[freq_mask, :].max()))
    axes[1, 1].set_title('Filtered Spectrogram')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].axhline(y=analyzer.config.lowpass_cutoff, color='cyan',
                        linestyle='--', alpha=0.7, label=f'LP cutoff ({analyzer.config.lowpass_cutoff}Hz)')
    axes[1, 1].legend(fontsize=8, loc='upper right')
    plt.colorbar(im1, ax=axes[1, 1], label='PSD (µV²/Hz)', pad=0.01)

    # Row 3: PSD — 전체 구간 + 초기 5초 vs 안정 구간
    f_all, psd_all = compute_psd(raw, fs)
    freq_mask_psd = f_all <= freq_max

    # 초기 5초 (노이즈 가능성)
    early_end = min(int(5 * fs), len(raw))
    f_early, psd_early = compute_psd(raw[:early_end], fs, nperseg=min(512, early_end))

    # 안정 구간 (10-30초)
    stable_start = int(10 * fs)
    stable_end = min(int(30 * fs), len(raw))
    f_stable, psd_stable = compute_psd(raw[stable_start:stable_end], fs)

    freq_mask_e = f_early <= freq_max
    freq_mask_s = f_stable <= freq_max

    axes[2, 0].semilogy(f_early[freq_mask_e], psd_early[freq_mask_e],
                         color='red', linewidth=1.2, alpha=0.8, label='Initial 0-5s')
    axes[2, 0].semilogy(f_stable[freq_mask_s], psd_stable[freq_mask_s],
                         color='blue', linewidth=1.2, alpha=0.8, label='Stable 10-30s')
    axes[2, 0].axvspan(0, 5, alpha=0.1, color='green')
    axes[2, 0].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('PSD: Initial (0-5s) vs Stable (10-30s)')
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('PSD (µV²/Hz)')
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)

    # Filtered PSD
    f_filt_psd, psd_filt = compute_psd(filtered, fs)
    freq_mask_fp = f_filt_psd <= freq_max
    axes[2, 1].semilogy(f_all[freq_mask_psd], psd_all[freq_mask_psd],
                         color='gray', linewidth=1, alpha=0.6, label='Raw (full)')
    axes[2, 1].semilogy(f_filt_psd[freq_mask_fp], psd_filt[freq_mask_fp],
                         color='green', linewidth=1.5, alpha=0.9, label='Filtered (full)')
    axes[2, 1].axvline(x=analyzer.config.lowpass_cutoff, color='red',
                        linestyle='--', alpha=0.7, label=f'LP cutoff ({analyzer.config.lowpass_cutoff}Hz)')
    axes[2, 1].axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50Hz')
    axes[2, 1].set_title('PSD: Raw vs Filtered (full recording)')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('PSD (µV²/Hz)')
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"채널 상세 분석 저장: {save_path}")
    plt.close(fig)
    return fig


# ==================== Noise Overlay on Raw Signal ====================

def _bandpower_envelope(data: np.ndarray, fs: int,
                        band: Tuple[float, float],
                        window_sec: float = 1.0) -> np.ndarray:
    """Bandpass 후 sliding RMS envelope 계산."""
    lo, hi = band
    nyq = fs / 2
    # 대역이 Nyquist 이내인지 확인
    if hi >= nyq:
        hi = nyq - 1
    if lo >= hi:
        return np.zeros(len(data))
    sos = butter(4, [lo / nyq, hi / nyq], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, data)

    # sliding RMS
    win = int(window_sec * fs)
    if win < 2:
        win = 2
    cumsum = np.cumsum(np.insert(filtered ** 2, 0, 0))
    rms = np.sqrt((cumsum[win:] - cumsum[:-win]) / win)
    # 길이 맞추기 (앞뒤 패딩)
    pad_front = win // 2
    pad_back = len(data) - len(rms) - pad_front
    rms = np.pad(rms, (pad_front, pad_back), mode='edge')
    return rms


def plot_noise_overlay(analyzer: EAGAnalyzer,
                       channels: Optional[List[int]] = None,
                       emg_threshold_percentile: float = 90,
                       line_threshold_percentile: float = 90,
                       window_sec: float = 1.0,
                       save_path: Optional[str] = None):
    """Raw 시계열 위에 노이즈 구간을 하이라이트하여 시각화.

    - 빨간 배경: EMG 오염 (20-60Hz RMS > threshold)
    - 노란 배경: 50Hz 전원선 노이즈 (49-51Hz RMS > threshold)
    - 파란선: filtered EAG (LP 5Hz)

    Args:
        analyzer: EAGAnalyzer 인스턴스
        channels: 표시할 채널 (None이면 전체 8채널)
        emg_threshold_percentile: EMG envelope 상위 N%를 노이즈로 판정
        line_threshold_percentile: 50Hz envelope 상위 N%를 노이즈로 판정
        window_sec: RMS sliding window 크기 (초)
        save_path: 저장 경로
    """
    if channels is None:
        channels = list(range(EEG_CHANNELS))

    fs = analyzer.sample_rate
    raw = analyzer.eeg_data
    filtered = analyzer.get_filtered_data()
    time = analyzer.get_time_axis()

    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, 1, figsize=(18, 2.5 * n_ch), squeeze=False)
    fig.suptitle(f'Noise Overlay — {analyzer.filename}\n'
                 f'Red=EMG(20-60Hz)  Yellow=50Hz PowerLine  Blue=Filtered(LP5Hz)',
                 fontsize=12, fontweight='bold')

    for i, ch in enumerate(channels):
        ax = axes[i, 0]

        # Raw 시계열
        ax.plot(time, raw[:, ch], color='gray', linewidth=0.3, alpha=0.5, label='Raw')

        # Filtered 시계열
        ax.plot(time, filtered[:, ch], color=CHANNEL_COLORS[ch],
                linewidth=0.8, alpha=0.9, label='Filtered')

        # EMG envelope (20-60Hz)
        emg_env = _bandpower_envelope(raw[:, ch], fs, (20, 60), window_sec)
        emg_thresh = np.percentile(emg_env, emg_threshold_percentile)
        emg_mask = emg_env > emg_thresh

        # 50Hz envelope (49-51Hz)
        line_env = _bandpower_envelope(raw[:, ch], fs, (49, 51), window_sec)
        line_thresh = np.percentile(line_env, line_threshold_percentile)
        line_mask = line_env > line_thresh

        # 하이라이트: fill_between으로 배경 색칠
        ymin, ymax = np.min(raw[:, ch]), np.max(raw[:, ch])
        margin = (ymax - ymin) * 0.05
        ax.fill_between(time, ymin - margin, ymax + margin,
                         where=emg_mask, color='red', alpha=0.15, label='EMG noise')
        ax.fill_between(time, ymin - margin, ymax + margin,
                         where=line_mask & ~emg_mask, color='gold', alpha=0.2, label='50Hz noise')

        ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(µV)')
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=4)

    axes[-1, 0].set_xlabel('Time (s)')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Noise overlay 저장: {save_path}")
    plt.close(fig)
    return fig


# ==================== Batch Noise Screening ====================

def compute_noise_metrics(analyzer: EAGAnalyzer, ch: int,
                          stable_start: float = 5.0
                          ) -> dict:
    """단일 채널의 노이즈 지표를 계산한다.

    Returns:
        dict with keys:
        - snr_db: EAG band (0-5Hz) 대비 noise band (20-60Hz) SNR (dB)
        - power_line_50hz: 50Hz ±1Hz 대역 파워 (µV²/Hz)
        - power_eag_band: 0-5Hz 대역 평균 파워
        - power_emg_band: 20-60Hz 대역 평균 파워
        - power_hf_ratio: high-freq(>10Hz) / total 파워 비율 (0~1)
        - drift_range_uv: raw 신호의 전체 range (max-min, µV)
        - rms_raw: raw 신호 RMS
        - rms_filtered: filtered 신호 RMS
    """
    fs = analyzer.sample_rate
    raw = analyzer.eeg_data[:, ch]
    filtered = analyzer.get_filtered_data()[:, ch]

    # stable 구간만 사용 (초기 불안정 구간 제외)
    start_idx = int(stable_start * fs)
    if start_idx >= len(raw):
        start_idx = 0
    raw_seg = raw[start_idx:]
    filtered_seg = filtered[start_idx:]

    # PSD (Welch)
    nperseg = min(1024, len(raw_seg) // 2)
    if nperseg < 64:
        return {k: float('nan') for k in [
            'snr_db', 'power_line_50hz', 'power_eag_band',
            'power_emg_band', 'power_hf_ratio', 'drift_range_uv',
            'rms_raw', 'rms_filtered']}

    f, psd = welch(raw_seg, fs=fs, nperseg=nperseg, window='hann')

    # 대역별 파워
    mask_eag = (f >= 0.1) & (f <= 5)
    mask_emg = (f >= 20) & (f <= 60)
    mask_50hz = (f >= 49) & (f <= 51)
    mask_hf = f > 10
    mask_all = f > 0.1

    power_eag = float(np.mean(psd[mask_eag])) if np.any(mask_eag) else 0
    power_emg = float(np.mean(psd[mask_emg])) if np.any(mask_emg) else 0
    power_50hz = float(np.mean(psd[mask_50hz])) if np.any(mask_50hz) else 0
    power_hf = float(np.sum(psd[mask_hf])) if np.any(mask_hf) else 0
    power_total = float(np.sum(psd[mask_all])) if np.any(mask_all) else 1

    # SNR: EAG band / EMG band (dB)
    if power_emg > 0:
        snr_db = float(10 * np.log10(power_eag / power_emg))
    else:
        snr_db = float('inf')

    return {
        'snr_db': round(snr_db, 2),
        'power_line_50hz': round(power_50hz, 4),
        'power_eag_band': round(power_eag, 4),
        'power_emg_band': round(power_emg, 4),
        'power_hf_ratio': round(power_hf / power_total, 4) if power_total > 0 else 0,
        'drift_range_uv': round(float(np.max(raw_seg) - np.min(raw_seg)), 1),
        'rms_raw': round(float(np.sqrt(np.mean(raw_seg ** 2))), 1),
        'rms_filtered': round(float(np.sqrt(np.mean(filtered_seg ** 2))), 1),
    }


def flag_noise(metrics: dict) -> List[str]:
    """노이즈 지표 기반 플래그 생성.

    Thresholds are calibrated to flag ~top 10% outliers based on
    18-subject EAG dataset distribution (2026-05-05).
      rms_raw p90 = 16274, drift p90 = 2238, 50Hz p90 = 15.5,
      hf_ratio p90 = 0.6, snr p10 = 9.5
    """
    flags = []
    if metrics['snr_db'] < 10:
        flags.append('LOW_SNR')
    if metrics['power_line_50hz'] > 15:
        flags.append('50HZ_NOISE')
    if metrics['power_hf_ratio'] > 0.6:
        flags.append('HF_CONTAMINATION')
    if metrics['drift_range_uv'] > 3000:
        flags.append('LARGE_DRIFT')
    if metrics['rms_raw'] > 16000:
        flags.append('HIGH_RMS')
    return flags


def run_batch_noise_screening(base_dir: str = 'data',
                               subject_filter: Optional[str] = None,
                               save_flagged_png: bool = True):
    """전체 세션 batch 노이즈 스크리닝.

    사용법:
      python3 frequency_analyzer.py --batch
      python3 frequency_analyzer.py --batch --subject 주창민
      python3 frequency_analyzer.py --batch --no-png
    """
    from sync_analyzer import find_all_pairs

    pairs = find_all_pairs(base_dir)
    if subject_filter:
        pairs = [p for p in pairs if subject_filter in p.subject_name]

    print(f"=== Batch Noise Screening ===")
    print(f"Total sessions: {len(pairs)}")
    if subject_filter:
        print(f"Filter: {subject_filter}")
    print()

    all_rows = []
    flagged_sessions = []

    for i, pair in enumerate(pairs):
        tag = f"[{i+1}/{len(pairs)}]"
        try:
            analyzer = EAGAnalyzer(pair.eag_filepath)

            session_flags = []
            for ch in range(EEG_CHANNELS):
                metrics = compute_noise_metrics(analyzer, ch)
                flags = flag_noise(metrics)

                row = {
                    'subject': pair.subject_name,
                    'session': pair.session_name,
                    'channel': ch + 1,
                    'channel_name': CHANNEL_NAMES[ch],
                    **metrics,
                    'flags': '|'.join(flags) if flags else '',
                    'n_flags': len(flags),
                }
                all_rows.append(row)
                session_flags.extend(flags)

            n_flags = len(session_flags)
            if n_flags > 0:
                flag_summary = ', '.join(set(session_flags))
                print(f"  {tag} {pair.subject_name}/{pair.session_name}: "
                      f"{n_flags} flags [{flag_summary}]")
                flagged_sessions.append((pair, set(session_flags)))
            else:
                print(f"  {tag} {pair.subject_name}/{pair.session_name}: OK")

        except Exception as e:
            print(f"  {tag} {pair.subject_name}/{pair.session_name}: ERROR — {e}")

    if not all_rows:
        print("분석된 데이터 없음")
        return

    # CSV 저장
    import pandas as pd
    df = pd.DataFrame(all_rows)
    output_dir = Path('result') / 'noise_screening'
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{subject_filter}" if subject_filter else ""
    csv_path = output_dir / f'noise_metrics{suffix}.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n저장: {csv_path} ({len(df)} rows)")

    # 요약 통계
    print(f"\n=== Summary ===")
    total_sessions = df[['subject', 'session']].drop_duplicates().shape[0]
    flagged_count = df[df['n_flags'] > 0][['subject', 'session']].drop_duplicates().shape[0]
    clean_count = total_sessions - flagged_count
    print(f"  Clean: {clean_count}/{total_sessions} sessions")
    print(f"  Flagged: {flagged_count}/{total_sessions} sessions")

    # 플래그별 집계
    flag_cols = df[df['flags'] != '']
    if len(flag_cols) > 0:
        all_flags_flat = []
        for flags_str in flag_cols['flags']:
            all_flags_flat.extend(flags_str.split('|'))
        from collections import Counter
        flag_counts = Counter(all_flags_flat)
        print(f"\n  Flag distribution (channel-level):")
        for flag, count in flag_counts.most_common():
            print(f"    {flag}: {count} channels")

    # 채널별 평균 지표
    print(f"\n=== Channel-level Noise Metrics (mean) ===")
    print(f"  {'Ch':<6} {'SNR(dB)':<10} {'50Hz':<10} {'EAG pwr':<10} "
          f"{'EMG pwr':<10} {'HF ratio':<10} {'Drift(uV)':<10}")
    for ch in range(EEG_CHANNELS):
        ch_data = df[df['channel'] == ch + 1]
        print(f"  Ch{ch+1:<3} "
              f"{ch_data['snr_db'].mean():<10.1f} "
              f"{ch_data['power_line_50hz'].mean():<10.3f} "
              f"{ch_data['power_eag_band'].mean():<10.1f} "
              f"{ch_data['power_emg_band'].mean():<10.3f} "
              f"{ch_data['power_hf_ratio'].mean():<10.4f} "
              f"{ch_data['drift_range_uv'].mean():<10.0f}")

    # Flagged 세션 PSD PNG 저장
    if save_flagged_png and flagged_sessions:
        print(f"\n=== Flagged 세션 PSD 저장 ({len(flagged_sessions)}개) ===")
        png_dir = output_dir / 'flagged_psd'
        png_dir.mkdir(parents=True, exist_ok=True)

        for pair, flags in flagged_sessions:
            try:
                analyzer = EAGAnalyzer(pair.eag_filepath)
                save_path = png_dir / f'{pair.subject_name}_{pair.session_name}_psd.png'
                plot_psd_comparison(analyzer,
                                    noise_range=(0, 5), clean_range=(10, 20),
                                    freq_max=125.0, save_path=str(save_path))
            except Exception as e:
                print(f"  PNG error: {pair.subject_name}/{pair.session_name}: {e}")

    return df


# ==================== CLI ====================
def find_forceplate_csvs_in_dir(session_dir):
    """세션 폴더에서 EAG CSV 파일 찾기"""
    import glob
    return glob.glob(os.path.join(session_dir, 'BrainFlow-RAW_*.csv'))


def main():
    parser = argparse.ArgumentParser(description='EAG Frequency Analyzer')
    parser.add_argument('--channels', nargs='+', type=int, default=None,
                        help='분석할 채널 번호 (0-indexed, 예: 0 2 4)')
    parser.add_argument('--noise-range', nargs=2, type=float, default=[0, 5],
                        help='노이즈 구간 (초, 예: 0 5)')
    parser.add_argument('--clean-range', nargs=2, type=float, default=[10, 20],
                        help='정상 구간 (초, 예: 10 20)')
    parser.add_argument('--freq-max', type=float, default=60.0,
                        help='표시 최대 주파수 (Hz)')
    parser.add_argument('--spectrogram-only', action='store_true',
                        help='Spectrogram만 생성')
    parser.add_argument('--channel-detail', type=int, default=None,
                        help='특정 채널 상세 분석 (0-indexed)')
    parser.add_argument('--noise-overlay', action='store_true',
                        help='Raw 시계열 위에 노이즈 구간 하이라이트')
    parser.add_argument('--file', type=str, default=None,
                        help='직접 EAG CSV 파일 경로 지정')
    parser.add_argument('--batch', action='store_true',
                        help='전체 세션 batch 노이즈 스크리닝')
    parser.add_argument('--subject', type=str, default=None,
                        help='피험자 필터 (부분 일치, --batch와 함께 사용)')
    parser.add_argument('--no-png', action='store_true',
                        help='flagged 세션 PSD PNG 저장 안 함')
    args = parser.parse_args()

    if args.batch:
        run_batch_noise_screening(
            base_dir='data',
            subject_filter=args.subject,
            save_flagged_png=not args.no_png,
        )
        return

    # 파일 선택
    if args.file:
        filepath = args.file
    else:
        # 인터랙티브 선택
        subjects = list_subjects('data')
        if not subjects:
            print("data 폴더에 피험자 데이터가 없습니다.")
            return

        print("\n=== 피험자 선택 ===")
        for i, (display, path) in enumerate(subjects):
            print(f"  {i}: {display}")
        subj_idx = int(input("피험자 번호: "))
        _, subj_path = subjects[subj_idx]

        sessions = list_sessions(subj_path)
        print("\n=== 세션 선택 ===")
        for i, s in enumerate(sessions):
            print(f"  {i}: {s}")
        sess_idx = int(input("세션 번호: "))
        session_path = os.path.join(subj_path, sessions[sess_idx])

        eag_files = find_forceplate_csvs_in_dir(session_path)
        if not eag_files:
            print("EAG 파일을 찾을 수 없습니다.")
            return
        filepath = eag_files[0]

    print(f"\n분석 파일: {filepath}")
    analyzer = EAGAnalyzer(filepath)

    # 출력 폴더
    output_dir = get_output_dir(filepath)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_tag = Path(filepath).parent.name.split('-')[-1]  # s1, f2, c3 등

    if args.noise_overlay:
        # Raw 시계열 + 노이즈 구간 하이라이트
        save_path = output_dir / f'{session_tag}_noise_overlay.png'
        plot_noise_overlay(analyzer, channels=args.channels,
                           save_path=str(save_path))
    elif args.channel_detail is not None:
        # 단일 채널 상세 분석
        save_path = output_dir / f'{session_tag}_freq_detail_ch{args.channel_detail}.png'
        plot_single_channel_detail(analyzer, ch=args.channel_detail,
                                    freq_max=args.freq_max, save_path=str(save_path))
    else:
        # Spectrogram (전 채널)
        save_path = output_dir / f'{session_tag}_spectrogram.png'
        plot_spectrogram_all_channels(analyzer, channels=args.channels,
                                       freq_max=args.freq_max, save_path=str(save_path))

        if not args.spectrogram_only:
            # PSD 비교
            save_path = output_dir / f'{session_tag}_psd_comparison.png'
            plot_psd_comparison(analyzer,
                                noise_range=tuple(args.noise_range),
                                clean_range=tuple(args.clean_range),
                                channels=args.channels,
                                freq_max=args.freq_max,
                                save_path=str(save_path))


if __name__ == '__main__':
    main()
