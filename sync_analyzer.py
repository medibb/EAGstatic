"""
EAG-GRF Synchronized Analysis Tool
- EAG(8ch EEG)와 GRF(forceplate) 데이터를 시간 동기화
- GRF 체중이동 이벤트 감지 및 강도 분류
- EAG 변곡점(inflection point) 감지 및 통계
- 동기화 시각화 + annotation
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from eag_analyzer import (
    EAGAnalyzer, FilterConfig, EMGFilter, setup_korean_font,
    SAMPLE_RATE, EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS,
    find_csv_files, get_data_dir, get_result_dir,
    list_subjects, list_sessions,
)
from grf_viewer import (
    load_grf_data, extract_body_weight,
    find_forceplate_csvs_in_dir,
)


# ==================== Data Classes ====================

@dataclass
class GRFEvent:
    """A single detected weight-shift event from GRF data."""
    event_id: int
    onset_time: float       # seconds (unified time axis)
    offset_time: float
    duration: float
    peak_imbalance: float   # max |L-R|/(L+R)
    dominant_side: str      # 'left' or 'right'
    intensity: str          # 'strong', 'moderate', 'mild'
    peak_left_grf: float
    peak_right_grf: float


@dataclass
class EAGInflection:
    """A single EAG inflection point near a GRF event."""
    event_id: int
    channel: int            # 0-7
    time: float             # seconds (unified time axis)
    amplitude: float        # µV at inflection
    direction: str          # 'peak' or 'valley'
    latency: float          # seconds relative to GRF event onset


@dataclass
class SessionPair:
    """A paired EAG + GRF recording from the same session folder."""
    session_dir: str
    eag_filepath: str
    grf_filepath: str
    session_name: str       # e.g. 's1', 'f2'
    subject_name: str


# ==================== Time Parsing ====================

def parse_eag_start_sod(timestamps: np.ndarray, utc_offset: int = 9) -> float:
    """EAG Unix timestamp에서 KST seconds-of-day를 추출한다.

    Args:
        timestamps: EAG column 22 (Unix timestamps)
        utc_offset: UTC offset (default 9 for KST)

    Returns:
        seconds-of-day in local time
    """
    # Skip sample 0 if timestamp is 0 or very different from sample 1
    ts = timestamps[1] if len(timestamps) > 1 and timestamps[0] == 0 else timestamps[0]
    sod = (ts % 86400) + utc_offset * 3600
    if sod >= 86400:
        sod -= 86400
    return sod


def parse_grf_start_sod(filepath: str) -> float:
    """GRF 파일명에서 시작 시각의 seconds-of-day를 파싱한다.

    파일명 패턴: 양발_자세_성_이름N__DDM월YY_HH_MM_SS_DDM월YY_HH_MM_SS.csv
    """
    basename = os.path.basename(filepath)
    # 첫 번째 시간 패턴 매칭: DDM월YY_HH_MM_SS
    pattern = r'(\d{2})\d{1,2}월\d{2}_(\d{1,2})_(\d{2})_(\d{2})'
    matches = re.findall(pattern, basename)
    if not matches:
        raise ValueError(f"GRF 파일명에서 시간 정보를 파싱할 수 없습니다: {basename}")

    # 첫 번째 매치 = 시작 시각
    _, h, m, s = matches[0]
    return int(h) * 3600 + int(m) * 60 + int(s)


# ==================== Session Pairing ====================

def find_session_pair(session_dir: str) -> Optional[SessionPair]:
    """세션 폴더에서 EAG + GRF 파일 쌍을 찾는다."""
    eag_files = find_csv_files(session_dir)
    grf_files = find_forceplate_csvs_in_dir(session_dir)

    if not eag_files or not grf_files:
        return None

    # 세션명 추출 (폴더명의 마지막 '-' 이후 부분)
    dir_name = os.path.basename(session_dir)
    session_match = re.search(r'-([a-zA-Z]+\d*)$', dir_name)
    session_name = session_match.group(1) if session_match else dir_name

    # 피험자명 추출 (OpenBCISession 상위 폴더)
    parent = os.path.basename(os.path.dirname(session_dir))
    subject_name = parent

    return SessionPair(
        session_dir=session_dir,
        eag_filepath=eag_files[0],
        grf_filepath=grf_files[0],
        session_name=session_name,
        subject_name=subject_name,
    )


def find_all_pairs(base_dir: str) -> List[SessionPair]:
    """피험자 폴더 또는 data 폴더에서 모든 EAG+GRF 쌍을 찾는다."""
    pairs = []

    # OpenBCISession 폴더 직접 검색
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        dirname = os.path.basename(root)
        if dirname.startswith('OpenBCISession'):
            pair = find_session_pair(root)
            if pair:
                pairs.append(pair)

    return sorted(pairs, key=lambda p: p.eag_filepath)


# ==================== Sync Analyzer ====================

class SyncAnalyzer:
    """EAG-GRF synchronized analysis engine."""

    def __init__(self, session_pair: SessionPair,
                 config: Optional[FilterConfig] = None,
                 utc_offset: int = 9,
                 manual_offset: Optional[float] = None):
        self.pair = session_pair
        self.config = config or FilterConfig()
        self.config.start_time = 0.0  # 동기화 분석에서는 0초부터
        self.utc_offset = utc_offset

        # Load EAG
        self.eag = EAGAnalyzer(session_pair.eag_filepath, config=self.config)

        # Load GRF
        self.grf_df = load_grf_data(session_pair.grf_filepath)
        self.body_weight = extract_body_weight(session_pair.grf_filepath)

        # Time alignment
        self.time_offset = 0.0
        self.overlap_start = 0.0
        self.overlap_duration = 0.0
        self.unified_time_eag = None
        self.unified_time_grf = None
        self.eag_filtered = None
        self.grf_left = None
        self.grf_right = None

        # Manual offset: 명시적 파라미터 > JSON 조회 > auto
        if manual_offset is None:
            from offset_manager import get_manual_offset
            manual_offset = get_manual_offset(
                session_pair.subject_name, session_pair.session_name)
        self._manual_offset = manual_offset

        self._align_time()

        # Results
        self.grf_events: List[GRFEvent] = []
        self.eag_inflections: List[EAGInflection] = []

    @staticmethod
    def _detect_initial_event_grf(grf_df, body_weight: float,
                                  search_sec: float = 10.0,
                                  imb_threshold: float = 0.18) -> float:
        """GRF 데이터의 초기 한발서기(one-leg-stand) 이벤트 onset을 감지한다.

        step-on(발판 올라서기) 이벤트를 구분하여 skip한다:
        이벤트 직전 0.5초의 total GRF가 체중의 50% 미만이면 step-on으로 판정.

        Returns:
            onset time in seconds from GRF recording start
        """
        time = grf_df['time'].values
        left = grf_df['left_grf'].values
        right = grf_df['right_grf'].values
        mask = time <= search_sec
        t = time[mask]
        total = left[mask] + right[mask]
        safe_total = np.maximum(total, 0.1)
        imbalance = np.abs(left[mask] - right[mask]) / safe_total

        # imbalance threshold를 넘는 모든 구간 찾기
        above = np.where(imbalance > imb_threshold)[0]
        if len(above) == 0:
            raise ValueError(
                f"GRF 초기 {search_sec:.0f}초 내에 체중이동 이벤트를 찾을 수 없습니다. "
                f"(max imbalance={np.max(imbalance):.3f}, threshold={imb_threshold})")

        # 연속 구간(이벤트)으로 그룹화
        fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 250.0
        gap_samples = int(0.3 * fs)  # 0.3초 이상 떨어지면 다른 이벤트
        events_start = [above[0]]
        for i in range(1, len(above)):
            if above[i] - above[i - 1] > gap_samples:
                events_start.append(above[i])

        # 각 이벤트가 step-on인지 one-leg-stand인지 판정
        weight_threshold = body_weight * 0.50 if body_weight > 0 else 20.0
        pre_window = int(0.5 * fs)  # 직전 0.5초

        for ev_start_idx in events_start:
            onset_time = float(t[ev_start_idx])
            # 이벤트 직전 0.5초의 total GRF 평균
            pre_start = max(0, ev_start_idx - pre_window)
            pre_total = np.mean(total[pre_start:ev_start_idx]) if ev_start_idx > 0 else 0.0

            if pre_total >= weight_threshold:
                # 이미 서 있는 상태 → one-leg-stand → 동기화 이벤트
                return onset_time
            # else: step-on → skip하고 다음 이벤트 확인

        raise ValueError(
            f"GRF 초기 {search_sec:.0f}초 내에 한발서기 이벤트를 찾을 수 없습니다. "
            f"(감지된 이벤트 {len(events_start)}개 모두 step-on 패턴)")

    @staticmethod
    def _detect_initial_event_eag(eeg_data, search_sec: float = 5.0) -> float:
        """EAG 데이터의 초기 급격한 transient(체중이동) onset을 감지한다.

        모든 채널의 |gradient|를 합산하여 최대 activity 지점을 찾고,
        onset을 역추적한다.

        Returns:
            onset time in seconds from EAG recording start
        """
        n_search = int(search_sec * SAMPLE_RATE)
        start_idx = 1 if eeg_data[0, 0] == 0 else 0
        end_idx = min(start_idx + n_search, len(eeg_data))
        seg = eeg_data[start_idx:end_idx].astype(float)
        # 각 채널의 |gradient| 합산
        grad_sum = np.zeros(len(seg) - 1)
        for ch in range(min(EEG_CHANNELS, seg.shape[1])):
            grad_sum += np.abs(np.diff(seg[:, ch]))
        # Smoothing (50 samples = 0.2s window)
        kernel = np.ones(50) / 50
        grad_smooth = np.convolve(grad_sum, kernel, mode='same')
        peak_idx = np.argmax(grad_smooth)
        # onset = peak에서 역추적하여 gradient가 상승하기 시작하는 시점
        baseline = np.median(grad_smooth[:int(0.5 * SAMPLE_RATE)])
        threshold = baseline + 0.3 * (grad_smooth[peak_idx] - baseline)
        onset_idx = peak_idx
        for i in range(peak_idx, -1, -1):
            if grad_smooth[i] < threshold:
                onset_idx = i
                break
        return (start_idx + onset_idx) / SAMPLE_RATE

    def _sync_by_xcorr(self) -> float:
        """Cross-correlation으로 EAG-GRF 시간 오프셋을 추정한다.

        GRF: |Left - Right| envelope
        EAG: Σ(8ch) |d/dt(LP filtered)| envelope
        두 envelope의 cross-correlation peak lag를 반환한다.

        Returns:
            offset in seconds (EAG_time = GRF_time + offset)
        """
        # --- GRF envelope: |Left - Right| ---
        grf_time = self.grf_df['time'].values
        grf_left = self.grf_df['left_grf'].values
        grf_right = self.grf_df['right_grf'].values
        grf_env = np.abs(grf_left - grf_right)
        # Smoothing (0.5초 window)
        grf_fs = 1.0 / np.median(np.diff(grf_time)) if len(grf_time) > 1 else SAMPLE_RATE
        grf_kernel = np.ones(int(0.5 * grf_fs)) / int(0.5 * grf_fs)
        grf_env_smooth = np.convolve(grf_env, grf_kernel, mode='same')

        # --- EAG envelope: Σ|d/dt(LP filtered)| ---
        skip_n = 1 if self.eag.eeg_data[0, 0] == 0 else 0
        eag_raw = self.eag.eeg_data[skip_n:].astype(float)
        # LP filter 전체 EAG (mirror padding)
        pad_n = int(2.0 * SAMPLE_RATE)
        emg_filter = EMGFilter()
        grad_sum = np.zeros(len(eag_raw) - 1)
        for ch in range(EEG_CHANNELS):
            seg = eag_raw[:, ch].copy()
            front_pad = seg[pad_n:0:-1] if len(seg) > pad_n else seg[::-1]
            back_pad = seg[-2:-pad_n-2:-1] if len(seg) > pad_n else seg[::-1]
            padded = np.concatenate([front_pad, seg, back_pad])
            if self.config.lowpass_enabled:
                padded = emg_filter.apply_lowpass(
                    padded, self.config.lowpass_cutoff, SAMPLE_RATE, self.config.lowpass_order)
            filtered = padded[len(front_pad):len(front_pad) + len(seg)]
            grad_sum += np.abs(np.diff(filtered))
        # Smoothing (0.5초 window)
        eag_kernel = np.ones(int(0.5 * SAMPLE_RATE)) / int(0.5 * SAMPLE_RATE)
        eag_env_smooth = np.convolve(grad_sum, eag_kernel, mode='same')
        eag_env_time = np.arange(len(eag_env_smooth)) / SAMPLE_RATE + (skip_n + 0.5) / SAMPLE_RATE

        # --- Resample to common rate (GRF rate) ---
        common_fs = grf_fs
        common_duration = min(eag_env_time[-1], grf_time[-1])
        n_common = int(common_duration * common_fs)
        t_common = np.arange(n_common) / common_fs

        grf_resampled = np.interp(t_common, grf_time[:len(grf_env_smooth)], grf_env_smooth)
        eag_resampled = np.interp(t_common, eag_env_time, eag_env_smooth)

        # Normalize
        grf_resampled -= np.mean(grf_resampled)
        eag_resampled -= np.mean(eag_resampled)
        grf_std = np.std(grf_resampled)
        eag_std = np.std(eag_resampled)
        if grf_std > 0:
            grf_resampled /= grf_std
        if eag_std > 0:
            eag_resampled /= eag_std

        # --- Cross-correlation ---
        # 최대 ±10초 범위로 검색
        max_lag_samples = int(10.0 * common_fs)
        correlation = np.correlate(
            eag_resampled,
            grf_resampled[max_lag_samples:-max_lag_samples] if len(grf_resampled) > 2 * max_lag_samples
            else grf_resampled,
            mode='full' if len(grf_resampled) <= 2 * max_lag_samples else 'valid'
        )

        # 방법: full correlation 후 lag 범위 제한
        full_corr = np.correlate(eag_resampled, grf_resampled, mode='full')
        lags = np.arange(-len(grf_resampled) + 1, len(eag_resampled))
        lag_seconds = lags / common_fs
        # ±10초 범위만
        valid = (np.abs(lag_seconds) <= 10.0)
        full_corr_valid = full_corr[valid]
        lags_valid = lag_seconds[valid]

        best_idx = np.argmax(full_corr_valid)
        best_lag = lags_valid[best_idx]
        self._xcorr_confidence = full_corr_valid[best_idx] / len(t_common)

        return best_lag

    def _align_time(self):
        """EAG와 GRF의 시간축을 정렬한다.

        Manual offset이 있으면 auto alignment를 건너뛴다.
        없으면 2단계 전략:
        1차: 초기 이벤트 매칭 (2-3.5초 체중이동 이벤트)
        2차: Cross-correlation fallback (1차 실패 or |offset| > 2초일 때)
        """
        self.sync_method = "unknown"
        self._xcorr_confidence = 0.0
        eag_event_t = None
        grf_event_t = None

        if self._manual_offset is not None:
            # --- Manual offset ---
            self.time_offset = self._manual_offset
            self.sync_method = "manual"
        else:
            # --- 1차: 초기 이벤트 매칭 ---
            event_ok = False
            try:
                eag_event_t = self._detect_initial_event_eag(self.eag.eeg_data)
                grf_event_t = self._detect_initial_event_grf(self.grf_df, self.body_weight)
                offset_event = eag_event_t - grf_event_t
                if abs(offset_event) <= 2.0:
                    self.time_offset = offset_event
                    self.sync_method = "event"
                    event_ok = True
            except ValueError:
                pass

            # --- 2차: Cross-correlation fallback ---
            if not event_ok:
                offset_xcorr = self._sync_by_xcorr()
                self.time_offset = offset_xcorr
                self.sync_method = "xcorr"

        self._compute_unified_axes()

        # --- Print alignment info ---
        print(f"\n{'='*60}")
        print(f"=== 시간 정렬 결과 ({self.sync_method}) ===")
        print(f"{'='*60}")
        if eag_event_t is not None:
            print(f"  EAG 초기 이벤트: {eag_event_t:.3f}초")
        if grf_event_t is not None:
            print(f"  GRF 초기 이벤트: {grf_event_t:.3f}초")
        if self.sync_method == "manual":
            print(f"  동기화 방법: Manual offset")
        elif self.sync_method == "event":
            print(f"  동기화 방법: 초기 이벤트 매칭")
        else:
            print(f"  동기화 방법: Cross-correlation fallback")
            print(f"  XCorr confidence: {self._xcorr_confidence:.4f}")
        print(f"  오프셋: {self.time_offset:+.3f}초")
        print(f"  중첩 구간: {self.overlap_duration:.1f}초")
        print(f"  EAG 샘플: {len(self.unified_time_eag)}")
        print(f"  GRF 샘플: {len(self.unified_time_grf)}")
        print(f"{'='*60}\n")

    def _compute_unified_axes(self):
        """time_offset 기반으로 unified time axis, filtered EAG, GRF를 계산한다."""
        skip_n = 1 if self.eag.eeg_data[0, 0] == 0 else 0
        eag_raw = self.eag.eeg_data[skip_n:]
        eag_time = np.arange(len(eag_raw)) / SAMPLE_RATE + (skip_n / SAMPLE_RATE)

        grf_time = self.grf_df['time'].values
        grf_time_aligned = grf_time + self.time_offset

        overlap_start = max(eag_time[0], grf_time_aligned[0])
        overlap_end = min(eag_time[-1], grf_time_aligned[-1])

        if overlap_end <= overlap_start:
            raise ValueError(
                f"시간 정렬 후 중첩 구간이 없습니다. "
                f"method={self.sync_method}, offset={self.time_offset:.3f}s"
            )

        self.overlap_duration = overlap_end - overlap_start

        eag_unified = eag_time - overlap_start
        eag_mask = (eag_unified >= 0) & (eag_unified <= self.overlap_duration)

        grf_unified = grf_time_aligned - overlap_start
        grf_mask = (grf_unified >= 0) & (grf_unified <= self.overlap_duration)

        self.unified_time_eag = eag_unified[eag_mask]
        self.unified_time_grf = grf_unified[grf_mask]

        # --- EAG 필터링 (mirror padding) ---
        eag_indices = np.where(eag_mask)[0]
        n_overlap = len(eag_indices)
        eag_overlap_raw = eag_raw[eag_indices]
        pad_n = int(2.0 * SAMPLE_RATE)

        emg_filter = EMGFilter()
        self.eag_filtered = np.zeros((n_overlap, EEG_CHANNELS), dtype=float)
        for ch in range(EEG_CHANNELS):
            seg = eag_overlap_raw[:, ch].copy().astype(float)
            front_pad = seg[pad_n:0:-1] if len(seg) > pad_n else seg[::-1]
            back_pad = seg[-2:-pad_n-2:-1] if len(seg) > pad_n else seg[::-1]
            padded = np.concatenate([front_pad, seg, back_pad])
            if self.config.lowpass_enabled:
                padded = emg_filter.apply_lowpass(
                    padded, self.config.lowpass_cutoff, SAMPLE_RATE, self.config.lowpass_order)
            overlap_data = padded[len(front_pad):len(front_pad) + n_overlap]
            if self.config.drift_enabled and self.config.drift_method != "none":
                overlap_data = emg_filter.remove_drift(
                    overlap_data, method=self.config.drift_method,
                    window_sec=self.config.drift_window_sec, fs=SAMPLE_RATE)
            self.eag_filtered[:, ch] = overlap_data

        # --- GRF data ---
        self.grf_left = self.grf_df['left_grf'].values[grf_mask]
        self.grf_right = self.grf_df['right_grf'].values[grf_mask]

    # ==================== GRF Event Detection ====================

    def detect_grf_events(self,
                          imbalance_threshold: float = 0.30,
                          min_duration_sec: float = 0.10,
                          merge_gap_sec: float = 0.25) -> List[GRFEvent]:
        """GRF 데이터에서 체중이동 이벤트를 감지한다."""
        total = self.grf_left + self.grf_right
        safe_total = np.maximum(total, 0.1)
        imbalance = np.abs(self.grf_left - self.grf_right) / safe_total
        signed_imb = (self.grf_left - self.grf_right) / safe_total

        # Smooth (0.1s window)
        window = max(int(0.1 * SAMPLE_RATE), 1)
        imbalance_smooth = pd.Series(imbalance).rolling(window, center=True, min_periods=1).mean().values

        # Threshold detection → contiguous regions
        above = imbalance_smooth > imbalance_threshold
        diff = np.diff(above.astype(int))
        onsets = np.where(diff == 1)[0] + 1
        offsets = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above[0]:
            onsets = np.concatenate(([0], onsets))
        if above[-1]:
            offsets = np.concatenate((offsets, [len(above)]))

        if len(onsets) == 0 or len(offsets) == 0:
            self.grf_events = []
            return []

        # Pair onsets/offsets
        min_samples = int(min_duration_sec * SAMPLE_RATE)
        merge_samples = int(merge_gap_sec * SAMPLE_RATE)

        events_raw = []
        for on, off in zip(onsets, offsets):
            if off - on >= min_samples:
                events_raw.append((on, off))

        # Merge close events
        if not events_raw:
            self.grf_events = []
            return []

        merged = [events_raw[0]]
        for on, off in events_raw[1:]:
            prev_on, prev_off = merged[-1]
            if on - prev_off <= merge_samples:
                merged[-1] = (prev_on, off)
            else:
                merged.append((on, off))

        # Build GRFEvent objects
        events = []
        for idx, (on, off) in enumerate(merged):
            seg_imb = imbalance[on:off]
            seg_signed = signed_imb[on:off]
            seg_left = self.grf_left[on:off]
            seg_right = self.grf_right[on:off]

            peak_imb = float(np.max(seg_imb))
            dominant = 'left' if np.mean(seg_signed) > 0 else 'right'

            if peak_imb > 0.80:
                intensity = 'strong'
            elif peak_imb > 0.50:
                intensity = 'moderate'
            else:
                intensity = 'mild'

            onset_t = float(self.unified_time_grf[on])
            offset_t = float(self.unified_time_grf[min(off - 1, len(self.unified_time_grf) - 1)])

            events.append(GRFEvent(
                event_id=idx + 1,
                onset_time=onset_t,
                offset_time=offset_t,
                duration=offset_t - onset_t,
                peak_imbalance=peak_imb,
                dominant_side=dominant,
                intensity=intensity,
                peak_left_grf=float(np.max(seg_left)),
                peak_right_grf=float(np.max(seg_right)),
            ))

        self.grf_events = events

        # Print summary
        print(f"GRF 이벤트 감지: {len(events)}개")
        counts = {'strong': 0, 'moderate': 0, 'mild': 0}
        for e in events:
            counts[e.intensity] += 1
        print(f"  Strong(>0.80): {counts['strong']}, "
              f"Moderate(0.50-0.80): {counts['moderate']}, "
              f"Mild(0.30-0.50): {counts['mild']}")

        return events

    # ==================== EAG Inflection Detection ====================

    def detect_eag_inflections(self,
                               window_before: float = 2.0,
                               window_after: float = 3.0,
                               min_prominence: float = 10.0,
                               min_distance_samples: int = 25
                               ) -> List[EAGInflection]:
        """각 GRF 이벤트 주변에서 EAG 변곡점을 감지한다."""
        all_inflections = []

        for event in self.grf_events:
            win_start = event.onset_time - window_before
            win_end = event.offset_time + window_after
            time_mask = (self.unified_time_eag >= win_start) & (self.unified_time_eag <= win_end)
            indices = np.where(time_mask)[0]

            if len(indices) < 10:
                continue

            for ch in range(EEG_CHANNELS):
                data = self.eag_filtered[indices, ch]
                times = self.unified_time_eag[indices]

                # Peaks
                peaks, props = find_peaks(
                    data, prominence=min_prominence, distance=min_distance_samples
                )
                for p in peaks:
                    all_inflections.append(EAGInflection(
                        event_id=event.event_id,
                        channel=ch,
                        time=float(times[p]),
                        amplitude=float(data[p]),
                        direction='peak',
                        latency=float(times[p]) - event.onset_time,
                    ))

                # Valleys
                valleys, props = find_peaks(
                    -data, prominence=min_prominence, distance=min_distance_samples
                )
                for v in valleys:
                    all_inflections.append(EAGInflection(
                        event_id=event.event_id,
                        channel=ch,
                        time=float(times[v]),
                        amplitude=float(data[v]),
                        direction='valley',
                        latency=float(times[v]) - event.onset_time,
                    ))

        self.eag_inflections = all_inflections
        print(f"EAG 변곡점 감지: {len(all_inflections)}개 ({len(self.grf_events)}개 이벤트)")
        return all_inflections

    # ==================== Analysis Pipeline ====================

    def run_analysis(self,
                     imbalance_threshold: float = 0.30,
                     min_duration_sec: float = 0.10,
                     merge_gap_sec: float = 0.25,
                     window_before: float = 2.0,
                     window_after: float = 3.0,
                     min_prominence: float = 10.0):
        """전체 분석 파이프라인 실행."""
        print(f"\n분석 시작: {self.pair.subject_name} / {self.pair.session_name}")

        self.detect_grf_events(imbalance_threshold, min_duration_sec, merge_gap_sec)
        self.detect_eag_inflections(window_before, window_after, min_prominence)

        return self

    # ==================== Visualization ====================

    def plot_synchronized(self, save_path: Optional[str] = None):
        """동기화된 EAG 8ch + GRF 그래프 (이벤트 구간 + 변곡점 annotation)."""
        setup_korean_font()

        fig, axes = plt.subplots(
            EEG_CHANNELS + 1, 1, figsize=(18, 2.5 * EEG_CHANNELS + 3),
            sharex=True,
            gridspec_kw={'height_ratios': [1] * EEG_CHANNELS + [1.3]}
        )

        # Y축 통일 range 계산
        ranges = []
        centers = []
        for ch in range(EEG_CHANNELS):
            d = self.eag_filtered[:, ch]
            ch_range = np.nanmax(d) - np.nanmin(d)
            ranges.append(ch_range)
            centers.append(np.nanmean(d))
        max_range = max(ranges)
        half = max_range * 1.1 / 2

        # 이벤트 색상
        event_colors = {'strong': ('orange', 0.22), 'moderate': ('#FFD700', 0.18), 'mild': ('gray', 0.13)}

        # --- EAG 채널들 ---
        for ch in range(EEG_CHANNELS):
            ax = axes[ch]
            ax.plot(self.unified_time_eag, self.eag_filtered[:, ch],
                    linewidth=0.5, color=CHANNEL_COLORS[ch])
            ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(µV)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(centers[ch] - half, centers[ch] + half)
            ax.set_xlim(0, self.overlap_duration)

            # 이벤트 구간 shading
            for event in self.grf_events:
                color, alpha = event_colors[event.intensity]
                ax.axvspan(event.onset_time, event.offset_time, alpha=alpha, color=color)

            # 변곡점 markers
            ch_inflections = [ip for ip in self.eag_inflections if ip.channel == ch]
            for ip in ch_inflections:
                marker = '^' if ip.direction == 'peak' else 'v'
                color = '#E53935' if ip.direction == 'peak' else '#1E88E5'
                ax.plot(ip.time, ip.amplitude, marker=marker, color=color,
                        markersize=4, markeredgewidth=0.5, alpha=0.7)

        # --- GRF ---
        ax_grf = axes[EEG_CHANNELS]
        ax_grf.plot(self.unified_time_grf, self.grf_left,
                    color='#2196F3', linewidth=0.8, label='Left GRF')
        ax_grf.plot(self.unified_time_grf, self.grf_right,
                    color='#F44336', linewidth=0.8, label='Right GRF')
        ax_grf.set_ylabel('GRF (kg)', fontsize=9)
        ax_grf.set_xlabel('Time (sec)', fontsize=10)
        ax_grf.grid(True, alpha=0.3)
        ax_grf.legend(loc='upper right', fontsize=8)
        ax_grf.set_xlim(0, self.overlap_duration)

        if self.body_weight > 0:
            ax_grf.axhline(y=self.body_weight / 2, color='gray',
                           linestyle=':', alpha=0.4, linewidth=0.8)

        # GRF 이벤트 shading + 번호 라벨
        for event in self.grf_events:
            color, alpha = event_colors[event.intensity]
            ax_grf.axvspan(event.onset_time, event.offset_time, alpha=alpha, color=color)
            mid = (event.onset_time + event.offset_time) / 2
            ax_grf.text(mid, ax_grf.get_ylim()[1] * 0.95,
                        f'E{event.event_id}', ha='center', va='top',
                        fontsize=7, fontweight='bold', alpha=0.7)

        # 범례 추가 (이벤트 강도)
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor='orange', alpha=0.4, label='Strong (>0.80)'),
            Patch(facecolor='#FFD700', alpha=0.4, label='Moderate (0.50-0.80)'),
            Patch(facecolor='gray', alpha=0.3, label='Mild (0.30-0.50)'),
        ]
        axes[0].legend(handles=legend_patches, loc='upper right', fontsize=7, framealpha=0.8)

        fig.suptitle(
            f'EAG-GRF Synchronized Analysis (LP {self.config.lowpass_cutoff}Hz + {self.config.drift_method})\n'
            f'Subject: {self.pair.subject_name}, Session: {self.pair.session_name}',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프 저장: {save_path}")

        plt.close(fig)

    # ==================== Statistics ====================

    def compute_statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """이벤트별/변곡점별 통계를 계산한다.

        Returns:
            (events_df, inflections_df, summary_dict)
        """
        # --- Events DataFrame ---
        event_rows = []
        for e in self.grf_events:
            row = {
                'event_id': e.event_id,
                'onset_sec': round(e.onset_time, 3),
                'offset_sec': round(e.offset_time, 3),
                'duration_sec': round(e.duration, 3),
                'peak_imbalance': round(e.peak_imbalance, 3),
                'dominant_side': e.dominant_side,
                'intensity': e.intensity,
                'peak_left_grf': round(e.peak_left_grf, 1),
                'peak_right_grf': round(e.peak_right_grf, 1),
            }
            # 채널별 inflection 수
            event_infl = [ip for ip in self.eag_inflections if ip.event_id == e.event_id]
            row['inflections_total'] = len(event_infl)
            for ch in range(EEG_CHANNELS):
                row[f'inflections_ch{ch+1}'] = len([ip for ip in event_infl if ip.channel == ch])

            # 이벤트 onset 후 첫 inflection latency (평균)
            post_onset = [ip.latency for ip in event_infl if ip.latency >= 0]
            row['mean_latency_sec'] = round(np.mean(post_onset), 3) if post_onset else np.nan

            event_rows.append(row)

        events_df = pd.DataFrame(event_rows)

        # --- Inflections DataFrame ---
        infl_rows = []
        for ip in self.eag_inflections:
            infl_rows.append({
                'event_id': ip.event_id,
                'channel': ip.channel + 1,
                'time_sec': round(ip.time, 3),
                'amplitude_uv': round(ip.amplitude, 1),
                'direction': ip.direction,
                'latency_sec': round(ip.latency, 3),
            })
        inflections_df = pd.DataFrame(infl_rows)

        # --- Summary ---
        summary = {
            'subject': self.pair.subject_name,
            'session': self.pair.session_name,
            'overlap_duration_sec': round(self.overlap_duration, 1),
            'body_weight_kg': round(self.body_weight, 1),
            'time_offset_sec': round(self.time_offset, 1),
            'total_events': len(self.grf_events),
            'strong_events': len([e for e in self.grf_events if e.intensity == 'strong']),
            'moderate_events': len([e for e in self.grf_events if e.intensity == 'moderate']),
            'mild_events': len([e for e in self.grf_events if e.intensity == 'mild']),
            'total_inflections': len(self.eag_inflections),
        }

        # 강도별 latency 통계
        for intensity in ['strong', 'moderate', 'mild']:
            event_ids = {e.event_id for e in self.grf_events if e.intensity == intensity}
            latencies = [ip.latency for ip in self.eag_inflections
                         if ip.event_id in event_ids and ip.latency >= 0]
            if latencies:
                summary[f'{intensity}_mean_latency'] = round(np.mean(latencies), 3)
                summary[f'{intensity}_std_latency'] = round(np.std(latencies), 3)

        # 채널별 inflection 수 (mean ± std per event)
        if self.grf_events:
            for ch in range(EEG_CHANNELS):
                counts = []
                for e in self.grf_events:
                    n = len([ip for ip in self.eag_inflections
                             if ip.event_id == e.event_id and ip.channel == ch])
                    counts.append(n)
                summary[f'ch{ch+1}_mean_inflections'] = round(np.mean(counts), 2)
                summary[f'ch{ch+1}_std_inflections'] = round(np.std(counts), 2)

        return events_df, inflections_df, summary

    def save_statistics(self, output_dir: str):
        """통계를 CSV로 저장하고 콘솔에 요약을 출력한다."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.pair.session_name

        events_df, inflections_df, summary = self.compute_statistics()

        # CSV 저장
        events_path = output_dir / f'{prefix}_events.csv'
        inflections_path = output_dir / f'{prefix}_inflections.csv'
        summary_path = output_dir / f'{prefix}_summary.csv'

        events_df.to_csv(events_path, index=False)
        inflections_df.to_csv(inflections_path, index=False)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

        print(f"\n통계 저장:")
        print(f"  {events_path}")
        print(f"  {inflections_path}")
        print(f"  {summary_path}")

        # 콘솔 요약
        print(f"\n{'='*60}")
        print(f"=== 세션 요약: {self.pair.subject_name} / {self.pair.session_name} ===")
        print(f"{'='*60}")
        print(f"  중첩 구간: {summary['overlap_duration_sec']}초")
        print(f"  체중: {summary['body_weight_kg']}kg")
        print(f"  시간 오프셋: {summary['time_offset_sec']:+.1f}초")
        print(f"  GRF 이벤트: {summary['total_events']}개 "
              f"(Strong:{summary['strong_events']}, "
              f"Moderate:{summary['moderate_events']}, "
              f"Mild:{summary['mild_events']})")
        print(f"  EAG 변곡점: {summary['total_inflections']}개")

        # 강도별 latency
        for intensity in ['strong', 'moderate', 'mild']:
            key_m = f'{intensity}_mean_latency'
            key_s = f'{intensity}_std_latency'
            if key_m in summary:
                print(f"  {intensity.capitalize()} latency: "
                      f"{summary[key_m]:.3f} ± {summary[key_s]:.3f}초")

        # 채널별 inflection
        print(f"\n  채널별 변곡점 수 (per event, mean ± std):")
        for ch in range(EEG_CHANNELS):
            m = summary.get(f'ch{ch+1}_mean_inflections', 0)
            s = summary.get(f'ch{ch+1}_std_inflections', 0)
            print(f"    Ch{ch+1}: {m:.1f} ± {s:.1f}")

        print(f"{'='*60}\n")

        return events_df, inflections_df, summary


# ==================== Interactive Selection ====================

def interactive_select_sync() -> Optional[List[SessionPair]]:
    """인터랙티브하게 분석할 세션을 선택한다."""
    data_dir = get_data_dir()
    if not os.path.exists(data_dir):
        print(f"data 폴더가 없습니다: {data_dir}")
        return None

    subjects = list_subjects(data_dir)
    if not subjects:
        print("data 폴더에 피험자 데이터가 없습니다.")
        return None

    print("\n" + "=" * 60)
    print("=== 피험자 선택 (EAG-GRF 동기화 분석) ===")
    print("=" * 60)
    for i, (display_name, path) in enumerate(subjects):
        pair_count = len(find_all_pairs(path))
        print(f"  {i}: {display_name} ({pair_count}개 세션 쌍)")
    print(f"  a: 전체 분석")
    print("=" * 60)

    while True:
        choice = input(f"피험자 번호를 선택하세요 (0-{len(subjects)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return find_all_pairs(data_dir)
        try:
            idx = int(choice)
            if 0 <= idx < len(subjects):
                break
            print(f"0-{len(subjects)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    _, subject_dir = subjects[idx]
    all_pairs = find_all_pairs(subject_dir)

    if not all_pairs:
        print("EAG+GRF 쌍을 찾을 수 없습니다.")
        return None

    if len(all_pairs) == 1:
        print(f"\n선택된 세션: {all_pairs[0].session_name}")
        return all_pairs

    print("\n" + "=" * 60)
    print(f"=== 세션 선택 ({os.path.basename(subject_dir)}) ===")
    print("=" * 60)
    for i, pair in enumerate(all_pairs):
        print(f"  {i}: {pair.session_name} ({os.path.basename(pair.session_dir)})")
    print(f"  a: 전체 분석")
    print("=" * 60)

    while True:
        choice = input(f"세션 번호를 선택하세요 (0-{len(all_pairs)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return all_pairs
        try:
            idx = int(choice)
            if 0 <= idx < len(all_pairs):
                return [all_pairs[idx]]
            print(f"0-{len(all_pairs)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")


# ==================== CLI ====================

def get_output_dir(pair: SessionPair, override: Optional[str] = None) -> Path:
    """결과 저장 경로를 결정한다."""
    if override:
        return Path(override)
    result_base = Path(get_result_dir())
    return result_base / pair.subject_name


def main():
    parser = argparse.ArgumentParser(
        description='EAG-GRF 동기화 분석 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 인터랙티브 모드
  python sync_analyzer.py

  # 특정 세션 분석
  python sync_analyzer.py --dir ./data/피험자/OpenBCISession_...

  # 피험자 전체 세션 분석
  python sync_analyzer.py --subject ./data/피험자/

  # 전체 배치 분석
  python sync_analyzer.py --all

  # 이벤트 감지 임계값 변경
  python sync_analyzer.py --imbalance-threshold 0.50

  # 통계만 출력 (그래프 생략)
  python sync_analyzer.py --stats-only
        """
    )

    # Input
    parser.add_argument('--dir', '-d', type=str, help='분석할 세션 디렉토리')
    parser.add_argument('--subject', '-s', type=str, help='피험자 디렉토리 (전체 세션 분석)')
    parser.add_argument('--all', action='store_true', help='전체 배치 분석')
    parser.add_argument('--interactive', '-i', action='store_true', help='인터랙티브 모드')

    # Output
    parser.add_argument('--output', '-o', type=str, help='결과 저장 디렉토리')
    parser.add_argument('--stats-only', action='store_true', help='통계만 출력')

    # EAG filter
    parser.add_argument('--lowpass', '-lp', type=float, default=5.0,
                        help='저역통과 필터 (Hz, 기본: 5)')
    parser.add_argument('--no-lowpass', action='store_true')
    parser.add_argument('--no-drift', action='store_true')
    parser.add_argument('--drift-method', type=str, default='detrend',
                        choices=['detrend', 'moving', 'none'])

    # GRF event detection
    parser.add_argument('--imbalance-threshold', type=float, default=0.30,
                        help='이벤트 감지 imbalance 임계값 (기본: 0.30)')
    parser.add_argument('--min-event-duration', type=float, default=0.10,
                        help='최소 이벤트 지속시간 (초, 기본: 0.10)')
    parser.add_argument('--merge-gap', type=float, default=0.25,
                        help='이벤트 병합 간격 (초, 기본: 0.25)')

    # EAG inflection detection
    parser.add_argument('--window-before', type=float, default=2.0,
                        help='이벤트 전 분석 윈도우 (초, 기본: 2.0)')
    parser.add_argument('--window-after', type=float, default=3.0,
                        help='이벤트 후 분석 윈도우 (초, 기본: 3.0)')
    parser.add_argument('--min-prominence', type=float, default=10.0,
                        help='변곡점 최소 prominence (µV, 기본: 10.0)')

    # Timezone
    parser.add_argument('--utc-offset', type=int, default=9,
                        help='UTC 오프셋 (기본: 9, KST)')

    args = parser.parse_args()

    # FilterConfig
    config = FilterConfig()
    config.lowpass_enabled = not args.no_lowpass
    config.lowpass_cutoff = args.lowpass
    config.drift_enabled = not args.no_drift
    config.drift_method = args.drift_method if not args.no_drift else "none"
    config.start_time = 0.0

    # Determine session pairs
    pairs = None
    if args.dir:
        pair = find_session_pair(args.dir)
        pairs = [pair] if pair else None
    elif args.subject:
        pairs = find_all_pairs(args.subject)
    elif args.all:
        data_dir = get_data_dir()
        pairs = find_all_pairs(data_dir)
    elif args.interactive:
        pairs = interactive_select_sync()
    else:
        # Default: interactive if data folder exists
        data_dir = get_data_dir()
        if os.path.exists(data_dir):
            pairs = interactive_select_sync()
        else:
            print("data 폴더가 없습니다. --dir 또는 --subject 옵션을 사용하세요.")
            return

    if not pairs:
        print("분석할 세션 쌍을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(pairs)}개 세션 분석 시작\n")

    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {pair.subject_name} / {pair.session_name}")
        print("-" * 60)

        try:
            analyzer = SyncAnalyzer(pair, config=config, utc_offset=args.utc_offset)
            analyzer.run_analysis(
                imbalance_threshold=args.imbalance_threshold,
                min_duration_sec=args.min_event_duration,
                merge_gap_sec=args.merge_gap,
                window_before=args.window_before,
                window_after=args.window_after,
                min_prominence=args.min_prominence,
            )

            output_dir = get_output_dir(pair, args.output)

            if not args.stats_only:
                plot_path = str(output_dir / f'{pair.session_name}_sync.png')
                analyzer.plot_synchronized(save_path=plot_path)

            analyzer.save_statistics(str(output_dir))

        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
