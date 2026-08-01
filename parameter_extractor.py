"""
Phase 1 + Phase 3 Parameter Extractor — 변곡점 기반 파라미터 추출

Phase 1 (시간 영역):
  GRF 단독:  time_to_peak, recovery_time, symmetry_index, sway_variability, event_frequency
  EAG-GRF:   onset_latency, peak_latency (채널별)

Phase 3 (주파수 영역):
  EAG-GRF coherence: 채널별 magnitude-squared coherence (0-10Hz)
  ERSP: Event-Related Spectral Perturbation (이벤트 전후 주파수 파워 변화)

사용법:
  # Phase 1 전체 batch
  python3 parameter_extractor.py --batch

  # Phase 1 + Phase 3
  python3 parameter_extractor.py --batch --phase3

  # 특정 피험자만
  python3 parameter_extractor.py --batch --subject 주창민 --phase3
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import coherence as scipy_coherence, spectrogram as scipy_spectrogram
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from pathlib import Path
import argparse
import traceback

from sync_analyzer import (
    SyncAnalyzer, SessionPair, GRFEvent, EAGInflection,
    find_all_pairs, SAMPLE_RATE
)
from eag_analyzer import EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS, FilterConfig, EAGAnalyzer
from frequency_analyzer import compute_noise_metrics, flag_noise
from grf_triggered_annotator import compute_offset, extract_responses

GRF_SAMPLE_RATE = 250  # Kinvent K-Plate


# ==================== Noise Evaluation ====================

def evaluate_channel_quality(analyzer_eag: 'EAGAnalyzer') -> dict:
    """채널별 노이즈 평가 → quality_flags dict 반환.

    Returns:
        {0: {'flag': 'PASS', 'snr_db': 15.2, ...}, 1: {...}, ...}
    """
    quality = {}
    for ch in range(EEG_CHANNELS):
        metrics = compute_noise_metrics(analyzer_eag, ch)
        flags = flag_noise(metrics)
        quality[ch] = {
            'flag': flags[0] if flags else 'PASS',
            'flags': flags,
            'snr_db': metrics.get('snr_db', float('nan')),
            'power_hf_ratio': metrics.get('power_hf_ratio', float('nan')),
        }
    return quality


# ==================== Phase 1 Parameter Dataclasses ====================

@dataclass
class GRFEventParams:
    """Phase 1 GRF parameters for a single event."""
    event_id: int
    onset_time: float
    duration: float
    peak_imbalance: float
    dominant_side: str
    intensity: str

    # Phase 1 new parameters
    time_to_peak: float       # onset → peak imbalance 도달 시간 (s)
    recovery_time: float      # peak → baseline 복귀 시간 (s), NaN if not recovered
    peak_imbalance_time: float  # peak imbalance 시점 (absolute time)


@dataclass
class SessionGRFParams:
    """Phase 1 GRF session-level parameters."""
    symmetry_index: float     # mean(L-R)/mean(L+R) 전체 구간
    sway_variability: float   # imbalance의 std (안정 구간)
    event_frequency: float    # events / minute
    n_events: int
    n_strong: int
    n_moderate: int
    n_mild: int
    mean_duration: float
    mean_time_to_peak: float
    mean_recovery_time: float  # NaN 제외


@dataclass
class EAGGRFCrossParams:
    """Phase 1 EAG-GRF cross parameters for a single event × channel."""
    event_id: int
    channel: int
    channel_name: str

    # First Significant Inflection (FSI) — GRF onset 후 첫 변곡점
    fsi_latency: float         # GRF onset 대비 FSI 시간차 (s), 음수=선행
    fsi_amplitude: float       # FSI 진폭 (µV)
    fsi_direction: str         # 'peak' or 'valley'

    # Maximum Inflection (MI) — 이벤트 구간 내 최대 진폭 변곡점
    mi_latency: float          # GRF onset 대비 MI 시간차 (s)
    mi_amplitude: float        # MI 진폭 (µV)
    mi_direction: str

    # GRF peak 대비 EAG peak latency
    peak_latency: float        # GRF peak_imbalance_time 대비 MI 시간차 (s)


# ==================== Extraction Functions ====================

def extract_grf_event_params(analyzer: SyncAnalyzer) -> List[GRFEventParams]:
    """GRF 이벤트별 Phase 1 파라미터 추출.

    time_to_peak: onset부터 peak imbalance까지의 시간
    recovery_time: peak부터 imbalance가 baseline(threshold=0.15)으로 돌아올 때까지
    """
    results = []

    total = analyzer.grf_left + analyzer.grf_right
    safe_total = np.maximum(total, 0.1)
    imbalance = np.abs(analyzer.grf_left - analyzer.grf_right) / safe_total

    time_grf = analyzer.unified_time_grf
    fs = SAMPLE_RATE

    recovery_threshold = 0.15  # imbalance가 이 아래로 떨어지면 recovery

    for event in analyzer.grf_events:
        # 이벤트 구간 인덱싱
        onset_idx = np.searchsorted(time_grf, event.onset_time)
        offset_idx = np.searchsorted(time_grf, event.offset_time)

        if offset_idx <= onset_idx:
            continue

        seg_imb = imbalance[onset_idx:offset_idx]
        seg_time = time_grf[onset_idx:offset_idx]

        # time_to_peak: onset → peak imbalance
        peak_local_idx = np.argmax(seg_imb)
        time_to_peak = float(seg_time[peak_local_idx] - seg_time[0])
        peak_imbalance_time = float(seg_time[peak_local_idx])

        # recovery_time: peak → imbalance < recovery_threshold
        post_peak_imb = imbalance[onset_idx + peak_local_idx:]
        post_peak_time = time_grf[onset_idx + peak_local_idx:]

        recovery_time = float('nan')
        if len(post_peak_imb) > 0:
            below = np.where(post_peak_imb < recovery_threshold)[0]
            if len(below) > 0:
                recovery_time = float(post_peak_time[below[0]] - seg_time[peak_local_idx])

        results.append(GRFEventParams(
            event_id=event.event_id,
            onset_time=event.onset_time,
            duration=event.duration,
            peak_imbalance=event.peak_imbalance,
            dominant_side=event.dominant_side,
            intensity=event.intensity,
            time_to_peak=time_to_peak,
            recovery_time=recovery_time,
            peak_imbalance_time=peak_imbalance_time,
        ))

    return results


def extract_session_grf_params(analyzer: SyncAnalyzer,
                                event_params: List[GRFEventParams]) -> SessionGRFParams:
    """GRF 세션 레벨 파라미터 추출.

    symmetry_index: 전체 안정 구간(이벤트 외)의 좌-우 비대칭
    sway_variability: 안정 구간 imbalance std
    """
    total = analyzer.grf_left + analyzer.grf_right
    safe_total = np.maximum(total, 0.1)
    signed_imb = (analyzer.grf_left - analyzer.grf_right) / safe_total
    abs_imb = np.abs(signed_imb)

    time_grf = analyzer.unified_time_grf

    # 이벤트 구간 마스크 (이벤트 외 = 안정 구간)
    stable_mask = np.ones(len(time_grf), dtype=bool)
    for event in analyzer.grf_events:
        event_mask = (time_grf >= event.onset_time) & (time_grf <= event.offset_time)
        stable_mask &= ~event_mask

    # symmetry_index: mean(L-R) / mean(L+R), 안정 구간
    if np.any(stable_mask):
        stable_left = analyzer.grf_left[stable_mask]
        stable_right = analyzer.grf_right[stable_mask]
        mean_diff = np.mean(stable_left - stable_right)
        mean_sum = np.mean(stable_left + stable_right)
        symmetry_index = float(mean_diff / mean_sum) if mean_sum > 0 else 0.0
        sway_variability = float(np.std(abs_imb[stable_mask]))
    else:
        symmetry_index = 0.0
        sway_variability = 0.0

    # event_frequency: events per minute
    duration_min = analyzer.overlap_duration / 60.0
    event_frequency = len(analyzer.grf_events) / duration_min if duration_min > 0 else 0.0

    # Event counts
    n_strong = sum(1 for e in analyzer.grf_events if e.intensity == 'strong')
    n_moderate = sum(1 for e in analyzer.grf_events if e.intensity == 'moderate')
    n_mild = sum(1 for e in analyzer.grf_events if e.intensity == 'mild')

    # Mean event params
    durations = [ep.duration for ep in event_params]
    ttps = [ep.time_to_peak for ep in event_params]
    recs = [ep.recovery_time for ep in event_params if not np.isnan(ep.recovery_time)]

    return SessionGRFParams(
        symmetry_index=symmetry_index,
        sway_variability=sway_variability,
        event_frequency=event_frequency,
        n_events=len(analyzer.grf_events),
        n_strong=n_strong,
        n_moderate=n_moderate,
        n_mild=n_mild,
        mean_duration=float(np.mean(durations)) if durations else 0.0,
        mean_time_to_peak=float(np.mean(ttps)) if ttps else 0.0,
        mean_recovery_time=float(np.mean(recs)) if recs else float('nan'),
    )


def extract_eag_grf_cross_params(analyzer: SyncAnalyzer,
                                  event_params: List[GRFEventParams]
                                  ) -> List[EAGGRFCrossParams]:
    """EAG-GRF 교차 파라미터 추출: 이벤트별 × 채널별.

    FSI(First Significant Inflection): GRF onset 이후 첫 EAG 변곡점
    MI(Maximum Inflection): 이벤트 구간 내 최대 진폭 변곡점
    peak_latency: GRF peak imbalance 시점 대비 MI 시간차
    """
    results = []
    ep_map = {ep.event_id: ep for ep in event_params}

    for event in analyzer.grf_events:
        ep = ep_map.get(event.event_id)
        if ep is None:
            continue

        event_inflections = [ip for ip in analyzer.eag_inflections
                             if ip.event_id == event.event_id]

        for ch in range(EEG_CHANNELS):
            ch_infl = [ip for ip in event_inflections if ip.channel == ch]
            ch_infl_sorted = sorted(ch_infl, key=lambda x: x.time)

            # FSI: onset 이후 (또는 가장 가까운) 첫 변곡점
            post_onset = [ip for ip in ch_infl_sorted if ip.time >= event.onset_time - 0.5]
            if post_onset:
                fsi = post_onset[0]
                fsi_latency = fsi.time - event.onset_time
                fsi_amplitude = fsi.amplitude
                fsi_direction = fsi.direction
            else:
                fsi_latency = float('nan')
                fsi_amplitude = float('nan')
                fsi_direction = ''

            # MI: 이벤트 구간 내 최대 |amplitude| 변곡점
            event_window = [ip for ip in ch_infl_sorted
                            if event.onset_time - 1.0 <= ip.time <= event.offset_time + 1.0]
            if event_window:
                mi = max(event_window, key=lambda x: abs(x.amplitude))
                mi_latency = mi.time - event.onset_time
                mi_amplitude = mi.amplitude
                mi_direction = mi.direction
                peak_latency = mi.time - ep.peak_imbalance_time
            else:
                mi_latency = float('nan')
                mi_amplitude = float('nan')
                mi_direction = ''
                peak_latency = float('nan')

            results.append(EAGGRFCrossParams(
                event_id=event.event_id,
                channel=ch,
                channel_name=CHANNEL_NAMES[ch],
                fsi_latency=fsi_latency,
                fsi_amplitude=fsi_amplitude,
                fsi_direction=fsi_direction,
                mi_latency=mi_latency,
                mi_amplitude=mi_amplitude,
                mi_direction=mi_direction,
                peak_latency=peak_latency,
            ))

    return results


# ==================== Phase 2: EAG Standalone + Cross (APA, Co-activation) ====================

@dataclass
class EAGChannelParams:
    """Phase 2: EAG 채널별 세션 레벨 파라미터."""
    channel: int
    channel_name: str
    rms_overall: float          # 전체 구간 RMS (µV)
    rms_stable: float           # 안정 구간 RMS (이벤트 외)
    rms_event: float            # 이벤트 구간 RMS
    channel_dominance: float    # 이 채널의 RMS / 전체 채널 RMS 합 (0~1)
    mean_peak_amplitude: float  # 변곡점 peak 진폭 평균
    mean_peak_to_peak: float    # 연속 peak→valley 진폭차 평균
    inflection_rate: float      # 변곡점 수 / 이벤트 시간 (Hz)


@dataclass
class APAParams:
    """Phase 2: Anticipatory Postural Adjustment (이벤트별 × 채널별)."""
    event_id: int
    channel: int
    # APA 지표: GRF onset 전 구간의 EAG 변화
    has_apa: bool               # onset 전 2초 내 유의미한 변곡점 존재 여부
    apa_amplitude: float        # onset 전 변곡점 진폭 (µV), 없으면 NaN
    apa_latency: float          # onset 전 변곡점 시간 (음수, s), 없으면 NaN
    # Reactive 지표: GRF onset 후 구간
    reactive_amplitude: float   # onset 후 첫 변곡점 진폭
    reactive_latency: float     # onset 후 첫 변곡점까지 시간


@dataclass
class CoActivationParams:
    """Phase 2: Co-activation index (이벤트별)."""
    event_id: int
    co_activation_index: float  # 동시 활성 채널 비율 (0~1)
    n_active_channels: int      # 이벤트 구간 내 활성 채널 수
    dominant_channel: int       # 최대 진폭 채널
    activation_pattern: str     # 예: "1,3,5,6" (활성 채널 목록)


def extract_phase2_eag_channels(analyzer: SyncAnalyzer) -> List[EAGChannelParams]:
    """Phase 2: EAG 채널별 파라미터 (RMS, dominance, inflection stats)."""
    time_eag = analyzer.unified_time_eag
    filtered = analyzer.eag_filtered

    # 이벤트/안정 구간 마스크
    stable_mask = np.ones(len(time_eag), dtype=bool)
    event_mask = np.zeros(len(time_eag), dtype=bool)
    total_event_duration = 0.0

    for ev in analyzer.grf_events:
        m = (time_eag >= ev.onset_time) & (time_eag <= ev.offset_time)
        event_mask |= m
        stable_mask &= ~m
        total_event_duration += ev.duration

    # 전체 채널 RMS (dominance 계산용)
    all_rms = []
    for ch in range(EEG_CHANNELS):
        all_rms.append(np.sqrt(np.mean(filtered[:, ch] ** 2)))
    total_rms_sum = sum(all_rms)

    results = []
    for ch in range(EEG_CHANNELS):
        data = filtered[:, ch]

        rms_overall = float(np.sqrt(np.mean(data ** 2)))
        rms_stable = float(np.sqrt(np.mean(data[stable_mask] ** 2))) if np.any(stable_mask) else 0.0
        rms_event = float(np.sqrt(np.mean(data[event_mask] ** 2))) if np.any(event_mask) else 0.0
        dominance = rms_overall / total_rms_sum if total_rms_sum > 0 else 0.0

        # 변곡점 통계
        ch_infl = [ip for ip in analyzer.eag_inflections if ip.channel == ch]
        peaks = [ip for ip in ch_infl if ip.direction == 'peak']
        valleys = [ip for ip in ch_infl if ip.direction == 'valley']

        mean_peak_amp = float(np.mean([abs(ip.amplitude) for ip in peaks])) if peaks else 0.0

        # peak_to_peak: 연속 peak-valley 진폭차
        ch_infl_sorted = sorted(ch_infl, key=lambda x: x.time)
        p2p_values = []
        for j in range(len(ch_infl_sorted) - 1):
            if ch_infl_sorted[j].direction != ch_infl_sorted[j+1].direction:
                p2p_values.append(abs(ch_infl_sorted[j].amplitude - ch_infl_sorted[j+1].amplitude))
        mean_p2p = float(np.mean(p2p_values)) if p2p_values else 0.0

        # inflection_rate: 이벤트 구간 내 변곡점 밀도
        infl_rate = len(ch_infl) / total_event_duration if total_event_duration > 0 else 0.0

        results.append(EAGChannelParams(
            channel=ch,
            channel_name=CHANNEL_NAMES[ch],
            rms_overall=round(rms_overall, 2),
            rms_stable=round(rms_stable, 2),
            rms_event=round(rms_event, 2),
            channel_dominance=round(dominance, 4),
            mean_peak_amplitude=round(mean_peak_amp, 2),
            mean_peak_to_peak=round(mean_p2p, 2),
            inflection_rate=round(infl_rate, 2),
        ))

    return results


def extract_phase2_apa(analyzer: SyncAnalyzer,
                       event_params: List[GRFEventParams],
                       pre_window: float = 2.0,
                       post_window: float = 1.0) -> List[APAParams]:
    """Phase 2: Anticipatory Postural Adjustment + Reactive amplitude.

    각 GRF 이벤트별 × 각 채널에서:
    - APA: onset 전 pre_window 내 변곡점 유무 및 진폭
    - Reactive: onset 후 post_window 내 첫 변곡점 진폭 및 latency
    """
    results = []

    for ev in analyzer.grf_events:
        event_infl = [ip for ip in analyzer.eag_inflections if ip.event_id == ev.event_id]

        for ch in range(EEG_CHANNELS):
            ch_infl = sorted([ip for ip in event_infl if ip.channel == ch],
                             key=lambda x: x.time)

            # APA: onset 전 변곡점 (onset - pre_window ~ onset)
            pre_infl = [ip for ip in ch_infl
                        if ev.onset_time - pre_window <= ip.time < ev.onset_time]

            has_apa = len(pre_infl) > 0
            if has_apa:
                # onset에 가장 가까운 변곡점
                closest_pre = max(pre_infl, key=lambda x: x.time)
                apa_amplitude = closest_pre.amplitude
                apa_latency = closest_pre.time - ev.onset_time  # 음수
            else:
                apa_amplitude = float('nan')
                apa_latency = float('nan')

            # Reactive: onset 후 첫 변곡점
            post_infl = [ip for ip in ch_infl
                         if ev.onset_time <= ip.time <= ev.onset_time + post_window]
            if post_infl:
                first_post = post_infl[0]
                reactive_amplitude = first_post.amplitude
                reactive_latency = first_post.time - ev.onset_time
            else:
                # post_window 넓혀서 재시도 (3초까지)
                post_infl_wide = [ip for ip in ch_infl
                                  if ev.onset_time <= ip.time <= ev.onset_time + 3.0]
                if post_infl_wide:
                    first_post = post_infl_wide[0]
                    reactive_amplitude = first_post.amplitude
                    reactive_latency = first_post.time - ev.onset_time
                else:
                    reactive_amplitude = float('nan')
                    reactive_latency = float('nan')

            results.append(APAParams(
                event_id=ev.event_id,
                channel=ch,
                has_apa=has_apa,
                apa_amplitude=round(apa_amplitude, 2) if not np.isnan(apa_amplitude) else float('nan'),
                apa_latency=round(apa_latency, 4) if not np.isnan(apa_latency) else float('nan'),
                reactive_amplitude=round(reactive_amplitude, 2) if not np.isnan(reactive_amplitude) else float('nan'),
                reactive_latency=round(reactive_latency, 4) if not np.isnan(reactive_latency) else float('nan'),
            ))

    return results


def extract_phase2_coactivation(analyzer: SyncAnalyzer,
                                 activation_threshold_percentile: float = 75
                                 ) -> List[CoActivationParams]:
    """Phase 2: Co-activation index (이벤트별).

    각 이벤트 구간에서 RMS가 threshold 이상인 채널 수 / 전체 채널 수.
    threshold = 안정 구간 RMS의 percentile.
    """
    time_eag = analyzer.unified_time_eag
    filtered = analyzer.eag_filtered

    # 안정 구간 RMS 기반 threshold (채널별)
    stable_mask = np.ones(len(time_eag), dtype=bool)
    for ev in analyzer.grf_events:
        m = (time_eag >= ev.onset_time) & (time_eag <= ev.offset_time)
        stable_mask &= ~m

    ch_thresholds = []
    for ch in range(EEG_CHANNELS):
        if np.any(stable_mask):
            stable_rms_vals = np.abs(filtered[stable_mask, ch])
            threshold = np.percentile(stable_rms_vals, activation_threshold_percentile)
        else:
            threshold = np.percentile(np.abs(filtered[:, ch]), activation_threshold_percentile)
        ch_thresholds.append(threshold)

    results = []
    for ev in analyzer.grf_events:
        mask = (time_eag >= ev.onset_time) & (time_eag <= ev.offset_time)
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue

        active_channels = []
        ch_rms_values = []
        for ch in range(EEG_CHANNELS):
            seg = filtered[indices, ch]
            seg_rms = float(np.sqrt(np.mean(seg ** 2)))
            ch_rms_values.append(seg_rms)
            if seg_rms > ch_thresholds[ch]:
                active_channels.append(ch)

        n_active = len(active_channels)
        co_activation = n_active / EEG_CHANNELS
        dominant = int(np.argmax(ch_rms_values))
        pattern = ",".join(str(ch + 1) for ch in active_channels)

        results.append(CoActivationParams(
            event_id=ev.event_id,
            co_activation_index=round(co_activation, 3),
            n_active_channels=n_active,
            dominant_channel=dominant + 1,
            activation_pattern=pattern,
        ))

    return results


# ==================== Phase 2 Visualization ====================

def plot_phase2_summary(eag_ch_params: List[EAGChannelParams],
                        apa_params: List[APAParams],
                        coact_params: List[CoActivationParams],
                        subject: str, session: str,
                        save_path: Optional[str] = None):
    """Phase 2 요약 시각화: 4패널."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Phase 2 Parameters — {subject} / {session}',
                 fontsize=13, fontweight='bold')

    # 1) Channel Dominance + RMS
    ax = axes[0, 0]
    channels = [p.channel_name for p in eag_ch_params]
    rms_event = [p.rms_event for p in eag_ch_params]
    rms_stable = [p.rms_stable for p in eag_ch_params]
    x = np.arange(len(channels))
    w = 0.35
    ax.bar(x - w/2, rms_stable, w, label='Stable RMS', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, rms_event, w, label='Event RMS', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel('RMS (µV)')
    ax.set_title('Channel RMS: Stable vs Event')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dominance 비율을 twin axis에
    ax2 = ax.twinx()
    dom = [p.channel_dominance for p in eag_ch_params]
    ax2.plot(x, dom, 'ko-', markersize=6, linewidth=1.5, label='Dominance')
    ax2.set_ylabel('Dominance (ratio)')
    ax2.set_ylim(0, max(dom) * 1.5)

    # 2) APA presence rate by channel
    ax = axes[0, 1]
    apa_rates = []
    for ch in range(EEG_CHANNELS):
        ch_apa = [p for p in apa_params if p.channel == ch]
        if ch_apa:
            rate = sum(1 for p in ch_apa if p.has_apa) / len(ch_apa)
        else:
            rate = 0
        apa_rates.append(rate * 100)
    ax.bar(range(EEG_CHANNELS), apa_rates, color=CHANNEL_COLORS, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(EEG_CHANNELS))
    ax.set_xticklabels([f'Ch{i+1}' for i in range(EEG_CHANNELS)])
    ax.set_ylabel('APA Rate (%)')
    ax.set_title('Anticipatory Activity (APA) Rate by Channel')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 3) Co-activation index distribution
    ax = axes[1, 0]
    if coact_params:
        cai = [p.co_activation_index for p in coact_params]
        ax.hist(cai, bins=np.arange(0, 1.125, 0.125), color='mediumpurple',
                alpha=0.8, edgecolor='black')
        ax.axvline(np.mean(cai), color='red', linestyle='--',
                   label=f'Mean: {np.mean(cai):.2f}')
        ax.legend()
    ax.set_xlabel('Co-activation Index')
    ax.set_ylabel('Count')
    ax.set_title('Co-activation Index Distribution')
    ax.grid(True, alpha=0.3)

    # 4) APA latency vs Reactive latency (채널별 scatter)
    ax = axes[1, 1]
    for ch in range(EEG_CHANNELS):
        ch_apa = [p for p in apa_params if p.channel == ch and p.has_apa
                  and not np.isnan(p.reactive_latency)]
        if ch_apa:
            apa_lat = [p.apa_latency for p in ch_apa]
            react_lat = [p.reactive_latency for p in ch_apa]
            ax.scatter(apa_lat, react_lat, color=CHANNEL_COLORS[ch],
                       alpha=0.5, s=15, label=f'Ch{ch+1}')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('APA Latency (s, negative = before onset)')
    ax.set_ylabel('Reactive Latency (s, after onset)')
    ax.set_title('APA vs Reactive Timing')
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==================== Phase 3: Frequency-domain Parameters ====================

@dataclass
class CoherenceResult:
    """Phase 3: EAG-GRF coherence (채널별)."""
    channel: int
    freq: np.ndarray            # 주파수 배열
    coh: np.ndarray             # coherence 값 (0~1)
    mean_coh_0_2hz: float       # 0-2Hz 대역 평균 coherence
    mean_coh_2_5hz: float       # 2-5Hz 대역 평균 coherence
    peak_coh_freq: float        # 최대 coherence 주파수
    peak_coh_value: float       # 최대 coherence 값


@dataclass
class ERSPResult:
    """Phase 3: Event-Related Spectral Perturbation (채널별)."""
    channel: int
    freq: np.ndarray
    time: np.ndarray
    ersp: np.ndarray            # dB change relative to baseline
    mean_ersp_0_2hz: float      # 이벤트 후 0-2Hz 대역 평균 ERSP (dB)
    mean_ersp_2_5hz: float      # 이벤트 후 2-5Hz 대역 평균 ERSP (dB)


def compute_eag_grf_coherence(analyzer: SyncAnalyzer,
                               nperseg: int = 512,
                               freq_max: float = 10.0) -> List[CoherenceResult]:
    """Phase 3: GRF imbalance와 각 EAG 채널 간 magnitude-squared coherence.

    EAG를 GRF 시간축에 보간 후 계산.
    """
    total = analyzer.grf_left + analyzer.grf_right
    safe_total = np.maximum(total, 0.1)
    grf_signal = np.abs(analyzer.grf_left - analyzer.grf_right) / safe_total
    grf_time = analyzer.unified_time_grf
    fs = GRF_SAMPLE_RATE

    actual_nperseg = min(nperseg, len(grf_signal) // 4)
    if actual_nperseg < 64:
        return []

    results = []
    for ch in range(EEG_CHANNELS):
        eag_signal = np.interp(grf_time, analyzer.unified_time_eag,
                               analyzer.eag_filtered[:, ch])

        f, cxy = scipy_coherence(grf_signal, eag_signal, fs=fs,
                                  nperseg=actual_nperseg, window='hann')

        freq_mask = f <= freq_max
        mask_0_2 = (f >= 0) & (f <= 2)
        mask_2_5 = (f > 2) & (f <= 5)

        mean_0_2 = float(np.mean(cxy[mask_0_2])) if np.any(mask_0_2) else 0
        mean_2_5 = float(np.mean(cxy[mask_2_5])) if np.any(mask_2_5) else 0

        cxy_limited = cxy[freq_mask]
        f_limited = f[freq_mask]
        peak_idx = np.argmax(cxy_limited) if len(cxy_limited) > 0 else 0

        results.append(CoherenceResult(
            channel=ch,
            freq=f_limited,
            coh=cxy_limited,
            mean_coh_0_2hz=round(mean_0_2, 4),
            mean_coh_2_5hz=round(mean_2_5, 4),
            peak_coh_freq=round(float(f_limited[peak_idx]), 2) if len(f_limited) > 0 else 0,
            peak_coh_value=round(float(cxy_limited[peak_idx]), 4) if len(cxy_limited) > 0 else 0,
        ))

    return results


def compute_ersp(analyzer: SyncAnalyzer,
                 window_before: float = 2.0,
                 window_after: float = 4.0,
                 nperseg: int = 128,
                 noverlap: int = 112,
                 freq_max: float = 10.0) -> List[ERSPResult]:
    """Phase 3: Event-Related Spectral Perturbation.

    각 GRF 이벤트 중심으로 EAG spectrogram을 구하고,
    baseline(이벤트 전 구간) 대비 dB 변화를 계산.
    moderate/strong 이벤트 평균으로 ERSP 산출.
    """
    fs = SAMPLE_RATE
    eag_time = analyzer.unified_time_eag

    results = []
    for ch in range(EEG_CHANNELS):
        all_spectrograms = []
        common_time = None
        common_freq = None

        for ev in analyzer.grf_events:
            if ev.intensity == 'mild':
                continue

            win_start = ev.onset_time - window_before
            win_end = ev.onset_time + window_after
            mask = (eag_time >= win_start) & (eag_time <= win_end)
            indices = np.where(mask)[0]

            if len(indices) < nperseg * 2:
                continue

            seg = analyzer.eag_filtered[indices, ch]
            f, t, Sxx = scipy_spectrogram(seg, fs=fs, nperseg=nperseg,
                                           noverlap=noverlap, window='hann')

            t_event = t - window_before  # onset = 0
            freq_mask = f <= freq_max
            Sxx_limited = Sxx[freq_mask, :]

            all_spectrograms.append(Sxx_limited)
            if common_time is None:
                common_time = t_event
                common_freq = f[freq_mask]

        if not all_spectrograms or common_time is None:
            continue

        # 길이 맞추기
        min_t = min(s.shape[1] for s in all_spectrograms)
        aligned = [s[:, :min_t] for s in all_spectrograms]
        common_time = common_time[:min_t]

        mean_spec = np.mean(aligned, axis=0)

        # Baseline: onset 전 구간
        baseline_mask = common_time < 0
        if np.any(baseline_mask):
            baseline = np.mean(mean_spec[:, baseline_mask], axis=1, keepdims=True)
            baseline = np.maximum(baseline, 1e-10)
            ersp_db = 10 * np.log10(mean_spec / baseline)
        else:
            ersp_db = np.zeros_like(mean_spec)

        # 대역별 평균 ERSP (이벤트 후)
        post_mask = common_time >= 0
        mask_0_2 = (common_freq >= 0) & (common_freq <= 2)
        mask_2_5 = (common_freq > 2) & (common_freq <= 5)

        ersp_post = ersp_db[:, post_mask] if np.any(post_mask) else ersp_db
        mean_0_2 = float(np.mean(ersp_post[mask_0_2, :])) if np.any(mask_0_2) else 0
        mean_2_5 = float(np.mean(ersp_post[mask_2_5, :])) if np.any(mask_2_5) else 0

        results.append(ERSPResult(
            channel=ch,
            freq=common_freq,
            time=common_time,
            ersp=ersp_db,
            mean_ersp_0_2hz=round(mean_0_2, 2),
            mean_ersp_2_5hz=round(mean_2_5, 2),
        ))

    return results


# ==================== Phase 3 Visualization ====================

def plot_coherence(coh_results: List[CoherenceResult],
                   subject: str, session: str,
                   save_path: Optional[str] = None):
    """8채널 EAG-GRF coherence 시각화."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'EAG-GRF Coherence — {subject} / {session}',
                 fontsize=13, fontweight='bold')

    for i, cr in enumerate(coh_results):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.plot(cr.freq, cr.coh, color=CHANNEL_COLORS[i], linewidth=1.5)
        ax.axhspan(0, 0.2, alpha=0.05, color='gray')
        ax.axvline(2.0, color='gray', linestyle='--', alpha=0.4)
        ax.axvline(5.0, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylim(0, max(0.5, cr.peak_coh_value * 1.3))
        ax.set_title(f'{CHANNEL_NAMES[i]}  |  0-2Hz:{cr.mean_coh_0_2hz:.3f}  '
                     f'2-5Hz:{cr.mean_coh_2_5hz:.3f}', fontsize=9)
        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('Coherence')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ersp(ersp_results: List[ERSPResult],
              subject: str, session: str,
              save_path: Optional[str] = None):
    """8채널 ERSP 히트맵 시각화."""
    n_ch = len(ersp_results)
    if n_ch == 0:
        return

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2.5 * n_ch), squeeze=False)
    fig.suptitle(f'Event-Related Spectral Perturbation — {subject} / {session}',
                 fontsize=13, fontweight='bold')

    vmax = max(np.percentile(np.abs(er.ersp), 95) for er in ersp_results)
    vmax = max(vmax, 1.0)

    for i, er in enumerate(ersp_results):
        ax = axes[i, 0]
        im = ax.pcolormesh(er.time, er.freq, er.ersp,
                           shading='gouraud', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
        ax.axvline(0, color='black', linewidth=2, label='GRF onset')
        ax.set_ylabel(f'{CHANNEL_NAMES[er.channel]}\nFreq (Hz)')
        plt.colorbar(im, ax=ax, label='dB', pad=0.01, aspect=15)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1, 0].set_xlabel('Time relative to GRF onset (s)')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==================== Batch Processing ====================

def process_single_session(pair: SessionPair, output_dir: Path, do_phase2: bool = True,
                           apply_corrected_offset: bool = True,
                           do_grf_triggered: bool = True
                           ) -> Tuple[Optional[pd.DataFrame],
                                      Optional[pd.DataFrame],
                                      Optional[pd.DataFrame],
                                      Optional[dict],
                                      Optional[pd.DataFrame],
                                      Optional[dict]]:
    """단일 세션 Phase 1 (+ Phase 2 + GRF-triggered) 파라미터 추출.

    Returns:
        (grf_event_df, session_df, cross_df, phase2_dict,
         grf_triggered_df, offset_row) or (None,)*6
    """
    try:
        sa = SyncAnalyzer(pair)

        # 수동 offset이 설정돼 있으면 최고 권위 (SyncAnalyzer가 생성자에서 이미 적용).
        # 이 경우 auto 재보정으로 덮어쓰지 않는다.
        from offset_manager import get_manual_offset
        has_manual = get_manual_offset(pair.subject_name, pair.session_name) is not None

        # Step 0: GRF-triggered offset 진단·보정 (edge↔GRF 정렬)
        off, trans, _signed, _grf_t = compute_offset(
            sa, ref_ch=0, recompute_offset=not has_manual)
        offset_row = {
            'subject': pair.subject_name, 'session': pair.session_name,
            'auto_offset': round(off.auto_offset, 4),
            'residual': round(off.residual, 4),
            'corrected_offset': round(off.corrected_offset, 4),
            'match_auto': round(off.match_rate_auto, 3),
            'match_corrected': round(off.match_rate_corrected, 3),
            'method': off.method,
        }
        # 교정이 채택되면 corrected offset으로 SyncAnalyzer 재구성 → 전체 Phase 1이 개선된 정렬 사용
        # (수동 offset이 있으면 sa는 이미 그 값을 쓰므로 재구성 불필요)
        if apply_corrected_offset and not has_manual and off.residual != 0.0:
            sa = SyncAnalyzer(pair, manual_offset=off.corrected_offset)
            _off2, trans, _s, _g = compute_offset(sa, ref_ch=0, recompute_offset=False)

        # Step 1: 노이즈 평가
        ch_quality = evaluate_channel_quality(sa.eag)
        n_good = sum(1 for v in ch_quality.values() if v['flag'] == 'PASS')

        # GRF-triggered 파라미터 (corrected offset 프레임에서, residual=0)
        grf_triggered_df = None
        if do_grf_triggered:
            te_corr = sa.unified_time_eag  # 이미 corrected offset 적용됨
            # 프로토콜(한발서기 4회 × 부하 시작/종료 = 8 이벤트)만 anchor로 쓴다.
            # 발구름·전후 양발 서기가 파라미터에 섞이지 않고, detect_grf_transitions가
            # 진폭 문턱 때문에 놓치던 가장 가벼운 단계도 포함된다.
            # cycle이 기대치(4회)와 다르면 검토 대상이므로 기존 전이를 그대로 쓴다
            # (edge_review.py --dir data 로 그런 세션을 추린다).
            import grf_triggered_annotator as _G
            _signed_c = _G.signed_imbalance(sa.grf_left, sa.grf_right)
            _rest, _cycles, _cinfo = _G.detect_load_cycles_expected(
                sa.unified_time_grf, _signed_c, sa.grf_left, sa.grf_right)
            anchors = (_G.cycles_to_transitions(_cycles, trans)
                       if len(_cycles) == _G.EXPECTED_CYCLES else trans)
            gt_resp = extract_responses(sa, anchors, te_corr, list(range(1, EEG_CHANNELS + 1)))
            grf_triggered_df = pd.DataFrame([asdict(r) for r in gt_resp])
            grf_triggered_df['subject'] = pair.subject_name
            grf_triggered_df['session'] = pair.session_name
            grf_triggered_df['n_cycles'] = len(_cycles)
            grf_triggered_df['protocol_ok'] = len(_cycles) == _G.EXPECTED_CYCLES
            # cycle별 실측 체중부하율(%) — 설계 20-50-80-100%의 개인별 실제값
            if len(_cycles) == _G.EXPECTED_CYCLES:
                _cyc_of = {i: _cycles[i // 2] for i in range(2 * len(_cycles))}
                grf_triggered_df['cycle_id'] = grf_triggered_df['trans_id'].map(
                    lambda i: _cyc_of[i].cycle_id if i in _cyc_of else None)
                grf_triggered_df['event_kind'] = grf_triggered_df['trans_id'].map(
                    lambda i: ('on' if i % 2 == 0 else 'off') if i in _cyc_of else '')
                grf_triggered_df['load_pct'] = grf_triggered_df['trans_id'].map(
                    lambda i: round(_cyc_of[i].load_pct, 1) if i in _cyc_of else None)
                grf_triggered_df['test_side'] = grf_triggered_df['trans_id'].map(
                    lambda i: _cyc_of[i].test_side if i in _cyc_of else '')
                # 부하/이탈 방향이 같으면 한쪽만 자동 채택하고 그 사실을 라벨로 남긴다
                grf_triggered_df = _G.label_single_sided(grf_triggered_df)

        sa.run_analysis()

        if not sa.grf_events:
            print(f"  ⚠️ {pair.subject_name}/{pair.session_name}: GRF 이벤트 없음")
            return None, None, None, None, grf_triggered_df, offset_row

        # Phase 1
        event_params = extract_grf_event_params(sa)
        session_params = extract_session_grf_params(sa, event_params)
        cross_params = extract_eag_grf_cross_params(sa, event_params)

        grf_event_df = pd.DataFrame([asdict(ep) for ep in event_params])
        grf_event_df['subject'] = pair.subject_name
        grf_event_df['session'] = pair.session_name

        session_row = asdict(session_params)
        session_row['subject'] = pair.subject_name
        session_row['session'] = pair.session_name
        session_row['n_good_channels'] = n_good
        session_df = pd.DataFrame([session_row])

        cross_df = pd.DataFrame([asdict(cp) for cp in cross_params])
        cross_df['subject'] = pair.subject_name
        cross_df['session'] = pair.session_name
        # quality_flag 추가 (채널별)
        cross_df['channel_quality'] = cross_df['channel'].map(
            lambda ch: ch_quality[ch]['flag'])
        cross_df['channel_snr_db'] = cross_df['channel'].map(
            lambda ch: ch_quality[ch]['snr_db'])

        # Phase 2
        phase2_dict = None
        if do_phase2:
            eag_ch_params = extract_phase2_eag_channels(sa)
            apa_params = extract_phase2_apa(sa, event_params)
            coact_params = extract_phase2_coactivation(sa)

            # Phase 2 시각화
            session_dir = output_dir / pair.subject_name
            session_dir.mkdir(parents=True, exist_ok=True)
            plot_phase2_summary(eag_ch_params, apa_params, coact_params,
                                pair.subject_name, pair.session_name,
                                save_path=str(session_dir / f'{pair.session_name}_phase2.png'))

            phase2_dict = {
                'eag_channels': eag_ch_params,
                'apa': apa_params,
                'coactivation': coact_params,
                'ch_quality': ch_quality,
            }

        return grf_event_df, session_df, cross_df, phase2_dict, grf_triggered_df, offset_row

    except Exception as e:
        print(f"  ❌ {pair.subject_name}/{pair.session_name}: {e}")
        return None, None, None, None, None, None


def run_batch(subject_filter: Optional[str] = None, do_phase3: bool = False,
              reprocess_manual: bool = False):
    """전체 세션 batch 분석 → CSV 저장."""
    pairs = find_all_pairs('data')

    if subject_filter:
        pairs = [p for p in pairs if subject_filter in p.subject_name]

    if reprocess_manual:
        from offset_manager import load_manual_offsets
        manual = load_manual_offsets()
        pairs = [p for p in pairs
                 if p.subject_name in manual
                 and p.session_name in manual[p.subject_name]]
        if not pairs:
            print("Manual offset이 설정된 세션이 없습니다.")
            return

    phases = "Phase 1+2" + (" +3" if do_phase3 else "")
    print(f"=== {phases} Parameter Extraction ===")
    print(f"Total sessions: {len(pairs)}")
    if subject_filter:
        print(f"Filter: {subject_filter}")
    print()

    all_grf_events = []
    all_sessions = []
    all_cross = []
    all_coh_rows = []
    all_ersp_rows = []
    all_eag_ch_rows = []
    all_apa_rows = []
    all_coact_rows = []
    all_grf_triggered = []
    all_offset_rows = []
    success = 0
    fail = 0

    for i, pair in enumerate(pairs):
        tag = f"[{i+1}/{len(pairs)}]"
        grf_df, sess_df, cross_df, phase2_dict, gt_df, offset_row = process_single_session(
            pair, Path('result'), do_phase2=True)

        if offset_row is not None:
            all_offset_rows.append(offset_row)
        if gt_df is not None and len(gt_df):
            all_grf_triggered.append(gt_df)

        if grf_df is not None:
            all_grf_events.append(grf_df)
            all_sessions.append(sess_df)
            all_cross.append(cross_df)
            n_ev = len(grf_df)

            # Phase 2 결과 수집
            if phase2_dict:
                ch_quality = phase2_dict.get('ch_quality', {})
                for cp in phase2_dict['eag_channels']:
                    row = asdict(cp)
                    row['subject'] = pair.subject_name
                    row['session'] = pair.session_name
                    cq = ch_quality.get(cp.channel, {})
                    row['channel_quality'] = cq.get('flag', 'PASS')
                    row['channel_snr_db'] = cq.get('snr_db', float('nan'))
                    all_eag_ch_rows.append(row)
                for ap in phase2_dict['apa']:
                    row = asdict(ap)
                    row['subject'] = pair.subject_name
                    row['session'] = pair.session_name
                    cq = ch_quality.get(ap.channel, {})
                    row['channel_quality'] = cq.get('flag', 'PASS')
                    all_apa_rows.append(row)
                for ca in phase2_dict['coactivation']:
                    row = asdict(ca)
                    row['subject'] = pair.subject_name
                    row['session'] = pair.session_name
                    all_coact_rows.append(row)

            print(f"  {tag} ✅ {pair.subject_name}/{pair.session_name}: "
                  f"{n_ev} events, "
                  f"sym={sess_df['symmetry_index'].iloc[0]:.3f}", end="")

            # Phase 3
            if do_phase3:
                try:
                    sa = SyncAnalyzer(pair)
                    sa.run_analysis()

                    if sa.grf_events:
                        coh_results = compute_eag_grf_coherence(sa)
                        ersp_results = compute_ersp(sa)

                        # Phase 3 시각화
                        session_dir = Path('result') / pair.subject_name
                        session_dir.mkdir(parents=True, exist_ok=True)
                        if coh_results:
                            plot_coherence(coh_results, pair.subject_name, pair.session_name,
                                           save_path=str(session_dir / f'{pair.session_name}_coherence.png'))
                        if ersp_results:
                            plot_ersp(ersp_results, pair.subject_name, pair.session_name,
                                      save_path=str(session_dir / f'{pair.session_name}_ersp.png'))

                        # Phase 3 CSV rows
                        for cr in coh_results:
                            all_coh_rows.append({
                                'subject': pair.subject_name,
                                'session': pair.session_name,
                                'channel': cr.channel + 1,
                                'mean_coh_0_2hz': cr.mean_coh_0_2hz,
                                'mean_coh_2_5hz': cr.mean_coh_2_5hz,
                                'peak_coh_freq': cr.peak_coh_freq,
                                'peak_coh_value': cr.peak_coh_value,
                            })
                        for er in ersp_results:
                            all_ersp_rows.append({
                                'subject': pair.subject_name,
                                'session': pair.session_name,
                                'channel': er.channel + 1,
                                'mean_ersp_0_2hz_db': er.mean_ersp_0_2hz,
                                'mean_ersp_2_5hz_db': er.mean_ersp_2_5hz,
                            })
                        print(f"  [P3: coh={len(coh_results)}ch, ersp={len(ersp_results)}ch]")
                    else:
                        print(f"  [P3: skip, no events]")
                except Exception as e:
                    print(f"  [P3 err: {e}]")
            else:
                print()

            success += 1
        else:
            fail += 1

    print(f"\n=== 완료: {success} 성공, {fail} 실패 ===")

    if not all_sessions:
        print("추출된 데이터 없음")
        return

    # Concatenate
    df_events = pd.concat(all_grf_events, ignore_index=True)
    df_sessions = pd.concat(all_sessions, ignore_index=True)
    df_cross = pd.concat(all_cross, ignore_index=True)

    # Save
    output_dir = Path('result') / 'phase1_params'
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{subject_filter}" if subject_filter else ""

    path_events = output_dir / f'grf_event_params{suffix}.csv'
    path_sessions = output_dir / f'session_params{suffix}.csv'
    path_cross = output_dir / f'eag_grf_cross_params{suffix}.csv'

    df_events.to_csv(path_events, index=False, float_format='%.4f')
    df_sessions.to_csv(path_sessions, index=False, float_format='%.4f')
    df_cross.to_csv(path_cross, index=False, float_format='%.4f')

    print(f"\n저장 완료 (Phase 1):")
    print(f"  GRF 이벤트: {path_events} ({len(df_events)} rows)")
    print(f"  세션 요약:  {path_sessions} ({len(df_sessions)} rows)")
    print(f"  EAG-GRF:   {path_cross} ({len(df_cross)} rows)")

    # GRF-triggered 파라미터 + offset 진단 저장
    if all_offset_rows:
        df_offset = pd.DataFrame(all_offset_rows)
        path_offset = output_dir / f'offset_diagnostics{suffix}.csv'
        df_offset.to_csv(path_offset, index=False, float_format='%.4f')
        n_corr = int(df_offset['method'].astype(str).str.startswith('xcorr').sum())
        print(f"  offset 진단: {path_offset} ({len(df_offset)} rows, {n_corr} 교정)")
    if all_grf_triggered:
        df_gt = pd.concat(all_grf_triggered, ignore_index=True)
        path_gt = output_dir / f'grf_triggered_params{suffix}.csv'
        df_gt.to_csv(path_gt, index=False, float_format='%.4f')
        n_match = int(df_gt['matched'].sum())
        print(f"  GRF-triggered: {path_gt} ({len(df_gt)} rows, {n_match} matched)")

    # Phase 2 CSV 저장
    if all_eag_ch_rows:
        phase2_dir = Path('result') / 'phase2_params'
        phase2_dir.mkdir(parents=True, exist_ok=True)

        df_eag_ch = pd.DataFrame(all_eag_ch_rows)
        df_apa = pd.DataFrame(all_apa_rows)
        df_coact = pd.DataFrame(all_coact_rows)

        path_eag_ch = phase2_dir / f'eag_channel_params{suffix}.csv'
        path_apa = phase2_dir / f'apa_params{suffix}.csv'
        path_coact = phase2_dir / f'coactivation_params{suffix}.csv'

        df_eag_ch.to_csv(path_eag_ch, index=False, float_format='%.4f')
        df_apa.to_csv(path_apa, index=False, float_format='%.4f')
        df_coact.to_csv(path_coact, index=False, float_format='%.4f')

        print(f"\n저장 완료 (Phase 2):")
        print(f"  EAG 채널:    {path_eag_ch} ({len(df_eag_ch)} rows)")
        print(f"  APA:         {path_apa} ({len(df_apa)} rows)")
        print(f"  Co-act:      {path_coact} ({len(df_coact)} rows)")

    # Phase 3 CSV 저장
    if do_phase3 and all_coh_rows:
        phase3_dir = Path('result') / 'phase3_params'
        phase3_dir.mkdir(parents=True, exist_ok=True)

        df_coh = pd.DataFrame(all_coh_rows)
        df_ersp = pd.DataFrame(all_ersp_rows)

        path_coh = phase3_dir / f'coherence{suffix}.csv'
        path_ersp = phase3_dir / f'ersp{suffix}.csv'

        df_coh.to_csv(path_coh, index=False, float_format='%.4f')
        df_ersp.to_csv(path_ersp, index=False, float_format='%.4f')

        print(f"  Coherence: {path_coh} ({len(df_coh)} rows)")
        print(f"  ERSP:      {path_ersp} ({len(df_ersp)} rows)")

    # Summary statistics
    print(f"\n=== Session-Level Summary ===")
    print(f"  symmetry_index:  mean={df_sessions['symmetry_index'].mean():.4f}, "
          f"std={df_sessions['symmetry_index'].std():.4f}")
    print(f"  sway_variability: mean={df_sessions['sway_variability'].mean():.4f}")
    print(f"  event_frequency: mean={df_sessions['event_frequency'].mean():.1f}/min")
    print(f"  mean_time_to_peak: {df_sessions['mean_time_to_peak'].mean():.3f}s")
    rec = df_sessions['mean_recovery_time'].dropna()
    if len(rec) > 0:
        print(f"  mean_recovery_time: {rec.mean():.3f}s")

    print(f"\n=== EAG-GRF Cross Summary (Phase 1) ===")
    fsi_lat = df_cross['fsi_latency'].dropna()
    mi_lat = df_cross['mi_latency'].dropna()
    pk_lat = df_cross['peak_latency'].dropna()
    print(f"  FSI latency:  mean={fsi_lat.mean():.3f}s, std={fsi_lat.std():.3f}s (n={len(fsi_lat)})")
    print(f"  MI latency:   mean={mi_lat.mean():.3f}s, std={mi_lat.std():.3f}s")
    print(f"  Peak latency: mean={pk_lat.mean():.3f}s, std={pk_lat.std():.3f}s")

    # Phase 2 Summary
    if all_eag_ch_rows:
        df_eag_ch = pd.DataFrame(all_eag_ch_rows)
        df_apa = pd.DataFrame(all_apa_rows)
        df_coact = pd.DataFrame(all_coact_rows)
        print(f"\n=== Phase 2: EAG Channel Summary ===")
        for ch in range(EEG_CHANNELS):
            ch_data = df_eag_ch[df_eag_ch['channel'] == ch]
            print(f"  Ch{ch+1}: RMS event={ch_data['rms_event'].mean():.1f}µV, "
                  f"dominance={ch_data['channel_dominance'].mean():.3f}, "
                  f"infl_rate={ch_data['inflection_rate'].mean():.1f}/s")

        print(f"\n=== Phase 2: APA Summary ===")
        apa_rate = df_apa['has_apa'].mean() * 100
        apa_lat = df_apa.loc[df_apa['has_apa'], 'apa_latency'].dropna()
        react_lat = df_apa['reactive_latency'].dropna()
        print(f"  APA detection rate: {apa_rate:.1f}%")
        if len(apa_lat) > 0:
            print(f"  APA latency:  mean={apa_lat.mean():.3f}s (before onset)")
        if len(react_lat) > 0:
            print(f"  Reactive lat: mean={react_lat.mean():.3f}s (after onset)")

        print(f"\n=== Phase 2: Co-activation Summary ===")
        cai = df_coact['co_activation_index']
        print(f"  Co-activation index: mean={cai.mean():.3f}, std={cai.std():.3f}")
        print(f"  Mean active channels: {df_coact['n_active_channels'].mean():.1f}/{EEG_CHANNELS}")

    # Phase 3 Summary
    if do_phase3 and all_coh_rows:
        df_coh = pd.DataFrame(all_coh_rows)
        print(f"\n=== Phase 3: Coherence Summary ===")
        print(f"  0-2Hz mean coherence: {df_coh['mean_coh_0_2hz'].mean():.4f}")
        print(f"  2-5Hz mean coherence: {df_coh['mean_coh_2_5hz'].mean():.4f}")
        print(f"  Peak coherence freq:  {df_coh['peak_coh_freq'].mean():.2f} Hz")

    return df_events, df_sessions, df_cross


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description='Phase 1+2+3 Parameter Extractor')
    parser.add_argument('--batch', action='store_true', help='전체 세션 batch 분석')
    parser.add_argument('--subject', type=str, default=None, help='피험자 필터 (부분 일치)')
    parser.add_argument('--session', type=str, default=None, help='세션 필터 (예: s1)')
    parser.add_argument('--phase3', action='store_true', help='Phase 3 (coherence, ERSP) 포함')
    parser.add_argument('--reprocess-manual', action='store_true',
                        help='Manual offset이 설정된 세션만 재처리')
    args = parser.parse_args()

    if args.batch or args.subject:
        run_batch(subject_filter=args.subject, do_phase3=args.phase3,
                  reprocess_manual=args.reprocess_manual)
    else:
        print("사용법:")
        print("  python3 parameter_extractor.py --batch              # Phase 1만")
        print("  python3 parameter_extractor.py --batch --phase3     # Phase 1 + 3")
        print("  python3 parameter_extractor.py --subject 주창민 --phase3")


if __name__ == '__main__':
    main()
