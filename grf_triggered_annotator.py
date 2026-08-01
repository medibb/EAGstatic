"""
GRF-triggered EAG annotation — GRF 체중이동 전이를 trigger로 EAG knee 반응을 검출

핵심 아이디어:
  GRF(force plate 좌우 imbalance)가 원인, EAG(무릎 전위)가 결과.
  → GRF 전이 시점을 anchor로 삼아 그 직후 윈도우에서만 EAG knee를 찾는다.
  → drift/artifact에서 생기는 가짜 edge를 원천 차단하고, latency(반응 지연)를 얻는다.

오프셋 재추정 (기존 초기-이벤트 매칭보다 강건):
  GRF 전이 열(train)과 EAG edge 열을 전체적으로 cross-correlation → residual offset.
  이벤트 1개가 아니라 모든 전이를 사용하므로 초기 이벤트가 애매한 세션에서도 안정적.

기존 sync_analyzer.SyncAnalyzer(로딩+시간축+auto offset)를 재사용한다.

사용법:
  python3 grf_triggered_annotator.py --session <session_dir> --channels 1-8
  python3 grf_triggered_annotator.py --dir data --channels 1        # batch
  python3 grf_triggered_annotator.py --session <dir> --no-recompute-offset
"""

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import detrend

from sync_analyzer import SyncAnalyzer, find_all_pairs, find_session_pair, SAMPLE_RATE
from eag_analyzer import EEG_CHANNELS, setup_korean_font
from edge_annotator import detect_edges as _detect_edges_signal
import edge_store

setup_korean_font()  # 플롯 한글 라벨 (offset_review/edge_editor/parameter_extractor에 전파)


# ==================== 데이터 구조 ====================

@dataclass
class GRFTransition:
    trans_id: int
    time: float            # 전이 시각 (unified frame, s)
    direction: str         # 'L->R' | 'R->L'
    from_level: float      # 전이 전 signed imbalance
    to_level: float        # 전이 후 signed imbalance


@dataclass
class LoadCycle:
    """체중부하 1회 = 휴식 자세에서 검사측으로 체중을 옮겼다가 되돌아오는 구간.

    프로토콜: 초기 발구름(offset 세팅) → 한발서기 4회(부하를 점진적으로 늘림).
    한 cycle은 EAG에서 knee-pair 2개(부하 시작 시 1개, 이탈 시 1개)를 만든다
    → 세션당 4 cycle × 2 = 8 이벤트가 분석 대상이다.
    """
    cycle_id: int
    onset_time: float      # 부하 시작 (rest → load)
    offset_time: float     # 부하 종료 (load → rest)
    rest_level: float      # 휴식 자세 signed imbalance
    load_level: float      # 부하 구간 signed imbalance (중앙값)
    duration: float
    test_side: str = ''    # 부하가 실리는(검사) 다리 'L'|'R' — 휴식 시 비어 있던 쪽
    load_ratio: float = float('nan')   # 부하 구간 평균: 검사측 / 전체 (0~1)
    rest_ratio: float = float('nan')   # 휴식 구간 기준 검사측 비율 (기저)

    @property
    def load_step(self) -> float:
        """dose = load_level - rest_level (부호 포함). |값|이 클수록 체중부하가 큼."""
        return self.load_level - self.rest_level

    @property
    def load_pct(self) -> float:
        """검사측 체중부하율 (%). 실험설계 20-50-80-100%에 대응하는 실측값."""
        return self.load_ratio * 100.0


@dataclass
class TriggeredResponse:
    """한 GRF 전이(rise/fall)에 대응하는 EAG 변화. onset+offset 두 knee로 변화 크기 측정.

    한 번의 체중부하 cycle = GRF 상승 전이 + 하강 전이 → 각 전이마다 이 레코드 1개
    (= knee 2개) → cycle당 4 knee. amplitude가 그 변화의 크기(µV).
    """
    trans_id: int
    trans_time: float      # GRF 전이(onset) 시각 (s)
    grf_direction: str     # 'R->L' | 'L->R' (weight shift 방향)
    grf_from_level: float  # 전이 전 signed imbalance plateau (부하 시작 단계)
    grf_to_level: float    # 전이 후 signed imbalance plateau (부하 도달 단계)
    grf_step: float        # to_level - from_level (부하 단계 크기 = dose, 부호 포함)
    channel: int
    matched: bool          # 윈도우 내 유효 EAG edge 검출 여부
    onset_time: float      # EAG knee 1: 변화 시작 (직전 plateau 끝) (s)
    onset_amp: float       # knee 1 진폭 (µV, detrended)
    offset_time: float     # EAG knee 2: 변화 끝 (다음 plateau 시작) (s)
    offset_amp: float      # knee 2 진폭 (µV)
    amplitude: float       # offset_amp - onset_amp (변화 크기, µV, 부호 포함)
    transition_time: float # offset_time - onset_time (s)
    slope: float           # amplitude / transition_time (µV/s)
    latency: float         # onset_time - trans_time (s), 동시성이면 ≈0
    eag_direction: str     # 'rise' | 'fall'


@dataclass
class OffsetResult:
    auto_offset: float          # SyncAnalyzer 자동 offset
    residual: float             # edge-train 정렬로 얻은 보정량
    corrected_offset: float     # auto + residual
    n_grf_trans: int
    match_rate_auto: float      # 보정 전 매칭율
    match_rate_corrected: float # 보정 후 매칭율
    latency_mad_auto: float     # 보정 전 latency 분산(MAD)
    latency_mad_corrected: float
    method: str                 # 'auto-ok' | 'xcorr-corrected' | 'matchprofile' | 'auto (...)'
    needs_review: bool = False  # 수동 검토 권장 (저match/애매/검색한계)
    review_reason: str = ''     # 검토 필요 사유
    profile_margin: float = float('nan')  # best match와 2nd peak 차이 (신뢰도)


# ==================== GRF 전이 검출 ====================

def signed_imbalance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right) / np.maximum(left + right, 0.1)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    return np.convolve(x, np.ones(win) / win, mode='same')


def detect_grf_transitions(time: np.ndarray,
                           signed: np.ndarray,
                           fs: int = SAMPLE_RATE,
                           min_amp: float = 0.3,
                           slope_k: float = 3.0,
                           min_gap_sec: float = 0.30) -> List[GRFTransition]:
    """signed imbalance 사각파의 레벨 전이(rise/fall corner)를 검출한다.

    EAG에 튜닝된 edge_annotator.detect_edges를 signed imbalance에 그대로 적용한다.
    기존 gradient-percentile 방식이 준주기 구간에서 전이를 놓치던 문제를 해결
    (예: s7에서 8→10개). GRFTransition.time = onset knee (전이 시작 = 인과 트리거).
    min_amp는 signed imbalance 스케일(≈±1)에 맞춰 0.3.
    """
    edges = _detect_edges_signal(time, signed, fs=fs,
                                 slope_k=slope_k, min_amp=min_amp,
                                 min_gap_sec=min_gap_sec)
    trans: List[GRFTransition] = []
    for i, e in enumerate(edges):
        trans.append(GRFTransition(
            trans_id=i,
            time=float(e.onset_time),
            direction='R->L' if e.amplitude > 0 else 'L->R',
            from_level=float(e.plateau_before),
            to_level=float(e.plateau_after),
        ))
    return trans


# ==================== 프로토콜(체중부하 cycle) 검출 ====================

EXPECTED_CYCLES = 4          # 세션당 한발서기 횟수
EXPECTED_EVENTS = 8          # = EXPECTED_CYCLES × 2 (부하 시작/종료)


def find_plateaus(time: np.ndarray, signed: np.ndarray,
                  tol: float = 0.08, min_dur: float = 1.0) -> List[tuple]:
    """signed가 tol 안에서 min_dur 이상 유지되는 구간 = plateau.

    Returns: [(i0, i1, level, duration), ...]
    """
    out = []
    n = len(signed)
    i = 0
    while i < n:
        j = i + 1
        lo = hi = signed[i]
        while j < n:
            lo2, hi2 = min(lo, signed[j]), max(hi, signed[j])
            if hi2 - lo2 > 2 * tol:
                break
            lo, hi = lo2, hi2
            j += 1
        if time[j - 1] - time[i] >= min_dur:
            out.append([i, j - 1, float(np.median(signed[i:j])),
                        float(time[j - 1] - time[i])])
            i = j
        else:
            i += 1
    # 완만한 드리프트로 한 plateau가 여러 조각으로 나뉘면 개수를 왜곡하므로,
    # 시간상 맞닿아 있고 레벨이 비슷한 조각은 하나로 합친다.
    merged = []
    for p in out:
        if merged and p[0] - merged[-1][1] <= 2 and abs(p[2] - merged[-1][2]) <= tol:
            q = merged[-1]
            q[1] = p[1]
            q[2] = float(np.median(signed[q[0]:q[1] + 1]))
            q[3] = float(time[q[1]] - time[q[0]])
        else:
            merged.append(p)
    return [tuple(p) for p in merged]


def rest_posture_level(time: np.ndarray, signed: np.ndarray,
                       cluster_tol: float = 0.15) -> float:
    """휴식 자세의 signed imbalance.

    프로토콜상 휴식 자세는 부하 4회 사이사이에 **반복해서 돌아오는** 레벨이라
    plateau가 5개(부하 전/사이/후) 생기는 반면, 각 부하 단계 레벨은 1번씩만 나온다.
    따라서 plateau를 레벨로 군집화해 **구성원이 가장 많은 군집**을 휴식으로 본다
    (동수면 총 지속시간이 긴 쪽).

    단순 최빈값(히스토그램)은 프로토콜 전후의 양발 서기가 길거나 부하 단계가
    한 bin에 몰릴 때 엉뚱한 레벨을 고르는 문제가 있어 이 방식을 쓴다.
    """
    pl = find_plateaus(time, signed)
    if not pl:
        return float(np.median(signed))
    order = sorted(pl, key=lambda p: p[2])
    groups, cur = [], [order[0]]
    for p in order[1:]:
        if p[2] - cur[-1][2] <= cluster_tol:
            cur.append(p)
        else:
            groups.append(cur); cur = [p]
    groups.append(cur)
    best = max(groups, key=lambda g: (len(g), sum(p[3] for p in g)))
    return float(np.median([p[2] for p in best]))


def detect_load_cycles(time: np.ndarray, signed: np.ndarray,
                       enter: float = 0.10, leave: float = 0.06,
                       min_dur: float = 2.0, rest_need: float = 0.8,
                       plateau_tol: float = 0.10,
                       left: Optional[np.ndarray] = None,
                       right: Optional[np.ndarray] = None
                       ) -> Tuple[float, List[LoadCycle]]:
    """휴식 자세에서 벗어났다 되돌아오는 구간 = 체중부하 cycle.

    detect_grf_transitions는 사각파 edge를 min_amp(=0.3)로 잡기 때문에, 가장 가벼운
    1단계(예: signed 0.99→0.82, 진폭 0.17)를 놓쳐 4회 중 3회만 검출되는 세션이 많다.
    여기서는 문턱을 진폭이 아니라 "휴식 레벨로부터의 이탈"로 잡아 가벼운 단계도 포착한다.

    프로토콜 전후의 양발 서기는 앞이나 뒤가 휴식 자세로 둘러싸이지 않으므로 제외된다.

    Returns: (rest_level, cycles)
    """
    rest = rest_posture_level(time, signed)
    dev = np.abs(signed - rest)
    idx = np.flatnonzero(dev > enter)
    if len(idx) == 0:
        return rest, []

    runs = []                                  # enter를 넘은 연속 구간
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > 1:
            runs.append((s, p)); s = i
        p = i
    runs.append((s, p))

    expanded = []                              # 히스테리시스: leave 아래로 내려갈 때까지 확장
    for a, b in runs:
        while a > 0 and dev[a - 1] > leave: a -= 1
        while b < len(dev) - 1 and dev[b + 1] > leave: b += 1
        expanded.append((a, b))
    merged = []
    for a, b in expanded:
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))

    def rest_span(i0, step, ref, search=4.0):
        """ref에서 step 방향 search초 안에 rest_need 이상 '이어지는' 휴식 구간이 있는가.

        경계 직후 신호가 휴식 레벨로 서서히 안정되므로(예: 0.89→0.98에 1초 소요),
        '경계 직후부터 연속' 조건은 가벼운 단계를 탈락시킨다. 창 안에서 찾는다.
        """
        i, run0 = i0, None
        while 0 <= i < len(time) and abs(time[i] - ref) <= search:
            if dev[i] <= leave:
                if run0 is None:
                    run0 = time[i]
                elif abs(time[i] - run0) >= rest_need:
                    return True
            else:
                run0 = None
            i += step
        return False

    cycles: List[LoadCycle] = []
    for a, b in merged:
        if time[b] - time[a] < min_dur:                       # 순간적 흔들림
            continue
        if not (rest_span(a - 1, -1, time[a]) and
                rest_span(b + 1, +1, time[b])):               # 프로토콜 전후 구간
            continue
        # 이벤트 시각은 "움직임이 시작되는 순간"이어야 EAG knee와 맞는다.
        #   부하 시작 = 휴식 plateau를 벗어나는 순간(=a)
        #   부하 종료 = 부하 plateau를 벗어나는 순간(=plateau 끝). b는 복귀 램프의 '끝'이라
        #              1초 가까이 늦어 EAG rise와 어긋난다.
        load = float(np.median(signed[a:b + 1]))
        near = np.flatnonzero(np.abs(signed[a:b + 1] - load) <= plateau_tol)
        if len(near):
            p0, p1 = a + int(near[0]), a + int(near[-1])
            load = float(np.median(signed[p0:p1 + 1]))
        else:
            p1 = b
        # 검사측 = 휴식 시 비어 있던 다리 (signed>0이면 왼쪽에 실은 상태 → 검사측은 오른쪽)
        side = 'R' if rest > 0 else 'L'
        ratio = rest_r = float('nan')
        if left is not None and right is not None:
            tot = np.asarray(left, float) + np.asarray(right, float)
            test = np.asarray(right if side == 'R' else left, float)
            ok = tot > 1e-6
            seg = slice(a, p1 + 1)
            m = ok[seg]
            if m.any():
                ratio = float(np.mean(test[seg][m] / tot[seg][m]))
            rm = ok & (np.abs(signed - rest) <= leave)      # 휴식 구간 전체
            if rm.any():
                rest_r = float(np.mean(test[rm] / tot[rm]))
        cycles.append(LoadCycle(
            cycle_id=len(cycles),
            onset_time=float(time[a]), offset_time=float(time[p1]),
            rest_level=rest, load_level=load,
            duration=float(time[p1] - time[a]),
            test_side=side, load_ratio=ratio, rest_ratio=rest_r))
    return rest, cycles


def detect_load_cycles_expected(time: np.ndarray, signed: np.ndarray,
                                left: Optional[np.ndarray] = None,
                                right: Optional[np.ndarray] = None,
                                expected: int = EXPECTED_CYCLES
                                ) -> Tuple[float, List[LoadCycle], dict]:
    """cycle이 정확히 expected개가 되도록 문턱을 탐색한다.

    프로토콜상 모든 세션이 4회를 시행했으므로, 3회로 잡히는 것은 가장 가벼운 단계가
    노이즈/문턱에 묻힌 것이다. enter 문턱을 낮춰가며(그리고 최소 지속시간을 완화하며)
    4회가 나오는 조합을 찾는다. 못 찾으면 4에 가장 가까운 결과를 쓰고 info로 알린다.

    Returns: (rest, cycles, info) — info={'enter','min_dur','found','searched'}
    """
    best = None
    tried = 0
    for enter in (0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02):
        for min_dur in (2.0, 1.5, 1.0):
            tried += 1
            rest, cyc = detect_load_cycles(time, signed, enter=enter,
                                           leave=min(0.06, enter * 0.6),
                                           min_dur=min_dur, left=left, right=right)
            if len(cyc) == expected:
                return rest, cyc, {'enter': enter, 'min_dur': min_dur,
                                   'found': True, 'searched': tried}
            key = (abs(len(cyc) - expected), -len(cyc))
            if best is None or key < best[0]:
                best = (key, rest, cyc, enter, min_dur)
    _, rest, cyc, enter, min_dur = best
    return rest, cyc, {'enter': enter, 'min_dur': min_dur,
                       'found': False, 'searched': tried}


def cycles_to_transitions(cycles: List[LoadCycle],
                          raw_trans: Optional[List[GRFTransition]] = None,
                          snap: float = 1.2) -> List[GRFTransition]:
    """cycle → 분석 anchor 전이 8개 (cycle당 부하 시작/종료 2개).

    extract_responses가 GRFTransition 목록을 받으므로, 프로토콜 이벤트만 anchor로
    쓰면 발구름·전후 양발 서기 구간이 파라미터에 섞이지 않는다.
    grf_step(dose)은 rest↔load 레벨 차이로 계산된다.

    detect_grf_transitions의 knee 시각이 cycle 경계(문턱 교차)보다 정확하므로,
    raw_trans가 주어지면 ±snap초 내 가장 가까운 전이로 시각을 보정한다.
    (raw는 가벼운 단계를 '놓치는' 것이 문제일 뿐, 잡은 것의 시각은 정확하다.)
    """
    rt = np.array([t.time for t in raw_trans]) if raw_trans else np.array([])

    def snapped(t):
        if len(rt):
            k = int(np.argmin(np.abs(rt - t)))
            if abs(rt[k] - t) <= snap:
                return float(rt[k])
        return float(t)

    trans: List[GRFTransition] = []
    for c in cycles:
        for t, frm, to in ((c.onset_time, c.rest_level, c.load_level),
                           (c.offset_time, c.load_level, c.rest_level)):
            trans.append(GRFTransition(
                trans_id=len(trans), time=snapped(t),
                direction='R->L' if to > frm else 'L->R',
                from_level=float(frm), to_level=float(to)))
    return trans


def detect_edge_near(te: np.ndarray, sig: np.ndarray, t0: float, fs: int,
                     lat_lo: float = -0.5, lat_hi: float = 1.5,
                     frac: float = 0.30, min_amp: float = 8.0,
                     smooth_sec: float = 0.20, max_dur: float = 2.5):
    """anchor 시각 t0 부근에서 EAG knee-pair를 국소 검출.

    detect_eag_edges는 전역 문턱(min_amp=25µV, slope_k)을 쓰기 때문에, 가장 가벼운
    부하 단계처럼 반응이 작은 이벤트를 통째로 놓친다. 여기서는 "GRF 이벤트가 그 시각에
    분명히 있다"는 사전정보를 이용해 창 안에서 가장 가파른 지점을 찾고, 기울기가
    정점의 frac 이하로 떨어지는 곳까지 좌우로 넓혀 knee 두 개를 잡는다.

    Returns: (onset_t, onset_a, amp, offset_t, offset_a) 또는 None
    """
    i0 = int(np.searchsorted(te, t0 + lat_lo))
    i1 = int(np.searchsorted(te, t0 + lat_hi))
    if i1 - i0 < 5:
        return None
    sm = _smooth(sig, max(1, int(smooth_sec * fs)))
    slope = np.gradient(sm) * fs
    k = i0 + int(np.argmax(np.abs(slope[i0:i1])))
    peak = abs(slope[k])
    if peak <= 0:
        return None
    thr = frac * peak
    lim = int(max_dur * fs)
    a = k
    while a > 0 and abs(slope[a - 1]) > thr and k - a < lim:
        a -= 1
    b = k
    while b < len(slope) - 1 and abs(slope[b + 1]) > thr and b - k < lim:
        b += 1
    amp = float(sm[b] - sm[a])
    if abs(amp) < min_amp:
        return None
    return (float(te[a]), float(sm[a]), amp, float(te[b]), float(sm[b]))


def detect_eag_edges_protocol(te: np.ndarray, sig: np.ndarray,
                              anchors: List[GRFTransition], fs: int,
                              lat_lo: float = -0.5, lat_hi: float = 1.5,
                              **kw) -> List[Tuple[float, float, float, float, float]]:
    """전역 검출 + anchor별 국소 보강.

    전역 detect_eag_edges 결과를 기준으로 하되, 매칭되는 edge가 없는 anchor에 대해서만
    detect_edge_near로 약한 반응을 회수한다. 프로토콜 이벤트(8개)를 최대한 채우면서
    전역 검출이 이미 잘 잡은 곳은 건드리지 않는다.
    """
    edges = list(detect_eag_edges(te, sig, fs=fs, **kw))
    for tr in anchors:
        e_on = np.array([e[0] for e in edges]) if edges else np.array([])
        if len(e_on):
            sel = np.where((e_on - tr.time >= lat_lo) & (e_on - tr.time <= lat_hi))[0]
            if len(sel):
                continue                       # 이미 매칭됨
        got = detect_edge_near(te, sig, tr.time, fs, lat_lo, lat_hi)
        if got is not None:
            edges.append(got)
    edges.sort(key=lambda e: e[0])
    return edges


def validate_cycle_edges(cycles: List[LoadCycle], edge_on: np.ndarray,
                         edge_amp: np.ndarray,
                         raw_trans: Optional[List[GRFTransition]] = None,
                         lat_lo: float = -0.5, lat_hi: float = 1.5) -> dict:
    """세션·채널의 EAG edge가 프로토콜(4 cycle × 2 = 8 이벤트)을 만족하는지 검사.

    각 cycle의 부하 시작/종료 anchor마다 [lat_lo, lat_hi] 안에 edge가 하나씩 있어야 하고,
    부하 시작 쪽과 종료 쪽의 방향(rise/fall)이 서로 반대여야 한다(= fall-rise 교대).

    Returns: {'ok', 'n_cycles', 'n_matched', 'n_edges', 'reasons', 'events'}
    """
    reasons = []
    if len(cycles) != EXPECTED_CYCLES:
        reasons.append(f'cycle {len(cycles)}회(기대 {EXPECTED_CYCLES})')

    anchors = cycles_to_transitions(cycles, raw_trans)
    events, dirs = [], []
    for i, tr in enumerate(anchors):
        kind = 'on' if i % 2 == 0 else 'off'
        t = tr.time
        k = -1
        if len(edge_on):
            sel = np.where((edge_on - t >= lat_lo) & (edge_on - t <= lat_hi))[0]
            if len(sel):
                k = int(sel[np.argmin(np.abs(edge_on[sel] - t))])
        d = '' if k < 0 else ('rise' if edge_amp[k] > 0 else 'fall')
        events.append({'cycle': i // 2, 'kind': kind, 'grf_time': float(t),
                       'edge_idx': k, 'direction': d,
                       'edge_time': float(edge_on[k]) if k >= 0 else None})
        dirs.append((kind, d))

    n_match = sum(1 for e in events if e['edge_idx'] >= 0)

    # cycle별 요약: 같은 부하의 rise/fall은 크기가 거의 같아야 하므로, 한쪽만 살아 있어도
    # 그 부하에서의 EAG 크기를 추정할 수 있다 → cycle은 "이벤트 ≥1개"면 측정 가능으로 본다.
    per_cycle = []
    for c in cycles:
        ev = [e for e in events if e['cycle'] == c.cycle_id]
        on_e = next((e for e in ev if e['kind'] == 'on' and e['edge_idx'] >= 0), None)
        off_e = next((e for e in ev if e['kind'] == 'off' and e['edge_idx'] >= 0), None)
        a_on = abs(float(edge_amp[on_e['edge_idx']])) if on_e else float('nan')
        a_off = abs(float(edge_amp[off_e['edge_idx']])) if off_e else float('nan')
        both = on_e is not None and off_e is not None
        amps = [a for a in (a_on, a_off) if np.isfinite(a)]
        asym = (abs(a_on - a_off) / max(1e-9, (a_on + a_off) / 2)) if both else float('nan')
        opposite = (both and on_e['direction'] != off_e['direction'])
        per_cycle.append({
            'cycle': c.cycle_id, 'n_events': len(amps),
            'amp_on': None if not np.isfinite(a_on) else round(a_on, 1),
            'amp_off': None if not np.isfinite(a_off) else round(a_off, 1),
            'amp': None if not amps else round(float(np.mean(amps)), 1),
            'asymmetry': None if not np.isfinite(asym) else round(asym, 3),
            'opposite_dir': bool(opposite) if both else None,
            'load_pct': None if not np.isfinite(c.load_ratio) else round(c.load_pct, 1),
        })

    n_measured = sum(1 for p in per_cycle if p['n_events'] >= 1)
    if cycles and n_measured < len(cycles):
        miss = [f"c{p['cycle']+1}" for p in per_cycle if p['n_events'] == 0]
        reasons.append(f'측정 불가 cycle {len(cycles)-n_measured}개({",".join(miss)})')

    bad_dir = [f"c{p['cycle']+1}" for p in per_cycle if p['opposite_dir'] is False]
    if bad_dir:
        reasons.append(f'부하/이탈 방향 같음({",".join(bad_dir)})')

    dup = len(set(e['edge_idx'] for e in events if e['edge_idx'] >= 0))
    if dup < n_match:
        reasons.append('한 edge가 두 이벤트에 중복 매칭')

    # anchor 근처가 아닌 edge = 노이즈 후보 (부하 구간 한가운데·휴식 구간에서 검출된 것)
    matched_idx = {e['edge_idx'] for e in events if e['edge_idx'] >= 0}
    noise = [i for i in range(len(edge_on)) if i not in matched_idx]

    return {'ok': not reasons, 'n_cycles': len(cycles), 'n_matched': n_match,
            'n_events': len(events), 'n_edges': int(len(edge_on)),
            'n_measured_cycles': n_measured, 'per_cycle': per_cycle,
            'noise_idx': noise,
            'reasons': '; '.join(reasons), 'events': events}


# ==================== EAG edge(knee) 검출 ====================

def detect_eag_edges(time: np.ndarray,
                     sig: np.ndarray,
                     fs: int = SAMPLE_RATE,
                     smooth_sec: float = 0.20,
                     slope_k: float = 2.5,
                     min_gap_sec: float = 0.30,
                     min_amp: float = 25.0) -> List[Tuple[float, float, float, float, float]]:
    """detrended EAG에서 edge의 onset/offset knee 목록 반환.

    Returns: [(onset_time, onset_amp, amplitude, offset_time, offset_amp), ...]
      한 전이의 두 corner point(시작 plateau 끝, 끝 plateau 시작)를 모두 담는다.
    """
    n = len(sig)
    sm = _smooth(sig, max(1, int(smooth_sec * fs)))
    slope = np.gradient(sm) * fs
    med = np.median(slope)
    mad = np.median(np.abs(slope - med)) + 1e-9
    thr = slope_k * 1.4826 * mad
    active = np.abs(slope - med) > thr

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
    gap = int(min_gap_sec * fs)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    pw = max(1, int(0.4 * fs))
    out = []
    for s, e in merged:
        pa, pb = max(0, s - pw), s
        qa, qb = e, min(n, e + pw)
        if pb - pa < 3 or qb - qa < 3:
            continue
        lb = float(np.median(sig[pa:pb]))
        la = float(np.median(sig[qa:qb]))
        amp = la - lb
        if abs(amp) < min_amp:
            continue
        band = abs(amp) * 0.10
        onset = s
        for idx in range(s, e + 1):
            if abs(sig[idx] - lb) > band:
                onset = idx
                break
        offset = e
        for idx in range(e, s - 1, -1):
            if abs(sig[idx] - la) > band:
                offset = idx
                break
        out.append((float(time[onset]), float(sig[onset]), float(amp),
                    float(time[offset]), float(sig[offset])))
    return out


# ==================== 오프셋 재추정 (연속 envelope cross-correlation) ====================

def _deriv_env(time: np.ndarray, sig: np.ndarray, fs: int, smooth_sec: float = 0.15) -> np.ndarray:
    """평활 미분의 절댓값 = 전이/반응 시점에서 peak를 갖는 envelope (z-score)."""
    sm = _smooth(sig, max(1, int(smooth_sec * fs)))
    env = np.abs(np.gradient(sm) * fs)
    return (env - env.mean()) / (env.std() + 1e-9)


def estimate_residual_offset(tg: np.ndarray, signed: np.ndarray,
                             te: np.ndarray, eag: np.ndarray,
                             fs: int,
                             search: float = 3.0,
                             dt: float = 0.01,
                             lat_target: float = 0.15,
                             plausible: Tuple[float, float] = (-0.10, 0.60)
                             ) -> Tuple[float, float, str]:
    """연속 미분-envelope cross-correlation으로 residual offset 추정.

    GRF |d(signed imbalance)/dt| 와 EAG |d(filtered)/dt| 는 각각 전이·반응 시점에서
    peak를 갖는다. 두 envelope의 xcorr peak lag = clock_error + 대표 latency.
    개별 이벤트 검출 누락에 강건 (전체 파형 사용).

    lag > 0: EAG가 GRF보다 lag 만큼 지연 (정상, EAG가 반응).
    residual = lag - lat_target  (정상 latency는 보존, gross 오차만 보정).
    lag가 plausible 범위 내면 auto offset 신뢰 → residual=0.

    Returns: (residual, lag, method)
      residual: EAG 시간축에서 빼줄 값 (양수면 EAG를 당김)
    """
    t0 = max(tg[0], te[0]); t1 = min(tg[-1], te[-1])
    if t1 - t0 < 5:
        return 0.0, float('nan'), 'auto (short overlap)'
    grid = np.arange(t0, t1, dt)
    genv = np.interp(grid, tg, _deriv_env(tg, signed, fs))
    eenv = np.interp(grid, te, _deriv_env(te, eag, fs))

    max_lag = int(search / dt)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([
        np.dot(genv[:len(genv) - l], eenv[l:]) if l >= 0
        else np.dot(genv[-l:], eenv[:len(eenv) + l])
        for l in lags
    ])
    lag = float(lags[int(np.argmax(corr))] * dt)

    if plausible[0] <= lag <= plausible[1]:
        return 0.0, lag, 'auto-ok'
    residual = lag - lat_target
    return float(residual), lag, 'xcorr-corrected'


def estimate_offset_by_edges(eag_on: np.ndarray, grf_on: np.ndarray,
                             search: float = 3.0, dt: float = 0.02,
                             win: float = 0.35
                             ) -> Tuple[float, float, int, int]:
    """EAG edge onset 열을 GRF edge onset 열에 정렬해 residual offset을 구한다.

    가정(깨끗한 GRF edge 기준): GRF 전이와 EAG 반응 edge는 거의 동시(latency≈0).
    δ = EAG 시간축에 더해 GRF와 겹치게 하는 이동량. residual = -δ
    (파이프라인 규약: te_corr = te - residual).
    준주기(≈5초) cycle-skip 방지: ±win 내 매칭 edge 수를 우선 최대화, 동률이면
    |중앙차| 최소. 개별 edge 검출 누락에도 열 전체 매칭이라 강건.

    Returns: (residual, median_resid, n_match, n_grf)
      median_resid = median(GRF - EAG_shift), 0에 가까울수록 동시성 정렬 우수.
    """
    eag_on = np.asarray(eag_on, dtype=float)
    grf_on = np.asarray(grf_on, dtype=float)
    if len(eag_on) == 0 or len(grf_on) == 0:
        return 0.0, float('nan'), 0, len(grf_on)
    cands = np.arange(-search, search + dt, dt)
    best = None  # (key, delta, median_resid, score)
    for d in cands:
        s = eag_on + d
        diffs = np.array([g - s[np.argmin(np.abs(s - g))] for g in grf_on])
        near = diffs[np.abs(diffs) < win]
        score = int(len(near))
        md = float(np.median(np.abs(near))) if len(near) else 9.0
        # 매칭 수 우선 → |중앙차| 작은 것 → 이동량 |δ| 작은 것(cycle-skip 억제, Occam)
        key = (score, -md, -abs(float(d)))
        if best is None or key > best[0]:
            best = (key, float(d),
                    float(np.median(near)) if len(near) else float('nan'), score)
    _, delta, med_resid, score = best
    return -delta, med_resid, score, len(grf_on)


def _match_stats(grf_t: np.ndarray, eag_t: np.ndarray,
                 lat_lo: float = -0.5, lat_hi: float = 1.5) -> Tuple[float, float, list]:
    """각 GRF 전이에 대해 [lat_lo, lat_hi] 윈도우 내 가장 가까운 EAG edge 매칭.

    동시성(latency≈0) 모델이라 하한을 음수(-0.5)로 열어 약간 앞선 반응도 매칭.
    Returns: (match_rate, latency_MAD, latencies)
    """
    lats = []
    for gt in grf_t:
        cand = eag_t[(eag_t - gt >= lat_lo) & (eag_t - gt <= lat_hi)]
        if len(cand):
            lats.append(float(cand[np.argmin(np.abs(cand - gt))] - gt))  # 가장 가까운(동시)
    if not lats:
        return 0.0, float('nan'), []
    lats = np.array(lats)
    mr = len(lats) / len(grf_t)
    mad = float(np.median(np.abs(lats - np.median(lats))))
    return float(mr), mad, list(lats)


def offset_match_profile(grf_t: np.ndarray, eag_edge_t: np.ndarray,
                         search: float = 10.0, dt: float = 0.1,
                         lat_lo: float = -0.2, lat_hi: float = 1.0
                         ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """후보 residual 전 구간의 match-rate 프로파일을 직접 계산.

    envelope xcorr가 준주기 cycle-skip으로 실패하는 극단 case용.
    match-rate를 직접 최적화하므로 큰 offset도 찾을 수 있으나,
    준주기 secondary peak가 있어 신뢰도(margin)를 함께 반환한다.

    Returns: (candidates, match_profile, best_residual, margin)
      margin = best match - (best에서 1초 이상 떨어진 최고 peak) — 클수록 명확.
    """
    if len(grf_t) < 3 or len(eag_edge_t) < 3:
        return np.array([]), np.array([]), 0.0, float('nan')
    cand = np.arange(-search, search + dt, dt)
    mrs = np.empty(len(cand))
    mads = np.empty(len(cand))
    for i, res in enumerate(cand):
        mr, mad, _ = _match_stats(grf_t, eag_edge_t - res, lat_lo, lat_hi)
        mrs[i] = mr
        mads[i] = mad if np.isfinite(mad) else 9.0
    # best: match 최대, 동률이면 latency 분산 최소
    best_i = int(np.lexsort((mads, -mrs))[0])  # -mrs 오름차순→match 큰 게 먼저? 확인 아래
    # lexsort는 마지막 key 우선. mrs 큰 것 우선 위해 -mrs, 동률 mad 작은 것
    order = np.lexsort((mads, -mrs))
    best_i = int(order[0])
    best_res = float(cand[best_i])
    best_mr = mrs[best_i]
    # margin: best에서 ≥1s 떨어진 후보 중 최대 match
    far = np.abs(cand - best_res) >= 1.0
    second = float(mrs[far].max()) if far.any() else 0.0
    margin = float(best_mr - second)
    return cand, mrs, best_res, margin


# ==================== 메인 파이프라인 ====================

def compute_offset(sa: SyncAnalyzer, ref_ch: int = 0,
                   recompute_offset: bool = True
                   ) -> Tuple[OffsetResult, List[GRFTransition], np.ndarray, np.ndarray]:
    """세션의 offset 진단·보정. annotate_session과 offset-report가 공유.

    Returns: (OffsetResult, transitions, signed_imbalance, grf_transition_times)
    """
    auto_offset = float(sa.time_offset)
    tg = sa.unified_time_grf
    signed = signed_imbalance(sa.grf_left, sa.grf_right)
    te = sa.unified_time_eag

    trans = detect_grf_transitions(tg, signed, fs=sa.eag.sample_rate)
    grf_t = np.array([tr.time for tr in trans])

    eag_ref = detrend(sa.eag_filtered[:, ref_ch])
    edges_ref = detect_eag_edges(te, eag_ref, fs=sa.eag.sample_rate)
    eag_edge_t = np.array([e[0] for e in edges_ref]) if edges_ref else np.array([])

    # offset 보정: GRF edge 기준 edge-train 정렬 (동시성 가정, latency≈0).
    # 깨끗한 GRF 전이 열에 EAG edge 열을 맞춰 residual을 구한다. 준주기 cycle-skip은
    # 매칭 edge 수 최대화로 방지. 개별 검출 누락에 강건(열 전체 매칭).
    mr_a, mad_a, _ = _match_stats(grf_t, eag_edge_t)
    residual = 0.0
    method = 'auto (recompute off)'
    n_match = 0
    med_resid = float('nan')
    if recompute_offset:
        residual, med_resid, n_match, n_grf = estimate_offset_by_edges(eag_edge_t, grf_t)
        method = f'grf-edge-align ({n_match}/{n_grf}, med={med_resid:+.02f}s)'
        if n_match < 3:  # edge가 너무 적으면 연속 envelope xcorr fallback
            residual, _lag, m2 = estimate_residual_offset(
                tg, signed, te, eag_ref, sa.eag.sample_rate)
            method = f'fallback-xcorr ({m2})'

    eag_edge_corr = eag_edge_t - residual
    mr_c, mad_c, _ = _match_stats(grf_t, eag_edge_corr)
    # 수용: (a) 매칭/latency 개선 AND (b) 교정 후 |offset|≤2.0(그럴싸한 clock 오차).
    # 큰 교정은 준주기 cycle-skip 또는 기기 시작차 → 자동적용 보류하고 review로만.
    large_review = False
    if residual != 0.0 and recompute_offset:
        improved = (mr_c >= mr_a - 0.05) or \
                   (np.isfinite(mad_c) and np.isfinite(mad_a) and mad_c <= mad_a)
        plausible = abs(auto_offset + residual) <= 2.0
        if not improved:
            residual = 0.0
            method = 'auto (correction rejected)'
            mr_c, mad_c = mr_a, mad_a
        elif not plausible:
            large_review = True
            method = f'large-offset review (edge-align={auto_offset + residual:+.2f})'
            residual = 0.0
            mr_c, mad_c = mr_a, mad_a

    # 극단(저매칭) 세션: match-profile 전역(±10s) 대안 제시. 자동 채택하지 않고
    # review로만 넘김 (준주기 cycle-skip 방지, 사람이 패널 확인 후 확정).
    margin = float('nan'); best_res = float('nan')
    if recompute_offset and mr_c < 0.4:
        _cand, _prof, best_res, margin = offset_match_profile(grf_t, eag_edge_t)

    corrected_offset = auto_offset + residual  # offset에 +residual == EAG축 -residual 이동

    # 수동 검토 플래그
    reasons = []
    if mr_c < 0.4:
        reasons.append(f'저match({mr_c:.0%})')
    if n_match and n_match < 0.5 * len(trans):
        reasons.append(f'edge매칭부족({n_match}/{len(trans)})')
    if abs(corrected_offset) > 2.9:
        reasons.append(f'큰offset({corrected_offset:+.1f}s)')
    if large_review:
        reasons.append('큰교정보류(재검토)')
    if np.isfinite(best_res) and abs(best_res - residual) > 0.5:
        reasons.append(f'대안제시(res={best_res:+.1f}s)')
    needs_review = len(reasons) > 0

    off = OffsetResult(auto_offset, residual, corrected_offset, len(trans),
                       mr_a, mr_c, mad_a, mad_c, method,
                       needs_review=needs_review, review_reason='; '.join(reasons),
                       profile_margin=margin)
    return off, trans, signed, grf_t


def annotate_session(session_dir: str,
                     channels: List[int],
                     recompute_offset: bool = True,
                     plot: bool = True) -> Tuple[List[TriggeredResponse], OffsetResult]:
    pair = find_session_pair(session_dir)
    if pair is None:
        raise FileNotFoundError(f"EAG+GRF 쌍을 찾을 수 없음: {session_dir}")
    sa = SyncAnalyzer(pair)

    off, trans, signed, grf_t = compute_offset(sa, channels[0] - 1, recompute_offset)
    te_corr = sa.unified_time_eag - off.residual  # 보정된 EAG 시간축

    responses = extract_responses(sa, trans, te_corr, channels)

    if plot:
        _plot(sa, trans, responses, off, channels[0], te_corr, session_dir)
    return responses, off


def extract_responses(sa: SyncAnalyzer, trans: List[GRFTransition],
                      te_corr: np.ndarray, channels: List[int],
                      lat_lo: float = -0.5, lat_hi: float = 1.5) -> List[TriggeredResponse]:
    """GRF 전이(anchor)별 × 채널별 EAG knee-pair 추출 (파일 IO 없음, 재사용용).

    각 GRF 전이 시점 부근 [lat_lo, lat_hi]에서 가장 가까운(동시) EAG edge를 골라
    그 onset/offset 두 knee와 변화 크기(amplitude)를 기록한다. GRF가 anchor이므로
    messy 채널에서도 전이당 반응이 빠짐없이 정렬된다.

    te_corr: offset 보정이 반영된 EAG 시간축. 동시성 가정이라 창을 ±방향으로 연다.
    """
    responses: List[TriggeredResponse] = []
    subject = getattr(sa.pair, 'subject_name', None)
    session = getattr(sa.pair, 'session_name', None)
    for ch in channels:
        # 수동 확정 edge(edge_store)가 있으면 자동 검출을 무시하고 그것을 사용
        man = edge_store.get_channel_edges(subject, session, ch) if subject else None
        if man is not None:
            edges_c = [(e['onset_time'], e['onset_amp'],
                        e['offset_amp'] - e['onset_amp'],
                        e['offset_time'], e['offset_amp']) for e in man]
        else:
            eag_c = detrend(sa.eag_filtered[:, ch - 1])
            # anchor(=프로토콜 이벤트)가 주어지면 놓친 약한 반응을 국소 보강해 회수한다
            edges_c = detect_eag_edges_protocol(te_corr, eag_c, trans,
                                                fs=sa.eag.sample_rate,
                                                lat_lo=lat_lo, lat_hi=lat_hi)
        e_on = np.array([e[0] for e in edges_c]) if edges_c else np.array([])
        for tr in trans:
            matched = False
            on_t = on_a = off_t = off_a = amp = tt = slope = lat = np.nan
            eag_dir = ''
            if len(e_on):
                sel = np.where((e_on - tr.time >= lat_lo) & (e_on - tr.time <= lat_hi))[0]
                if len(sel):
                    k = sel[np.argmin(np.abs(e_on[sel] - tr.time))]  # 가장 가까운(동시)
                    on_t, on_a, amp, off_t, off_a = edges_c[k]
                    tt = off_t - on_t
                    slope = amp / max(1e-6, tt)
                    lat = on_t - tr.time
                    eag_dir = 'rise' if amp > 0 else 'fall'
                    matched = True
            responses.append(TriggeredResponse(
                trans_id=tr.trans_id, trans_time=tr.time, grf_direction=tr.direction,
                grf_from_level=tr.from_level, grf_to_level=tr.to_level,
                grf_step=tr.to_level - tr.from_level,
                channel=ch, matched=matched,
                onset_time=on_t, onset_amp=on_a, offset_time=off_t, offset_amp=off_a,
                amplitude=amp, transition_time=tt, slope=slope, latency=lat,
                eag_direction=eag_dir))
    return responses


def _plot(sa, trans, responses, off: OffsetResult, ch: int, te_corr, session_dir: str):
    signed = signed_imbalance(sa.grf_left, sa.grf_right)
    eag_c = detrend(sa.eag_filtered[:, ch - 1])
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    a0.plot(sa.unified_time_grf, signed, color='green', lw=1)
    a0.axhline(0, color='gray', lw=0.5)
    for tr in trans:
        a0.axvline(tr.time, color='red', ls='--', alpha=0.5)
    a0.set_ylabel('GRF signed imbalance')
    a0.set_title(f'GRF transitions={len(trans)}  |  offset auto={off.auto_offset:+.3f}s '
                 f'-> corrected={off.corrected_offset:+.3f}s ({off.method})  '
                 f'match {off.match_rate_auto:.0%}->{off.match_rate_corrected:.0%}')
    a0.grid(alpha=0.3)
    a1.plot(te_corr, eag_c, color='#1f77b4', lw=1)
    for tr in trans:
        a1.axvline(tr.time, color='green', ls='--', alpha=0.4)
    for r in responses:
        if r.channel == ch and r.matched:
            col = '#d62728' if r.eag_direction == 'rise' else '#2ca02c'  # rise=red fall=green
            a1.plot([r.onset_time, r.offset_time], [r.onset_amp, r.offset_amp],
                    'o-', color=col, ms=7, lw=2.2, zorder=5)
    a1.set_ylabel(f'EAG Ch{ch} detrended (uV)')
    a1.set_xlabel('time (s)')
    a1.set_title(f'GRF-triggered knee (Ch{ch}, offset-corrected)')
    a1.grid(alpha=0.3)
    a1.set_xlim(te_corr[0], te_corr[-1])
    plt.tight_layout()
    out = Path(session_dir) / 'grf_triggered_annotation.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  → {out}")


def save_csv(responses: List[TriggeredResponse], session_dir: str):
    df = pd.DataFrame([asdict(r) for r in responses])
    out = Path(session_dir) / 'grf_triggered_params.csv'
    df.to_csv(out, index=False)
    print(f"  → {out}")
    return df


def parse_channels(spec: str) -> List[int]:
    chs = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            chs.extend(range(int(a), int(b) + 1))
        else:
            chs.append(int(part))
    return [c for c in chs if 1 <= c <= EEG_CHANNELS]


import sys
import contextlib
import io


@contextlib.contextmanager
def _quiet():
    """SyncAnalyzer의 stdout 로그 억제."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def offset_report(base_dir: str, ref_channel: int = 1,
                  out_csv: str = 'result/offset_report.csv') -> pd.DataFrame:
    """전체 data 배치 offset 진단 리포트 (annotation 없이 offset만, 빠름)."""
    pairs = find_all_pairs(base_dir)
    dirs = sorted({str(Path(p.eag_filepath).parent) for p in pairs})
    print(f"offset 진단: {len(dirs)}개 세션")
    rows = []
    for i, d in enumerate(dirs):
        pair = find_session_pair(d)
        if pair is None:
            continue
        try:
            with _quiet():
                sa = SyncAnalyzer(pair)
                off, trans, _, _ = compute_offset(sa, ref_channel - 1, True)
            rows.append({
                'subject': pair.subject_name, 'session': pair.session_name,
                'n_grf_trans': off.n_grf_trans,
                'auto_offset': round(off.auto_offset, 3),
                'residual': round(off.residual, 3),
                'corrected_offset': round(off.corrected_offset, 3),
                'match_auto': round(off.match_rate_auto, 3),
                'match_corrected': round(off.match_rate_corrected, 3),
                'latMAD_auto': round(off.latency_mad_auto, 3),
                'latMAD_corrected': round(off.latency_mad_corrected, 3),
                'method': off.method,
                'corrected': abs(off.residual) > 1e-9,
            })
        except Exception as e:
            rows.append({'subject': pair.subject_name, 'session': pair.session_name,
                         'method': f'ERROR: {e}', 'corrected': False})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dirs)}")
    df = pd.DataFrame(rows)
    out = Path(base_dir).parent / out_csv if not Path(out_csv).is_absolute() else Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # 요약
    err = df[df.method.astype(str).str.startswith('ERROR')]
    corr = df[df.get('corrected', False) == True]
    ok = df[(df.get('corrected', False) == False) & (~df.index.isin(err.index))]
    print(f"\n=== offset 진단 요약 ({len(df)}개 세션) ===")
    print(f"  auto 유지: {len(ok)}  |  edge-align 교정: {len(corr)}  |  에러: {len(err)}")
    if len(corr):
        gross = corr[corr.auto_offset.abs() > 0.7]
        print(f"  gross 오차 교정(|auto|>0.7s): {len(gross)}개")
        print(corr.sort_values('residual')[
            ['subject', 'session', 'auto_offset', 'corrected_offset',
             'match_auto', 'match_corrected', 'latMAD_auto', 'latMAD_corrected']
        ].head(20).to_string(index=False))
    print(f"\n리포트 저장: {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description='GRF-triggered EAG annotation')
    ap.add_argument('--session', '-s', help='세션 디렉터리 (EAG+GRF 쌍)')
    ap.add_argument('--dir', '-d', help='data 루트 batch')
    ap.add_argument('--channels', '-c', default='1', help='채널 (예: 1 또는 1-8 또는 1,3,5)')
    ap.add_argument('--offset-report', action='store_true',
                    help='--dir 전체 offset 진단 리포트만 생성 (annotation 생략)')
    ap.add_argument('--no-recompute-offset', action='store_true')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    if args.offset_report:
        if not args.dir:
            ap.error('--offset-report 는 --dir 필요')
        offset_report(args.dir, ref_channel=parse_channels(args.channels)[0])
        return

    chans = parse_channels(args.channels)

    def run_one(sdir):
        try:
            resp, off = annotate_session(
                sdir, chans, recompute_offset=not args.no_recompute_offset,
                plot=not args.no_plot)
            save_csv(resp, sdir)
            print(f"  offset auto={off.auto_offset:+.3f} residual={off.residual:+.3f} "
                  f"corrected={off.corrected_offset:+.3f} | match {off.match_rate_auto:.0%}->"
                  f"{off.match_rate_corrected:.0%} | latMAD {off.latency_mad_auto:.3f}->"
                  f"{off.latency_mad_corrected:.3f} | {off.method}")
        except Exception as e:
            print(f"  [SKIP] {sdir}: {e}")

    if args.dir:
        pairs = find_all_pairs(args.dir)
        dirs = sorted({str(Path(p.eag_filepath).parent) for p in pairs})
        print(f"batch: {len(dirs)}개 세션")
        for d in dirs:
            print(f"[{Path(d).name}]")
            run_one(d)
    elif args.session:
        run_one(args.session)
    else:
        ap.error('--session 또는 --dir 필요')


if __name__ == '__main__':
    main()
