#!/usr/bin/env python3
"""신뢰도 파일럿 도구 (ANNOTATION_PROTOCOL.md §5.3 / §8).

허용 오차(tolerance)를 임의로 정하지 않고 두 rater의 백지 주석에서 유도하기 위한
3단 절차를 제공한다. 보행 이벤트 라벨링에서 쓰이는 방식과 같다
(Wu et al. 2022, Front Neurosci: inter-labeler LoA를 먼저 재고 그것을 기준으로 삼음).

  1) sessions  대상 세션을 층화 추출해 pilot_sessions.csv로 동결 (시작 전 커밋할 것)
  2) baseline  그 세션들의 자동 검출 결과를 스냅샷 (auto_baseline.json)
  3) report    rater 디렉터리 2개를 비교해 LoA 산출 + tolerance 후보 제시

사용 예:

    python3 reliability_pilot.py sessions --n 10
    python3 reliability_pilot.py baseline

    # rater가 각자 격리된 저장소에 백지로 작업.
    # 두 rater는 **같은 일**을 각자 해야 한다. 한 명에게 offset만, 다른 한 명에게 edge만
    # 시키면 비교할 공통 항목이 없어 리포트가 통째로 빈다. 따라서 앱 4개 = 포트 4개다.
    R=$PWD/result/reliability        # 상대경로는 실행 위치에 따라 딴 데 쌓인다
    EAG_RESULT_DIR=$R/rater_main python3 offset_app.py --port 8768 --blank
    EAG_RESULT_DIR=$R/rater_main python3 edge_app.py   --port 8769 --blank
    EAG_RESULT_DIR=$R/rater_rel  python3 offset_app.py --port 8770 --blank
    EAG_RESULT_DIR=$R/rater_rel  python3 edge_app.py   --port 8771 --blank
    EAG_RESULT_DIR=$R/rater_ref  python3 offset_app.py --port 8772 --blank
    EAG_RESULT_DIR=$R/rater_ref  python3 edge_app.py   --port 8773 --blank

    # 쌍마다 재는 것이 다르다 (PROTOCOL §8.3). tolerance는 가장 넓은 쌍에서 채택한다.
    python3 reliability_pilot.py report --a rater_main --b rater_rel

세션 표본은 시드를 고정해 재현 가능하게 뽑되, **뽑은 결과를 파일로 동결**한다.
사후에 세션을 고르면 표본 선택 편의가 되기 때문이다.
"""

import argparse
import csv
import io
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REL_DIR = Path('result/reliability')
SESSIONS_CSV = REL_DIR / 'pilot_sessions.csv'
CHANNELS_CSV = REL_DIR / 'pilot_channels.csv'
BASELINE_JSON = REL_DIR / 'auto_baseline.json'
REPORT_DIR = REL_DIR / 'report'

# tolerance 후보 스윕 구간 (초). 매칭 수가 평평해지는 지점을 찾는다.
TOL_SWEEP = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]


# ==================== 공통 ====================

def _quiet():
    import contextlib

    @contextlib.contextmanager
    def cm():
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old
    return cm()


def load_worklist_keys() -> set:
    p = Path('result/offset_review/worklist.csv')
    if not p.exists():
        return set()
    with open(p, encoding='utf-8') as f:
        return {(r.get('subject', ''), r.get('session', '')) for r in csv.DictReader(f)}


def load_excluded_keys() -> dict:
    """표본에서 뺄 대상. 방문 단위(전수조사)와 세션 단위(사람이 표시한 분석제외).

    방문 판정은 cohort_manual.csv 원본이 아니라 manifest의 `audit_ok`를 읽는다.
    원본 CSV의 방문명에는 QC 접미사가 남아 있어 미러의 이름과 다르고, 그 정규화는
    build_flat_view가 이미 한 번 했다. 여기서 다시 하면 규칙이 두 곳으로 갈라진다.
    """
    visits, sessions = set(), set()
    man = Path('data_flat/manifest.csv')
    if man.exists():
        with open(man, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if str(r.get('audit_ok', 'True')).strip().lower() in ('false', '0', 'no'):
                    visits.add(r['visit'])
    exc = Path('result/exclusions.json')
    if exc.exists():
        with open(exc, encoding='utf-8') as f:
            for subj, sess_map in json.load(f).items():
                for sess in sess_map:
                    sessions.add((subj, sess))
    return {'visits': visits, 'sessions': sessions}


def read_pilot_sessions() -> list:
    if not SESSIONS_CSV.exists():
        raise SystemExit(f"{SESSIONS_CSV} 없음. 먼저 `sessions` 서브커맨드를 실행하세요.")
    with open(SESSIONS_CSV, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def bland_altman(diffs: list) -> dict:
    """차이 목록 → bias, SD, 95% LoA. Bland-Altman의 관례를 따른다."""
    n = len(diffs)
    if n == 0:
        return {'n': 0, 'bias': None, 'sd': None, 'loa_lo': None, 'loa_hi': None,
                'loa_halfwidth': None}
    bias = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    return {'n': n, 'bias': bias, 'sd': sd,
            'loa_lo': bias - 1.96 * sd, 'loa_hi': bias + 1.96 * sd,
            'loa_halfwidth': 1.96 * sd}


# ==================== 1) sessions ====================

def cmd_sessions(args):
    """worklist 포함/미포함 두 층에서 절반씩 뽑는다.

    worklist(163세션)는 이미 어려운 쪽으로 치우쳐 있다. 거기서만 뽑으면 최악 신뢰도,
    전체에서만 뽑으면 낙관적 신뢰도가 나오므로 두 층을 같이 본다.

    본 분석에서 빠진 자료는 표본에서도 뺀다. 분석에 쓰지 않을 세션의 일치도로 tolerance를
    정하면 그 tolerance가 적용될 자료와 다른 모집단에서 나온 값이 된다. 두 경로로 빠진다.
      - 방문 단위: cohort_manual.csv의 전수조사 판정(manifest의 audit_ok)
      - 세션 단위: exclusions.json (노이즈 등으로 사람이 표시한 것)
    """
    from offset_app import scan_sessions

    wl = load_worklist_keys()
    excluded = load_excluded_keys()
    with _quiet():
        allsess = scan_sessions()

    hard, easy, dropped = [], [], []
    for s in allsess:
        key = (s['subject'], s['session'])
        if s['subject'] in excluded['visits'] or key in excluded['sessions']:
            dropped.append(s)
            continue
        (hard if key in wl else easy).append(s)

    rng = random.Random(args.seed)
    half = args.n // 2
    pick_hard = rng.sample(hard, min(half, len(hard)))
    pick_easy = rng.sample(easy, min(args.n - len(pick_hard), len(easy)))
    rows = [dict(r, stratum='worklist') for r in pick_hard] + \
           [dict(r, stratum='non-worklist') for r in pick_easy]
    rows.sort(key=lambda r: (r['stratum'], r['subject'], r['session']))

    if SESSIONS_CSV.exists() and not args.force:
        raise SystemExit(f"{SESSIONS_CSV}가 이미 있습니다. 표본을 다시 뽑으면 사전 동결이 "
                         f"깨집니다. 정말 다시 뽑으려면 --force")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    fields = ['stratum', 'subject', 'session', 'dir']
    with open(SESSIONS_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"전체 세션 {len(allsess)} · 제외 {len(dropped)} "
          f"(방문 판정 {len(excluded['visits'])}개 · 세션 표시 {len(excluded['sessions'])}건)")
    print(f"추출 모집단 {len(hard) + len(easy)} (worklist {len(hard)} / 그 외 {len(easy)})")
    print(f"표본 {len(rows)}개 → {SESSIONS_CSV}  (seed={args.seed})")
    for r in rows:
        print(f"  [{r['stratum']:13s}] {r['subject']} / {r['session']}")
    print("\n⚠️ 작업 시작 전에 이 파일을 커밋해 표본을 동결하세요.")


# ==================== 1b) channels ====================

def cmd_channels(args):
    """파일럿의 **채널 분모**를 동결한다. 본 분석과 같은 PASS 규칙을 쓴다.

    노이즈로 오염된 채널은 본 분석(`stats_grf_eag`의 `channel_quality == 'PASS'`)에서
    이미 빠진다. 그런 채널의 불일치를 신뢰도에 넣으면 쓰지도 않을 자료로 tolerance를
    정하게 되므로 분모에서 뺀다.

    **다만 빼는 주체가 rater여서는 안 된다.** rater가 작업 중 어려운 채널을 건너뛰면
    분모가 사람마다 달라지고, 빠지는 쪽이 하필 불일치가 큰 자료라 LoA가 좁아진다
    (informative missingness). 그래서 주석 **전에** 신호 지표만으로 정하고 동결한다.
    `compute_noise_metrics`는 원시 신호만 보므로 rater와 무관하다.

    빼는 것은 '신호 품질'이지 '판정 난이도'가 아니다. 신호는 깨끗한데 knee가 완만해
    판단이 갈리는 채널은 남긴다 — 그건 프로토콜이 감당해야 할 진짜 불확실성이고
    tolerance가 포괄해야 할 대상이다 (§5.3의 Wu et al.: flat foot이 toe off보다 5배 넓다).
    """
    from sync_analyzer import find_session_pair, SyncAnalyzer
    from parameter_extractor import evaluate_channel_quality

    if CHANNELS_CSV.exists() and not args.force:
        raise SystemExit(f"{CHANNELS_CSV}가 이미 있습니다. 분모를 다시 정하면 사전 동결이 "
                         f"깨집니다. 정말 다시 뽑으려면 --force")

    rows = read_pilot_sessions()
    out = []
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {r['subject']} / {r['session']}", flush=True)
        try:
            pair = find_session_pair(r['dir'])
            if pair is None:
                print('   건너뜀 (EAG+GRF 쌍 없음)')
                continue
            with _quiet():
                sa = SyncAnalyzer(pair)
                q = evaluate_channel_quality(sa.eag)
        except Exception as e:
            print(f"   실패: {type(e).__name__}: {e}")
            continue
        for ch0 in sorted(q):
            m = q[ch0]
            out.append({
                'subject': r['subject'], 'session': r['session'],
                # 전극번호(1-based)로 적는다. manual_edges·exclusions의 채널 키가 1-based라
                # 0-based로 두면 한 칸씩 어긋난 채 조용히 합쳐진다.
                'channel': ch0 + 1,
                'include': m['flag'] == 'PASS',
                'quality': m['flag'], 'flags': '|'.join(m['flags']),
                'snr_db': round(float(m['snr_db']), 2),
                'power_hf_ratio': round(float(m['power_hf_ratio']), 4),
            })

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHANNELS_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    n_in = sum(r['include'] for r in out)
    print(f"\n채널 {len(out)}개 중 분모 {n_in}개 (제외 {len(out) - n_in})")
    drop = {}
    for r in out:
        if not r['include']:
            drop[r['quality']] = drop.get(r['quality'], 0) + 1
    for k, v in sorted(drop.items(), key=lambda kv: -kv[1]):
        print(f"  제외 사유 {k}: {v}")
    print(f"→ {CHANNELS_CSV}")
    print("\n⚠️ 작업 시작 전에 커밋해 분모를 동결하세요.")
    print("   rater에게는 include=True 채널만 작업하도록 안내합니다.")


def read_pilot_channels() -> set:
    """동결된 채널 분모 {(subject, session, channel:str)}. 없으면 빈 집합."""
    if not CHANNELS_CSV.exists():
        return set()
    out = set()
    with open(CHANNELS_CSV, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if str(r['include']).strip().lower() in ('true', '1', 'yes'):
                out.add((r['subject'], r['session'], str(r['channel'])))
    return out


# ==================== 2) baseline ====================

def cmd_baseline(args):
    """파일럿 세션의 자동 검출 결과를 스냅샷.

    offset은 저장소에 `auto_offset` 필드가 있지만 edge는 자동 원본이 어디에도 남지
    않는다. 나중에 "사람이 실제로 손댄 이벤트"를 가려내려면 이 스냅샷이 필요하다.
    """
    import numpy as np
    from sync_analyzer import find_session_pair, SyncAnalyzer
    import grf_triggered_annotator as G
    from scipy.signal import detrend

    rows = read_pilot_sessions()
    out = {}
    for i, r in enumerate(rows, 1):
        sdir = r['dir']
        print(f"[{i}/{len(rows)}] {r['subject']} / {r['session']}", flush=True)
        try:
            pair = find_session_pair(sdir)
            if pair is None:
                print('   건너뜀 (EAG+GRF 쌍 없음)')
                continue
            with _quiet():
                sa = SyncAnalyzer(pair)
                off, trans, _s, _g = G.compute_offset(sa, 0, True)
                signed = G.signed_imbalance(sa.grf_left, sa.grf_right)
                tg = sa.unified_time_grf
                rest, cycles, _ = G.detect_load_cycles_expected(
                    tg, signed, sa.grf_left, sa.grf_right)
                anchors = G.cycles_to_transitions(cycles, trans)
            te = sa.unified_time_eag - off.residual
            entry = {
                'auto_offset': round(float(off.auto_offset), 4),
                'residual': round(float(off.residual), 4),
                'corrected_offset': round(float(off.corrected_offset), 4),
                'method': off.method,
                'anchors': [round(float(a.time), 4) for a in anchors],
                'channels': {},
            }
            n_ch = sa.eag_filtered.shape[1]
            for ch in range(1, n_ch + 1):
                eag = detrend(sa.eag_filtered[:, ch - 1])
                with _quiet():
                    auto = G.detect_eag_edges_protocol(
                        te, eag, anchors, fs=sa.eag.sample_rate)
                entry['channels'][str(ch)] = [
                    {'onset_time': round(float(e[0]), 4), 'onset_amp': round(float(e[1]), 2),
                     'offset_time': round(float(e[3]), 4), 'offset_amp': round(float(e[4]), 2)}
                    for e in auto]
            out.setdefault(r['subject'], {})[r['session']] = entry
        except Exception as e:  # 한 세션 실패가 전체를 막지 않게
            print(f"   실패: {type(e).__name__}: {e}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n→ {BASELINE_JSON}  ({sum(len(v) for v in out.values())} 세션)")


# ==================== 3) report ====================

def _load_store(rater: str, filename: str) -> dict:
    p = REL_DIR / rater / filename
    if not p.exists():
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _match_edges(ea: list, eb: list, tol: float):
    """onset 시각 기준 최근접 1:1 매칭 (tol 이내). 반환: (매칭쌍, A단독, B단독).

    tolerance 안이면 '같은 이벤트를 옮긴 것', 밖이면 '삭제 + 추가'로 센다.
    이 규칙을 사전에 고정하지 않으면 수치가 임의로 움직인다 (PROTOCOL §8.1).
    """
    used_b, pairs = set(), []
    for a in sorted(ea, key=lambda e: e['onset_time']):
        best, bd = None, tol
        for j, b in enumerate(eb):
            if j in used_b:
                continue
            d = abs(a['onset_time'] - b['onset_time'])
            if d <= bd:
                best, bd = j, d
        if best is not None:
            used_b.add(best)
            pairs.append((a, eb[best]))
    only_a = len(ea) - len(pairs)
    only_b = len(eb) - len(pairs)
    return pairs, only_a, only_b


def _edge_channels(store: dict) -> set:
    """manual_edges → 주석한 채널 집합 {(subject, session, channel)}."""
    return {(s, ses, ch)
            for s, sm in store.items() for ses, cm in sm.items() for ch in cm}


def _excluded_channels(store: dict, denom: set) -> set:
    """exclusions → 제외 표시된 채널 집합. 채널 '0'은 세션 전체 제외라 펼친다.

    분모(denom)가 있으면 세션 제외를 그 세션의 분모 채널로 펼치고, 없으면 1~8로 펼친다.
    """
    out = set()
    for s, sm in store.items():
        for ses, cm in sm.items():
            for ch in cm:
                if str(ch) == '0':
                    chans = ([c for (a, b, c) in denom if a == s and b == ses]
                             or [str(i) for i in range(1, 9)])
                    out.update((s, ses, c) for c in chans)
                else:
                    out.add((s, ses, str(ch)))
    return out


def _kappa(both_ann: int, a_only: int, b_only: int, both_exc: int):
    """제외/주석 2범주에 대한 Cohen's kappa. 우연 일치를 넘는 부분만 남긴다.

    단순 일치율은 한쪽 범주가 드물면(제외가 거의 없으면) 자동으로 높게 나와
    "판단이 일치했다"의 근거가 되지 못한다.
    """
    n = both_ann + a_only + b_only + both_exc
    if n == 0:
        return None
    po = (both_ann + both_exc) / n
    # a_only = A는 주석 · B는 제외
    pa_ann, pb_ann = (both_ann + a_only) / n, (both_ann + b_only) / n
    pe = pa_ann * pb_ann + (1 - pa_ann) * (1 - pb_ann)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def cmd_report(args):
    ra, rb = args.a, args.b
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# 신뢰도 파일럿 리포트 ({ra} vs {rb})", ""]

    # ---------- 0. 커버리지 (이행 점검) ----------
    # 분모는 pilot_channels.csv로 사전 동결돼 있으므로 편향 보정이 아니라 이행 점검이다.
    # 그래도 반드시 찍는다 — 이 표가 없으면 rater가 빠뜨린 채널을 알아낼 방법이 없고,
    # 아래의 모든 LoA는 조용히 교집합에서만 계산된다.
    denom = read_pilot_channels()
    ea_all, eb_all = _load_store(ra, 'manual_edges.json'), _load_store(rb, 'manual_edges.json')
    xa, xb = _load_store(ra, 'exclusions.json'), _load_store(rb, 'exclusions.json')
    ca, cb = _edge_channels(ea_all), _edge_channels(eb_all)
    xca, xcb = _excluded_channels(xa, denom), _excluded_channels(xb, denom)

    lines += ["## 0. 커버리지", ""]
    if denom:
        done_a, done_b = (ca | xca) & denom, (cb | xcb) & denom
        miss_a, miss_b = denom - done_a, denom - done_b
        lines += [
            f"동결 분모 **{len(denom)}채널** (`pilot_channels.csv`, PASS 규칙)", "",
            "| | 처리 | 미처리 | 이행률 |", "|---|---|---|---|",
            f"| {ra} | {len(done_a)} | {len(miss_a)} | {len(done_a)/len(denom):.1%} |",
            f"| {rb} | {len(done_b)} | {len(miss_b)} | {len(done_b)/len(denom):.1%} |", ""]
        extra = ((ca | xca) | (cb | xcb)) - denom
        if extra:
            lines.append(f"⚠️ 분모 **밖** 채널 작업 {len(extra)}건 — 분모가 사후에 흔들린다. "
                         f"아래 계산에서는 제외했다.")
        for who, miss in ((ra, miss_a), (rb, miss_b)):
            if miss:
                lines.append(f"⚠️ **{who} 미처리 {len(miss)}채널** — 누락이지 편향이 아니다. "
                             f"마저 작업시킨 뒤 리포트를 다시 돌린다.")
                lines += ["", "```"] + [f"  {s}/{ses} ch{c}" for s, ses, c in sorted(miss)[:20]] + ["```"]
        if not miss_a and not miss_b:
            lines.append("✅ 양쪽 모두 분모를 100% 처리했다. 아래 LoA는 분모 전체에서 나온 값이다.")
        lines.append("")
    else:
        lines += ["⚠️ `pilot_channels.csv` 없음 — 분모가 동결되지 않았다. "
                  "아래 LoA는 **두 rater가 우연히 겹친 채널**에서만 계산되며, "
                  "빠지는 쪽이 불일치가 큰 자료에 몰리므로 좁게 나온다. "
                  "`reliability_pilot.py channels`를 먼저 실행할 것.", ""]

    # ---------- 0b. 제외 판단 일치도 ----------
    # "이 채널이 잴 수 있는 자료인가"에 대한 이견은 시점 불일치보다 큰 종류의 불일치인데,
    # edge 비교는 한쪽에 항목이 없다는 이유로 이걸 통째로 건너뛴다. 따로 센다.
    scope = denom or ((ca | xca) & (cb | xcb))
    both_ann = len((ca - xca) & (cb - xcb) & scope)
    a_only = len(((ca - xca) & xcb) & scope)      # A는 주석, B는 제외
    b_only = len((xca & (cb - xcb)) & scope)      # A는 제외, B는 주석
    both_exc = len(xca & xcb & scope)
    k = _kappa(both_ann, a_only, b_only, both_exc)
    lines += ["## 0b. 제외 판단 일치도", "",
              "| | B 주석 | B 제외 |", "|---|---|---|",
              f"| **A 주석** | {both_ann} | {a_only} |",
              f"| **A 제외** | {b_only} | {both_exc} |", ""]
    if k is None:
        lines += ["한쪽 범주가 비어 kappa를 정의할 수 없다 "
                  "(제외 판정이 전혀 없으면 정상이다).", ""]
    else:
        lines += [f"- Cohen's **kappa = {k:.3f}** (단순 일치율 "
                  f"{(both_ann + both_exc) / max(1, both_ann + a_only + b_only + both_exc):.1%})", ""]
    if a_only + b_only:
        lines += [f"⚠️ 한쪽만 제외한 채널 **{a_only + b_only}건**. 시점이 얼마나 다른가가 아니라 "
                  f"*잴 수 있는 자료인가*에 대한 이견이므로, 아래 LoA에는 잡히지 않는다. "
                  f"§5.5 기준을 다시 맞출 대상이다.", ""]

    # ---------- offset ----------
    oa, ob = _load_store(ra, 'manual_offsets.json'), _load_store(rb, 'manual_offsets.json')
    off_rows, off_diffs = [], []
    for subj, sess_map in oa.items():
        for sess, ea in sess_map.items():
            eb = ob.get(subj, {}).get(sess)
            if not eb:
                continue
            va, vb = float(ea['manual_offset']), float(eb['manual_offset'])
            off_rows.append({'subject': subj, 'session': sess,
                             'rater_a': round(va, 4), 'rater_b': round(vb, 4),
                             'diff': round(va - vb, 4)})
            off_diffs.append(va - vb)
    ba_off = bland_altman(off_diffs)

    # 세션 커버리지도 같이 센다. offset은 세션 단위라 채널 분모와 별개다.
    pilot_sess = {(r['subject'], r['session']) for r in read_pilot_sessions()} \
        if SESSIONS_CSV.exists() else set()
    sa_ = {(s, ses) for s, m in oa.items() for ses in m}
    sb_ = {(s, ses) for s, m in ob.items() for ses in m}

    lines += ["## 1. offset (세션 단위)", ""]
    if pilot_sess:
        lines += [f"동결 표본 {len(pilot_sess)}세션 · 확정 {ra} {len(sa_ & pilot_sess)} / "
                  f"{rb} {len(sb_ & pilot_sess)} · 공통 {len(sa_ & sb_ & pilot_sess)}", ""]
        for who, done in ((ra, sa_), (rb, sb_)):
            miss = pilot_sess - done
            if miss:
                lines.append(f"⚠️ **{who} 미확정 {len(miss)}세션**: "
                             + ', '.join(f"{s}/{ses}" for s, ses in sorted(miss)[:10]))
        lines.append("")
    if ba_off['n'] == 0:
        lines += ["두 rater가 공통으로 확정한 세션이 없습니다.", ""]
    else:
        lines += [
            f"- 공통 세션 **{ba_off['n']}개**",
            f"- bias (A - B) = **{ba_off['bias']:+.4f} s**, SD = {ba_off['sd']:.4f} s",
            f"- 95% LoA = **{ba_off['loa_lo']:+.4f} ~ {ba_off['loa_hi']:+.4f} s** "
            f"(반폭 {ba_off['loa_halfwidth']:.4f})",
            "",
            f"→ **offset tolerance 후보: {_round_tol(ba_off['loa_halfwidth'])} s**", ""]

    # ---------- edge ----------
    # 분모가 동결돼 있으면 그 안에서만 센다. 분모 밖 채널을 섞으면 사람마다 다른
    # 모집단에서 계산되고, §0에서 이행률을 보고한 대상과도 달라진다.
    eaS, ebS = ea_all, eb_all
    sweep, per_tol = [], {}
    for tol in TOL_SWEEP:
        n_pair = n_a = n_b = 0
        d_on, d_off, d_amp = [], [], []
        for subj, sess_map in eaS.items():
            for sess, ch_map in sess_map.items():
                for ch, entry in ch_map.items():
                    if denom and (subj, sess, str(ch)) not in denom:
                        continue
                    other = ebS.get(subj, {}).get(sess, {}).get(ch)
                    if not other:
                        continue
                    pairs, only_a, only_b = _match_edges(
                        entry.get('edges', []), other.get('edges', []), tol)
                    n_pair += len(pairs); n_a += only_a; n_b += only_b
                    for a, b in pairs:
                        d_on.append(a['onset_time'] - b['onset_time'])
                        d_off.append(a['offset_time'] - b['offset_time'])
                        d_amp.append((a['offset_amp'] - a['onset_amp'])
                                     - (b['offset_amp'] - b['onset_amp']))
        f1 = (2 * n_pair / (2 * n_pair + n_a + n_b)) if (2 * n_pair + n_a + n_b) else 0.0
        sweep.append({'tol': tol, 'matched': n_pair, 'only_a': n_a, 'only_b': n_b,
                      'f1': round(f1, 4)})
        per_tol[tol] = {'onset': bland_altman(d_on), 'offset': bland_altman(d_off),
                        'amp': bland_altman(d_amp)}

    lines += ["## 2. edge (이벤트 단위)", "",
              "### 2.1 tolerance 스윕", "",
              "| tol (s) | 매칭 | A 단독 | B 단독 | F1 |", "|---|---|---|---|---|"]
    for s in sweep:
        lines.append(f"| {s['tol']:.2f} | {s['matched']} | {s['only_a']} | "
                     f"{s['only_b']} | {s['f1']:.3f} |")
    lines += ["", "F1이 평평해지기 시작하는 tolerance가 실질 상한입니다. "
                  "그보다 키우면 서로 다른 이벤트를 억지로 묶기 시작합니다.", ""]

    ref = args.ref_tol if args.ref_tol else _plateau_tol(sweep)
    st = per_tol.get(ref)
    lines += [f"### 2.2 시점·진폭 일치도 (tol = {ref} s 기준)", ""]
    if st and st['onset']['n']:
        for key, label, unit in (('onset', 'onset 시각', 's'),
                                 ('offset', 'offset 시각', 's'),
                                 ('amp', 'amplitude', 'µV')):
            b = st[key]
            lines.append(f"- **{label}**: n={b['n']}, bias {b['bias']:+.4f} {unit}, "
                         f"95% LoA {b['loa_lo']:+.4f} ~ {b['loa_hi']:+.4f} {unit} "
                         f"(반폭 {b['loa_halfwidth']:.4f})")
        lines += ["",
                  f"→ **onset tolerance 후보: {_round_tol(st['onset']['loa_halfwidth'])} s**",
                  f"→ **offset tolerance 후보: {_round_tol(st['offset']['loa_halfwidth'])} s**",
                  ""]
    else:
        lines += ["두 rater가 공통으로 확정한 채널이 없습니다.", ""]

    # ---------- 오차 예산 ----------
    lines += ["## 3. 오차 예산 (PROTOCOL §8.5)", "",
              "Ch.4 test-retest **MDC95 = 1.784** (평균 기울기 1.45).",
              "필요한 진술은 `annotation LoA ≪ 1.784` 이다.", ""]
    if st and st['amp']['n']:
        lines.append(f"- 진폭 annotation LoA 반폭 = {st['amp']['loa_halfwidth']:.3f} µV")
    lines += ["- 기울기 환산은 파라미터 재추출 후 `reliability_report` 확장으로 산출한다", ""]

    lines += ["## 4. 확정 절차", "",
              "위 후보값을 반올림해 `ANNOTATION_PROTOCOL.md` §5.3 표에 기입하고,",
              "확정일과 근거(이 리포트 경로)를 함께 적은 뒤 커밋한다.", ""]

    # 저장
    if off_rows:
        with open(REPORT_DIR / 'offset_pairs.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(off_rows[0].keys()))
            w.writeheader(); w.writerows(off_rows)
    with open(REPORT_DIR / 'tolerance_sweep.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['tol', 'matched', 'only_a', 'only_b', 'f1'])
        w.writeheader(); w.writerows(sweep)
    md = REPORT_DIR / 'pilot_report.md'
    md.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print(f"\n→ {md}")


def _round_tol(x):
    """LoA 반폭을 사람이 쓰는 눈금으로 반올림 (0.02/0.05/0.10 … )."""
    if x is None:
        return None
    for step in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
        if x <= step:
            return step
    return round(x, 2)


def _plateau_tol(sweep):
    """F1 증가폭이 0.01 미만으로 떨어지는 첫 tolerance."""
    for prev, cur in zip(sweep, sweep[1:]):
        if cur['f1'] - prev['f1'] < 0.01:
            return prev['tol']
    return sweep[-1]['tol'] if sweep else 0.15


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('sessions', help='파일럿 대상 세션 층화 추출 후 동결')
    p1.add_argument('--n', type=int, default=10)
    p1.add_argument('--seed', type=int, default=20260818)
    p1.add_argument('--force', action='store_true', help='기존 표본을 덮어쓴다')
    p1.set_defaults(func=cmd_sessions)

    p1b = sub.add_parser('channels', help='채널 분모 동결 (PASS 규칙, 주석 전에 실행)')
    p1b.add_argument('--force', action='store_true', help='기존 분모를 덮어쓴다')
    p1b.set_defaults(func=cmd_channels)

    p2 = sub.add_parser('baseline', help='자동 검출 스냅샷 생성')
    p2.set_defaults(func=cmd_baseline)

    p3 = sub.add_parser('report', help='rater 두 명 비교 → LoA · tolerance 후보')
    p3.add_argument('--a', default='rater_main')
    p3.add_argument('--b', default='rater_rel')
    p3.add_argument('--ref-tol', type=float, default=None,
                    help='시점·진폭 LoA를 볼 기준 tolerance (기본: 스윕 plateau)')
    p3.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
