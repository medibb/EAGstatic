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

    # rater가 각자 격리된 저장소에 백지로 작업
    EAG_RESULT_DIR=result/reliability/rater_A python3 offset_app.py --port 8768 --blank
    EAG_RESULT_DIR=result/reliability/rater_B python3 edge_app.py   --port 8769 --blank

    python3 reliability_pilot.py report --a rater_A --b rater_B

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
    """
    from offset_app import scan_sessions

    wl = load_worklist_keys()
    with _quiet():
        allsess = scan_sessions()

    hard, easy = [], []
    for s in allsess:
        key = (s['subject'], s['session'])
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

    print(f"전체 세션 {len(allsess)} (worklist {len(hard)} / 그 외 {len(easy)})")
    print(f"표본 {len(rows)}개 → {SESSIONS_CSV}  (seed={args.seed})")
    for r in rows:
        print(f"  [{r['stratum']:13s}] {r['subject']} / {r['session']}")
    print("\n⚠️ 작업 시작 전에 이 파일을 커밋해 표본을 동결하세요.")


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


def cmd_report(args):
    ra, rb = args.a, args.b
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# 신뢰도 파일럿 리포트 ({ra} vs {rb})", ""]

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

    lines += ["## 1. offset (세션 단위)", ""]
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
    eaS, ebS = _load_store(ra, 'manual_edges.json'), _load_store(rb, 'manual_edges.json')
    sweep, per_tol = [], {}
    for tol in TOL_SWEEP:
        n_pair = n_a = n_b = 0
        d_on, d_off, d_amp = [], [], []
        for subj, sess_map in eaS.items():
            for sess, ch_map in sess_map.items():
                for ch, entry in ch_map.items():
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

    p2 = sub.add_parser('baseline', help='자동 검출 스냅샷 생성')
    p2.set_defaults(func=cmd_baseline)

    p3 = sub.add_parser('report', help='rater 두 명 비교 → LoA · tolerance 후보')
    p3.add_argument('--a', default='rater_A')
    p3.add_argument('--b', default='rater_B')
    p3.add_argument('--ref-tol', type=float, default=None,
                    help='시점·진폭 LoA를 볼 기준 tolerance (기본: 스윕 plateau)')
    p3.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
