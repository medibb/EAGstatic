"""
Edge 프로토콜 검토 — 세션·채널별로 "4 cycle × 2 = 8 이벤트"가 측정됐는지 진단.

연구 프로토콜:
  초기 발구름(offset 세팅) → 한발서기 4회(체중부하를 점진적으로 늘림)
  → 분석 대상은 4회 각각의 부하 시작/종료, 즉 **세션당 8개 이벤트**
  → EAG에서는 fall-rise가 4번 반복되어 knee-pair 8개가 나와야 한다.

이 도구는:
  1. --dir data   : 전 세션 스캔 → result/edge_review/worklist.csv (8개가 안 나온 세션·채널)
  2. --session <d>: 단일 세션 진단표 출력
  3. --list       : worklist 요약

worklist는 edge_app이 읽어 드롭다운에 "검토 필요"로 표시하므로, 사람이 그 세션만
열어 knee를 추가/삭제/이동해 8개를 맞추면 된다 (manual_edges.json이 최우선).

주의: offset이 먼저 확정돼 있어야 한다(te_corr 기준으로 edge가 저장되므로).
      REVIEW_WORKFLOW.md의 "offset 먼저, edge 나중" 원칙.
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from sync_analyzer import find_session_pair, SyncAnalyzer
import grf_triggered_annotator as G
import edge_store

REVIEW_DIR = Path('result') / 'edge_review'
WORKLIST = REVIEW_DIR / 'worklist.csv'


@contextlib.contextmanager
def _quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def diagnose_session(session_dir: str, channels=None) -> list:
    """세션의 채널별 프로토콜 적합성 진단. 채널당 dict 1개 반환."""
    pair = find_session_pair(session_dir)
    if pair is None:
        raise FileNotFoundError(f"EAG+GRF 쌍 없음: {session_dir}")
    with _quiet():
        sa = SyncAnalyzer(pair)
        off, trans, signed, _ = G.compute_offset(sa, 0, True)
    te = sa.unified_time_eag - off.residual
    rest, cycles = G.detect_load_cycles(sa.unified_time_grf, signed)
    anchors = G.cycles_to_transitions(cycles, trans)

    n_ch = int(sa.eag_filtered.shape[1])
    chans = channels or list(range(1, n_ch + 1))
    rows = []
    for ch in chans:
        man = edge_store.get_channel_edges(pair.subject_name, pair.session_name, ch)
        if man is not None:
            e_on = np.array([e['onset_time'] for e in man])
            e_amp = np.array([e['offset_amp'] - e['onset_amp'] for e in man])
            source = 'manual'
        else:
            eag = detrend(sa.eag_filtered[:, ch - 1])
            auto = G.detect_eag_edges_protocol(te, eag, anchors, fs=sa.eag.sample_rate)
            e_on = np.array([e[0] for e in auto]) if auto else np.array([])
            e_amp = np.array([e[4] - e[1] for e in auto]) if auto else np.array([])
            source = 'auto'
        v = G.validate_cycle_edges(cycles, e_on, e_amp, raw_trans=trans)
        rows.append({
            'subject': pair.subject_name, 'session': pair.session_name, 'channel': ch,
            'source': source, 'ok': v['ok'],
            'n_cycles': v['n_cycles'], 'n_matched': v['n_matched'],
            'n_events': v['n_events'], 'n_edges': v['n_edges'],
            'rest_level': round(rest, 3),
            'corrected_offset': round(float(off.corrected_offset), 3),
            'reasons': v['reasons'], 'session_dir': str(session_dir),
        })
    return rows


def scan(base_dir: str, only_bad: bool = True):
    """전 세션 스캔 → worklist.csv."""
    from offset_app import scan_sessions
    sessions = scan_sessions()
    print(f"프로토콜 검토 스캔: {len(sessions)}개 세션 "
          f"(기대: cycle {G.EXPECTED_CYCLES}회 · 이벤트 {G.EXPECTED_EVENTS}개)")
    all_rows, err = [], 0
    for i, s in enumerate(sessions):
        try:
            all_rows.extend(diagnose_session(s['dir']))
        except Exception as e:
            err += 1
            all_rows.append({'subject': s['subject'], 'session': s['session'],
                             'channel': 0, 'source': 'error', 'ok': False,
                             'n_cycles': -1, 'n_matched': -1, 'n_events': -1,
                             'n_edges': -1, 'rest_level': None, 'corrected_offset': None,
                             'reasons': f'{type(e).__name__}: {e}', 'session_dir': s['dir']})
        if (i + 1) % 25 == 0:
            bad = sum(1 for r in all_rows if not r['ok'])
            print(f"  {i+1}/{len(sessions)}  (검토대상 채널 {bad}개, 오류 {err})")

    df = pd.DataFrame(all_rows)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = df[~df['ok']] if only_bad else df
    out.to_csv(WORKLIST, index=False)
    df.to_csv(REVIEW_DIR / 'all_channels.csv', index=False)

    ses = df.groupby(['subject', 'session'])['ok'].agg(['sum', 'count'])
    print(f"\n=== 결과 ===")
    print(f"  세션 {len(ses)}개 · 채널 {len(df)}개")
    print(f"  통과 채널 {int(df['ok'].sum())} / {len(df)}  ({df['ok'].mean():.0%})")
    print(f"  전 채널 통과 세션 {(ses['sum'] == ses['count']).sum()} / {len(ses)}")
    if len(df):
        cyc = df[df['channel'] == 1]['n_cycles'].value_counts().sort_index()
        print(f"  cycle 검출 분포: {dict(cyc)}")
    print(f"\nworklist: {WORKLIST}  (검토 필요 채널 {len(out)}개)")
    print(f"전체표  : {REVIEW_DIR / 'all_channels.csv'}")
    print("→ edge_app에서 해당 세션을 열어 knee를 8개로 맞추세요 (python3 edge_app.py)")
    return df


def main():
    ap = argparse.ArgumentParser(description='EAG edge 프로토콜(4 cycle × 2) 검토')
    ap.add_argument('--dir', '-d', help='전 세션 스캔')
    ap.add_argument('--session', '-s', help='단일 세션 진단')
    ap.add_argument('--channel', '-c', type=int, help='단일 채널만')
    ap.add_argument('--all', action='store_true', help='통과 채널도 worklist에 포함')
    ap.add_argument('--list', action='store_true', help='worklist 요약')
    args = ap.parse_args()

    if args.list:
        if not WORKLIST.exists():
            print(f"{WORKLIST} 없음 — 먼저 `python3 edge_review.py --dir data` 실행")
            return
        df = pd.read_csv(WORKLIST)
        print(f"검토 필요 채널 {len(df)}개 / 세션 {df.groupby(['subject','session']).ngroups}개")
        print(df['reasons'].value_counts().head(12).to_string())
        return

    if args.session:
        rows = diagnose_session(args.session, [args.channel] if args.channel else None)
        df = pd.DataFrame(rows)
        print(df[['subject', 'session', 'channel', 'source', 'ok', 'n_cycles',
                  'n_matched', 'n_events', 'n_edges', 'reasons']].to_string(index=False))
        return

    if args.dir:
        scan(args.dir, only_bad=not args.all)
    else:
        ap.error('--dir / --session / --list 중 하나 필요')


if __name__ == '__main__':
    main()
