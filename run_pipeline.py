#!/usr/bin/env python3
"""
EAG-GRF 파이프라인 러너 — 자동화 단계를 한 명령으로 실행하고 진행상태를 요약.

파이프라인 5단계 중 2·3(offset/edge 수동검토)은 사람이 하는 게이트이고,
1·4·5(진단·파라미터추출·통계)는 자동이다. 이 스크립트는 자동 단계를 묶어
실행하고, `status`로 남은 수동검토 물량을 보여준다.

단계(stage):
  status  현재 진행상태 요약 (offset_report / worklist / manual override)
  diag    1. offset 진단  → grf_triggered_annotator.py --offset-report
  params  4. 파라미터 추출 → parameter_extractor.py --batch
  stats   5. dose-response 통계 → stats_grf_eag.py
  analyze = params + stats  (수동검토 끝난 뒤 재실행용 빠른 경로)
  all     = diag + params + stats

사용:
  python3 run_pipeline.py status                 # 남은 검토 물량 확인
  python3 run_pipeline.py all                    # 진단→추출→통계 일괄
  python3 run_pipeline.py analyze --exclude-review   # 검토 후 재분석
  python3 run_pipeline.py params --phase3        # Phase3(주파수)까지 추출
  python3 run_pipeline.py all --dry-run          # 실행할 명령만 출력

기본 stage(인자 없음)는 `analyze`(params+stats).
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # 현재 인터프리터로 하위 스크립트 실행 (Windows/Linux 공용)

STAGE_ORDER = ['status', 'diag', 'params', 'stats']
EXPAND = {'analyze': ['params', 'stats'], 'all': ['diag', 'params', 'stats']}


def build_commands(args):
    """선택된 stage → 실행할 명령 리스트(캐노니컬 순서)."""
    stages = []
    for s in args.stages:
        stages.extend(EXPAND.get(s, [s]))
    # 중복 제거 + 캐노니컬 순서
    stages = [s for s in STAGE_ORDER if s in set(stages)]

    cmds = []
    for s in stages:
        if s == 'status':
            cmds.append(('status', None))  # 내부 함수로 처리
        elif s == 'diag':
            cmds.append((s, [PY, 'grf_triggered_annotator.py',
                             '--dir', args.dir, '--channels', args.channels,
                             '--offset-report']))
        elif s == 'params':
            c = [PY, 'parameter_extractor.py', '--batch']
            if args.phase3:
                c.append('--phase3')
            if args.reprocess_manual:
                c.append('--reprocess-manual')
            cmds.append((s, c))
        elif s == 'stats':
            c = [PY, 'stats_grf_eag.py']
            if args.exclude_review:
                c.append('--exclude-review')
            if args.all_channels:
                c.append('--all-channels')
            cmds.append((s, c))
    return cmds


def print_status():
    """offset_report / worklist / manual override 현황 요약."""
    try:
        import pandas as pd
    except ImportError:
        print("[status] pandas 필요 (pip install -r requirements.txt)")
        return
    import json

    print("=" * 60)
    print("진행상태 (status)")
    print("=" * 60)

    rep = ROOT / 'result' / 'offset_report.csv'
    if rep.exists():
        df = pd.read_csv(rep)
        m = df['method'].astype(str)
        large = int(m.str.startswith('large-offset').sum())
        rej = int(m.str.startswith('auto (correction').sum())
        xcorr = int(m.str.startswith('fallback-xcorr').sum())
        auto_ok = len(df) - large - rej - xcorr
        print(f"offset_report: {len(df)} 세션")
        print(f"  auto 정렬 OK        : {auto_ok}")
        print(f"  large-offset 재검토  : {large}")
        print(f"  correction rejected : {rej}")
        print(f"  fallback-xcorr      : {xcorr}")
    else:
        print("offset_report.csv 없음 → 먼저 `diag` 실행")

    wl = ROOT / 'result' / 'offset_review' / 'worklist.csv'
    if wl.exists():
        w = pd.read_csv(wl)
        done = int(w['has_manual'].sum()) if 'has_manual' in w else 0
        remain = len(w) - done
        print(f"\n수동검토 worklist: {len(w)} 세션 "
              f"(확정 {done} / 남음 {remain})")
        if 'reason' in w and remain:
            cat = w[~w['has_manual']]['reason'].astype(str) if 'has_manual' in w \
                else w['reason'].astype(str)
            big = cat.str.contains('큰offset|큰교정').sum()
            lowm = cat.str.contains('저match').sum()
            edge = cat.str.contains('edge매칭부족').sum()
            print(f"  edge 매칭부족(대체로 저위험, 빠른 확인): {int(edge)}")
            print(f"  저match(판단 필요)                     : {int(lowm)}")
            print(f"  큰offset/큰교정보류(우선 검토)          : {int(big)}")
    else:
        print("\nworklist.csv 없음")

    mo = ROOT / 'result' / 'manual_offsets.json'
    me = ROOT / 'result' / 'manual_edges.json'
    n_off = sum(len(v) for v in json.load(open(mo)).values()) if mo.exists() else 0
    n_edge = len(json.load(open(me))) if me.exists() else 0
    print(f"\n확정 override: manual_offsets {n_off} 세션 · manual_edges {n_edge} 키")

    print("\n다음 액션:")
    print("  0) 평면 미러 갱신: python3 build_flat_view.py")
    print("  1) offset 검토: python3 offset_review.py --dir data_flat")
    print("  2) edge 검토  : python3 edge_app.py --host 0.0.0.0 --port 8765")
    print("  3) 검토 후    : python3 run_pipeline.py analyze")
    print("=" * 60)


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    print(f"[{'OK' if r.returncode == 0 else 'FAIL rc=%d' % r.returncode}] "
          f"{dt:.1f}s")
    return r.returncode


def main():
    ap = argparse.ArgumentParser(
        description='EAG-GRF 파이프라인 러너',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('stages', nargs='*',
                    choices=['status', 'diag', 'params', 'stats',
                             'analyze', 'all'],
                    help="실행할 단계 (기본: analyze = params+stats)")
    ap.add_argument('--dir', default='data_flat',
                    help='데이터 디렉토리 (diag). 기본은 평면 미러(data_flat)')
    ap.add_argument('--channels', default='1', help='EAG 채널 (diag)')
    ap.add_argument('--phase3', action='store_true',
                    help='파라미터 추출 시 Phase3(주파수)까지 (params)')
    ap.add_argument('--reprocess-manual', action='store_true',
                    help='manual override 세션만 재처리 (params)')
    ap.add_argument('--exclude-review', action='store_true',
                    help='needs_review offset 세션 제외 (stats)')
    ap.add_argument('--all-channels', action='store_true',
                    help='품질필터 없이 전체 채널 (stats)')
    ap.add_argument('--dry-run', action='store_true',
                    help='실행 대신 명령만 출력')
    args = ap.parse_args()
    if not args.stages:
        args.stages = ['analyze']  # 기본: params + stats

    cmds = build_commands(args)
    if not cmds:
        print("실행할 단계 없음."); return

    for name, cmd in cmds:
        if name == 'status':
            print_status()
            continue
        if args.dry_run:
            print("$ " + " ".join(cmd))
            continue
        if run(cmd) != 0:
            print(f"\n[중단] '{name}' 단계 실패 → 이후 단계 건너뜀.")
            sys.exit(1)


if __name__ == '__main__':
    main()
