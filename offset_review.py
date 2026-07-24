"""
Offset 수동 검토 UI — GRF-triggered offset 자동보정이 애매/실패한 세션을 사람이 확인·확정

자동 파이프라인(grf_triggered_annotator.compute_offset)은 대부분을 처리하지만,
준주기 신호의 cycle-skip으로 애매하거나(margin 작음), 검색한계에 걸리거나,
"큰데 맞는" offset(기기 녹화 시작 차이)은 사람 확인이 필요하다(needs_review).

이 도구는:
  1. --dir data       : 전체 진단 → needs_review 세션 worklist + 검토 패널 PNG 생성
  2. --session <dir>   : 단일 세션 검토 패널 생성 (match-profile + auto/corrected 오버레이)
  3. --set --subject S --session SS --offset V : manual_offsets.json에 확정 저장
  4. --clear / --list  : 수동 offset 제거 / 목록

manual_offsets.json은 SyncAnalyzer가 자동 조회하므로, 저장 후 parameter_extractor를
재실행하면 해당 세션은 확정된 offset을 사용한다 (자동보정 무시).
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend

from sync_analyzer import find_all_pairs, find_session_pair, SyncAnalyzer
import grf_triggered_annotator as G
from offset_manager import (set_manual_offset, clear_manual_offset,
                            list_all_offsets, get_manual_offset)

REVIEW_DIR = Path('result') / 'offset_review'


@contextlib.contextmanager
def _quiet():
    old = __import__('sys').stdout
    __import__('sys').stdout = io.StringIO()
    try:
        yield
    finally:
        __import__('sys').stdout = old


def _load(session_dir: str):
    """세션 로드 → 진단에 필요한 배열/offset 결과 반환."""
    pair = find_session_pair(session_dir)
    if pair is None:
        raise FileNotFoundError(f"EAG+GRF 쌍 없음: {session_dir}")
    with _quiet():
        sa = SyncAnalyzer(pair)
        off, trans, signed, grf_t = G.compute_offset(sa, 0, True)
    te = sa.unified_time_eag
    tg = sa.unified_time_grf
    eag = detrend(sa.eag_filtered[:, 0])
    edges = G.detect_eag_edges(te, eag, fs=sa.eag.sample_rate)
    eag_edge_t = np.array([e[0] for e in edges]) if edges else np.array([])
    return pair, sa, off, trans, grf_t, tg, signed, te, eag, eag_edge_t


def review_panel(session_dir: str, out_png: Path = None) -> dict:
    """검토 패널 PNG 생성: match-profile + auto/corrected 오버레이."""
    pair, sa, off, trans, grf_t, tg, signed, te, eag, eag_edge_t = _load(session_dir)

    # match-rate 프로파일 (±10s)
    cand, prof, best_res, margin = G.offset_match_profile(grf_t, eag_edge_t)

    def nz(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-9)

    fig, axs = plt.subplots(3, 1, figsize=(18, 11))
    # (1) match-rate 프로파일
    ax = axs[0]
    if len(cand):
        ax.plot(cand, prof, color='purple', lw=1.2)
        ax.axvline(0, color='gray', ls='-', alpha=0.5, label='auto (res=0)')
        ax.axvline(off.residual, color='red', ls='--', lw=2,
                   label=f'chosen res={off.residual:+.2f} ({off.method})')
        ax.axvline(best_res, color='green', ls=':', lw=2,
                   label=f'best-match res={best_res:+.2f}')
    ax.set_xlabel('EAG residual shift (s)')
    ax.set_ylabel('match rate')
    ax.set_title(f'{pair.subject_name}/{pair.session_name} — match-rate profile '
                 f'| auto={off.auto_offset:+.2f} corrected={off.corrected_offset:+.2f} '
                 f'| margin={margin:.2f} | REVIEW: {off.review_reason}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (2) AUTO 정렬 오버레이 (residual=0)
    ax = axs[1]
    ax.plot(tg, nz(signed), color='green', lw=1.2, label='GRF signed')
    ax.plot(te, nz(eag), color='#1f77b4', lw=1, alpha=0.8,
            label=f'EAG @ AUTO ({off.auto_offset:+.2f})')
    for t in grf_t:
        ax.axvline(t, color='gray', ls=':', alpha=0.4)
    ax.set_title('AUTO offset overlay'); ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlim(tg[0], tg[-1])

    # (3) CHOSEN(corrected) 정렬 오버레이
    ax = axs[2]
    ax.plot(tg, nz(signed), color='green', lw=1.2, label='GRF signed')
    ax.plot(te - off.residual, nz(eag), color='#d62728', lw=1, alpha=0.8,
            label=f'EAG @ CORRECTED ({off.corrected_offset:+.2f})')
    for t in grf_t:
        ax.axvline(t, color='gray', ls=':', alpha=0.4)
    ax.set_title('CORRECTED offset overlay'); ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlim(tg[0], tg[-1]); ax.set_xlabel('time (s)')

    plt.tight_layout()
    if out_png is None:
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out_png = REVIEW_DIR / f'{pair.subject_name}_{pair.session_name}_review.png'
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return {
        'subject': pair.subject_name, 'session': pair.session_name,
        'auto_offset': round(off.auto_offset, 3),
        'corrected_offset': round(off.corrected_offset, 3),
        'best_match_res': round(best_res, 3),
        'match_corrected': round(off.match_rate_corrected, 3),
        'margin': round(margin, 3) if np.isfinite(margin) else None,
        'method': off.method, 'reason': off.review_reason,
        'panel': str(out_png),
        'has_manual': get_manual_offset(pair.subject_name, pair.session_name) is not None,
    }


def review_batch(base_dir: str):
    """전체 진단 → needs_review 세션만 패널 생성 + worklist 저장."""
    pairs = find_all_pairs(base_dir)
    dirs = sorted({str(Path(p.eag_filepath).parent) for p in pairs})
    print(f"검토 스캔: {len(dirs)}개 세션")
    work = []
    for i, d in enumerate(dirs):
        try:
            pair, sa, off, *_ = _load(d)
        except Exception as e:
            print(f"  [SKIP] {Path(d).name}: {e}")
            continue
        if off.needs_review:
            row = review_panel(d)
            work.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dirs)}  (review 대상 {len(work)})")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(work)
    out = REVIEW_DIR / 'worklist.csv'
    df.to_csv(out, index=False)
    print(f"\n=== 검토 대상 {len(work)}개 세션 ===")
    if len(df):
        print(df[['subject', 'session', 'auto_offset', 'corrected_offset',
                  'best_match_res', 'match_corrected', 'margin', 'reason']].to_string(index=False))
    print(f"\nworklist: {out}")
    print(f"패널 PNG: {REVIEW_DIR}/")
    print("확정: python3 offset_review.py --set --subject <S> --session <SS> --offset <V>")
    return df


def main():
    ap = argparse.ArgumentParser(description='EAG-GRF offset 수동 검토 UI')
    ap.add_argument('--dir', '-d', help='전체 batch 검토 스캔')
    ap.add_argument('--session', '-s', help='단일 세션 검토 패널')
    ap.add_argument('--set', action='store_true', help='manual offset 확정')
    ap.add_argument('--clear', action='store_true', help='manual offset 제거')
    ap.add_argument('--list', action='store_true', help='설정된 manual offset 목록')
    ap.add_argument('--subject')
    ap.add_argument('--session-name', help='--set/--clear 시 세션명')
    ap.add_argument('--offset', type=float, help='확정할 offset 값(초)')
    ap.add_argument('--note', default='manual review', help='메모')
    args = ap.parse_args()

    if args.list:
        rows = list_all_offsets()
        if not rows:
            print("설정된 manual offset 없음")
        for r in rows:
            print(f"  {r['subject']}/{r['session']}: {r['manual_offset']:+.3f}s "
                  f"(auto={r.get('auto_offset', 0):+.3f}, {r.get('note', '')})")
        return

    if args.set:
        if not (args.subject and args.session_name and args.offset is not None):
            ap.error('--set 은 --subject --session-name --offset 필요')
        set_manual_offset(args.subject, args.session_name, args.offset,
                          note=args.note)
        print(f"✅ 저장: {args.subject}/{args.session_name} = {args.offset:+.3f}s")
        print("   parameter_extractor 재실행 시 이 offset 사용됨")
        return

    if args.clear:
        if not (args.subject and args.session_name):
            ap.error('--clear 은 --subject --session-name 필요')
        ok = clear_manual_offset(args.subject, args.session_name)
        print("✅ 제거됨" if ok else "⚠️ 해당 항목 없음")
        return

    if args.dir:
        review_batch(args.dir)
    elif args.session:
        info = review_panel(args.session)
        print(f"검토 패널: {info['panel']}")
        print(f"  auto={info['auto_offset']:+.3f} corrected={info['corrected_offset']:+.3f} "
              f"best-match={info['best_match_res']:+.3f} margin={info['margin']} "
              f"| {info['reason']}")
    else:
        ap.error('--dir / --session / --set / --clear / --list 중 하나 필요')


if __name__ == '__main__':
    main()
