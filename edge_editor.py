"""
Edge Editor — 최종 EAG edge(knee) 수동 검토·수정 도구 (headless: PNG 검토 + CLI 편집).

GRF-anchored 자동 검출이 부정확한 세션/채널에서, 번호 라벨 PNG를 보고 잘못된 edge를
삭제/추가/이동으로 최종 수정한다. 확정된 edge 목록은 edge_store(manual_edges.json)에
동결되고, parameter_extractor가 자동으로 반영한다 (자동 검출 무시).

워크플로우:
  # 1) 검토 PNG 생성 (edge에 [id] 라벨, GRF 전이선 표시)
  python3 edge_editor.py review  --session <dir> --channel 1
  # 2) 수정
  python3 edge_editor.py delete  --session <dir> --channel 1 --id 3
  python3 edge_editor.py add     --session <dir> --channel 1 --onset 30.1 --offset 30.6 --snap
  python3 edge_editor.py move    --session <dir> --channel 1 --id 2 --onset 19.2 --offset 19.7 --snap
  # 3) 재검토 → 확정되면 parameter_extractor 재실행
  python3 edge_editor.py reset   --session <dir> --channel 1   # 자동으로 복귀
  python3 edge_editor.py list

좌표계: 모든 edge time은 corrected offset이 반영된 EAG 시간축(te_corr) 기준.
"""

import argparse
import io
import sys
import contextlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import detrend

from sync_analyzer import find_session_pair, SyncAnalyzer
import grf_triggered_annotator as G
import edge_store

EDIT_DIR = Path('result') / 'edge_edit'


@contextlib.contextmanager
def _quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


# ==================== 로드 / 좌표 ====================

def _load(session_dir: str, channel: int, recompute: bool = True):
    """세션 로드 → (pair, sa, te_corr, eag_c, trans, off)."""
    pair = find_session_pair(session_dir)
    if pair is None:
        raise FileNotFoundError(f"EAG+GRF 쌍 없음: {session_dir}")
    with _quiet():
        sa = SyncAnalyzer(pair)
        off, trans, _signed, _grf = G.compute_offset(sa, channel - 1, recompute)
    te_corr = sa.unified_time_eag - off.residual
    eag_c = detrend(sa.eag_filtered[:, channel - 1])
    return pair, sa, te_corr, eag_c, trans, off


def _auto_seed(te_corr, eag_c, fs) -> list:
    """자동 검출 edge → dict 목록 (onset/offset_time·amp)."""
    edges = G.detect_eag_edges(te_corr, eag_c, fs=fs)
    return [{'onset_time': e[0], 'onset_amp': e[1],
             'offset_time': e[3], 'offset_amp': e[4]} for e in edges]


def _current(pair, channel, te_corr, eag_c, fs) -> tuple:
    """현재 유효 edge 목록과 출처. manual 있으면 그것, 없으면 auto seed."""
    man = edge_store.get_channel_edges(pair.subject_name, pair.session_name, channel)
    if man is not None:
        return man, 'manual'
    return _auto_seed(te_corr, eag_c, fs), 'auto'


def _amp_at(te_corr, eag_c, t) -> float:
    return float(eag_c[int(np.argmin(np.abs(te_corr - t)))])


def _snap_corner(te_corr, eag_c, t, lo=None, hi=None, half=0.3) -> tuple:
    """[lo,hi](없으면 t±half)에서 곡률(2차미분) 최대 지점(corner)으로 스냅 → (time, amp)."""
    a = t - half if lo is None else max(t - half, lo)
    b = t + half if hi is None else min(t + half, hi)
    idx = np.where((te_corr >= a) & (te_corr <= b))[0]
    if len(idx) < 5:
        j = int(np.argmin(np.abs(te_corr - t)))
        return float(te_corr[j]), float(eag_c[j])
    seg = eag_c[idx]
    d2 = np.abs(np.gradient(np.gradient(seg)))
    j = idx[int(np.argmax(d2))]
    return float(te_corr[j]), float(eag_c[j])


def _mk_edge(te_corr, eag_c, onset, offset, snap):
    if snap:
        mid = (onset + offset) / 2.0  # onset은 앞 절반, offset은 뒤 절반에서만 스냅 (붕괴 방지)
        on_t, on_a = _snap_corner(te_corr, eag_c, onset, hi=mid)
        off_t, off_a = _snap_corner(te_corr, eag_c, offset, lo=mid)
    else:
        on_t, on_a = onset, _amp_at(te_corr, eag_c, onset)
        off_t, off_a = offset, _amp_at(te_corr, eag_c, offset)
    return {'onset_time': on_t, 'onset_amp': on_a,
            'offset_time': off_t, 'offset_amp': off_a}


# ==================== 시각화 ====================

def _plot(pair, channel, te_corr, eag_c, edges, trans, off, source):
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [2.2, 1.0]})
    # EAG + 번호라벨 edge
    a1.plot(te_corr, eag_c, color='#1f77b4', lw=1.0)
    for tr in trans:
        for ax in (a0, a1):
            ax.axvline(tr.time, color='green', ls='--', alpha=0.4, lw=1.0)
    for i, e in enumerate(edges):
        amp = e['offset_amp'] - e['onset_amp']
        col = '#d62728' if amp > 0 else '#2ca02c'
        a1.plot([e['onset_time'], e['offset_time']], [e['onset_amp'], e['offset_amp']],
                'o-', color=col, ms=7, lw=2.3, zorder=6)
        a1.annotate(f"[{i}]", (e['onset_time'], e['onset_amp']),
                    fontsize=9, color='black', ha='right', va='bottom', zorder=7)
    a1.set_ylabel(f'EAG Ch{channel} detrended (uV)')
    a1.set_xlabel('time (s, corrected)')
    a1.grid(alpha=0.3)
    a1.set_xlim(te_corr[0], te_corr[-1])
    # GRF
    a0.plot(_GRF_T, _GRF_SIGNED, color='green', lw=1.0)
    a0.axhline(0, color='gray', lw=0.5)
    a0.set_ylabel('GRF signed imbalance')
    a0.set_title(f'{pair.subject_name}/{pair.session_name} Ch{channel}  '
                 f'[{source}]  edges={len(edges)} knees={2*len(edges)}  '
                 f'offset corr={off.corrected_offset:+.3f}s ({off.method})  '
                 f'| rise=red fall=green, [n]=edge id, GRF전이=초록선')
    a0.grid(alpha=0.3)
    plt.tight_layout()
    out = EDIT_DIR / f'{pair.subject_name}_{pair.session_name}_ch{channel}_edit.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# GRF 컨텍스트 전역 (플롯 단순화용)
_SA = None
_GRF_T = None
_GRF_SIGNED = None


def _set_grf_context(sa):
    global _SA, _GRF_T, _GRF_SIGNED
    _SA = sa
    _GRF_T = sa.unified_time_grf
    _GRF_SIGNED = G.signed_imbalance(sa.grf_left, sa.grf_right)


def _print_table(edges):
    print(f"{'id':>3} {'onset':>7} {'offset':>7} {'amp(uV)':>9} {'trans_t':>7} dir")
    for i, e in enumerate(edges):
        amp = e['offset_amp'] - e['onset_amp']
        tt = e['offset_time'] - e['onset_time']
        d = 'rise' if amp > 0 else 'fall'
        print(f"{i:>3} {e['onset_time']:>7.2f} {e['offset_time']:>7.2f} "
              f"{amp:>+9.1f} {tt:>7.2f} {d}")


# ==================== 명령 ====================

def cmd_review(args):
    ch = args.channel
    pair, sa, te_corr, eag_c, trans, off = _load(args.session, ch, not args.no_recompute_offset)
    _set_grf_context(sa)
    edges, source = _current(pair, ch, te_corr, eag_c, sa.eag.sample_rate)
    out = _plot(pair, ch, te_corr, eag_c, edges, trans, off, source)
    print(f"[{source}] {pair.subject_name}/{pair.session_name} Ch{ch}  "
          f"edges={len(edges)}  offset corr={off.corrected_offset:+.3f}")
    _print_table(edges)
    print(f"검토 PNG: {out}")


def _load_edit(args):
    """편집 명령 공통: 현재 edge를 manual로 확정(seed 포함)하고 반환."""
    ch = args.channel
    pair, sa, te_corr, eag_c, trans, off = _load(args.session, ch, not args.no_recompute_offset)
    _set_grf_context(sa)
    edges, source = _current(pair, ch, te_corr, eag_c, sa.eag.sample_rate)
    return pair, sa, te_corr, eag_c, trans, off, ch, list(edges)


def _save(pair, ch, edges, off, note):
    edge_store.set_channel_edges(pair.subject_name, pair.session_name, ch,
                                 edges, offset_used=off.corrected_offset, note=note)


def cmd_add(args):
    pair, sa, te_corr, eag_c, trans, off, ch, edges = _load_edit(args)
    e = _mk_edge(te_corr, eag_c, args.onset, args.offset, args.snap)
    edges.append(e)
    edges.sort(key=lambda x: x['onset_time'])
    _save(pair, ch, edges, off, args.note)
    print(f"✅ add: onset={e['onset_time']:.2f} offset={e['offset_time']:.2f} "
          f"amp={e['offset_amp']-e['onset_amp']:+.1f} (snap={args.snap})")
    out = _plot(pair, ch, te_corr, eag_c, edges, trans, off, 'manual')
    _print_table(edges); print(f"검토 PNG: {out}")


def cmd_delete(args):
    pair, sa, te_corr, eag_c, trans, off, ch, edges = _load_edit(args)
    if not (0 <= args.id < len(edges)):
        print(f"⚠️ id {args.id} 범위 밖 (0..{len(edges)-1})"); return
    rm = edges.pop(args.id)
    _save(pair, ch, edges, off, args.note)
    print(f"✅ delete [{args.id}]: onset={rm['onset_time']:.2f}")
    out = _plot(pair, ch, te_corr, eag_c, edges, trans, off, 'manual')
    _print_table(edges); print(f"검토 PNG: {out}")


def cmd_move(args):
    pair, sa, te_corr, eag_c, trans, off, ch, edges = _load_edit(args)
    if not (0 <= args.id < len(edges)):
        print(f"⚠️ id {args.id} 범위 밖 (0..{len(edges)-1})"); return
    onset = args.onset if args.onset is not None else edges[args.id]['onset_time']
    offset = args.offset if args.offset is not None else edges[args.id]['offset_time']
    edges[args.id] = _mk_edge(te_corr, eag_c, onset, offset, args.snap)
    edges.sort(key=lambda x: x['onset_time'])
    _save(pair, ch, edges, off, args.note)
    print(f"✅ move [{args.id}] → onset={onset:.2f} offset={offset:.2f} (snap={args.snap})")
    out = _plot(pair, ch, te_corr, eag_c, edges, trans, off, 'manual')
    _print_table(edges); print(f"검토 PNG: {out}")


def cmd_reset(args):
    ok = edge_store.clear_channel_edges(
        find_session_pair(args.session).subject_name,
        find_session_pair(args.session).session_name, args.channel)
    print("✅ 자동 검출로 복귀" if ok else "⚠️ 해당 채널 manual edge 없음")


def cmd_list(args):
    rows = edge_store.list_all()
    if not rows:
        print("확정된 manual edge 없음"); return
    for r in rows:
        print(f"  {r['subject']}/{r['session']} ch{r['channel']}: "
              f"{r['n_edges']} edges  {r['note']}")


def main():
    ap = argparse.ArgumentParser(description='EAG edge 수동 검토·수정')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add_common(p, need_ch=True):
        p.add_argument('--session', '-s', required=True, help='세션 디렉터리')
        if need_ch:
            p.add_argument('--channel', '-c', type=int, required=True)
        p.add_argument('--no-recompute-offset', action='store_true')
        p.add_argument('--note', default='manual edit')

    pr = sub.add_parser('review'); add_common(pr); pr.set_defaults(func=cmd_review)
    pa = sub.add_parser('add'); add_common(pa)
    pa.add_argument('--onset', type=float, required=True)
    pa.add_argument('--offset', type=float, required=True)
    pa.add_argument('--snap', action='store_true', help='corner로 스냅')
    pa.set_defaults(func=cmd_add)
    pd = sub.add_parser('delete'); add_common(pd)
    pd.add_argument('--id', type=int, required=True); pd.set_defaults(func=cmd_delete)
    pm = sub.add_parser('move'); add_common(pm)
    pm.add_argument('--id', type=int, required=True)
    pm.add_argument('--onset', type=float, default=None)
    pm.add_argument('--offset', type=float, default=None)
    pm.add_argument('--snap', action='store_true')
    pm.set_defaults(func=cmd_move)
    prs = sub.add_parser('reset'); add_common(prs); prs.set_defaults(func=cmd_reset)
    pl = sub.add_parser('list'); pl.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
