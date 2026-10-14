"""매칭 창 -0.5 vs -0.8 **짝비교** — 같은 코드·같은 주석 상태에서 창만 바꿔 비교한다.

`tools/compare_lat08.py`는 저장된 추출 파일 두 개를 비교하는데, 기준선
(`result/_pre_lat08/`)이 **2026-08-11** 파일이다. 그 이후 연구원이 offset 900여 건을
확정했고 제외 라벨도 늘었다. 그래서 그 비교는 창 변경 효과와 "주석이 늘어난 효과"를
섞어 보여준다 (실제로 행 수까지 78,720 → 78,752로 달라졌다 — 창 변경으로는 행이 늘 수
없다). 손실 175건·귀속변경 12,000건은 대부분 그 혼입이다.

여기서는 **한 번 로드한 데이터에 대해 `extract_responses`를 두 번** 호출한다
(lat_lo=-0.5, lat_lo=-0.8). 코드·주석·offset이 완전히 동일하므로 차이는 창에서만 온다.
`parameter_extractor.run_batch`의 GRF-triggered 블록을 그대로 따른다.

    python3 tools/window_apply_check.py                 # 전체 (약 30분)
    python3 tools/window_apply_check.py --limit 20      # 빠른 확인
"""

import argparse
import contextlib
import csv
import io
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

OUT = REPO / 'result' / 'window_apply' / 'paired.csv'
FIELDS = ['subject', 'session', 'channel', 'quality', 'trans_id', 'protocol_ok',
          'a_matched', 'a_onset', 'a_lat', 'a_amp',
          'b_matched', 'b_onset', 'b_lat', 'b_amp']


@contextlib.contextmanager
def quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def cohort() -> set:
    tail = re.compile(r'\((?:분석안됨|분석어려움)\)\s*$')
    p = REPO / 'result' / 'cohort_manual.csv'
    with open(p, encoding='utf-8-sig') as f:
        return {tail.sub('', r['visit']).strip() for r in csv.DictReader(f)
                if str(r['include']).strip().lower() in ('true', '1', 'yes')}


def run(limit: int):
    from sync_analyzer import find_session_pair, SyncAnalyzer
    from eag_analyzer import get_data_dir
    from parameter_extractor import evaluate_channel_quality, EEG_CHANNELS
    import grf_triggered_annotator as G

    keep = cohort()
    base = Path(get_data_dir())
    trials = []
    for v in sorted(os.listdir(base)):
        if v not in keep or not (base / v).is_dir():
            continue
        for t in sorted(os.listdir(base / v)):
            if (base / v / t).is_dir():
                trials.append(str(base / v / t))
    if limit:
        trials = trials[:limit]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    f = open(OUT, 'w', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()

    t0, n_ok, n_fail = time.time(), 0, 0
    for i, sdir in enumerate(trials, 1):
        try:
            pair = find_session_pair(sdir)
            if pair is None:
                n_fail += 1
                continue
            with quiet():
                sa = SyncAnalyzer(pair)
                off, trans, _s, _g = G.compute_offset(sa, 0, True)
                q = evaluate_channel_quality(sa.eag)
                # parameter_extractor.run_batch 와 동일한 anchor 구성
                te_corr = sa.unified_time_eag
                signed = G.signed_imbalance(sa.grf_left, sa.grf_right)
                _rest, cycles, _ci = G.detect_load_cycles_expected(
                    sa.unified_time_grf, signed, sa.grf_left, sa.grf_right)
                anchors = (G.cycles_to_transitions(cycles, trans)
                           if len(cycles) == G.EXPECTED_CYCLES else trans)
                chans = list(range(1, EEG_CHANNELS + 1))
                ra = G.extract_responses(sa, anchors, te_corr, chans,
                                         lat_lo=-0.5, lat_hi=1.5)   # 변경 전
                rb = G.extract_responses(sa, anchors, te_corr, chans)  # 현재(-0.8)
        except Exception as e:
            n_fail += 1
            print(f'  실패 {sdir.split("/")[-1]}: {type(e).__name__}: {e}', flush=True)
            continue

        ok = len(cycles) == G.EXPECTED_CYCLES
        for a, b in zip(ra, rb):
            assert a.channel == b.channel and a.trans_id == b.trans_id
            w.writerow({
                'subject': pair.subject_name, 'session': pair.session_name,
                'channel': a.channel,
                'quality': q.get(a.channel - 1, {}).get('flag', 'UNKNOWN'),
                'trans_id': a.trans_id, 'protocol_ok': int(ok),
                'a_matched': int(a.matched),
                'a_onset': '' if not a.matched else round(float(a.onset_time), 4),
                'a_lat': '' if not a.matched else round(float(a.latency), 4),
                'a_amp': '' if not a.matched else round(float(a.amplitude), 2),
                'b_matched': int(b.matched),
                'b_onset': '' if not b.matched else round(float(b.onset_time), 4),
                'b_lat': '' if not b.matched else round(float(b.latency), 4),
                'b_amp': '' if not b.matched else round(float(b.amplitude), 2),
            })
        n_ok += 1
        if i % 25 == 0:
            el = time.time() - t0
            print(f'[{i}/{len(trials)}] 성공 {n_ok} 실패 {n_fail} · '
                  f'{i/el:.2f} trial/s · 남은 약 {(len(trials)-i)/max(1e-9,i/el)/60:.0f}분',
                  flush=True)
        f.flush()
    f.close()
    print(f'\n완료: trial {n_ok}개 (실패 {n_fail}) → {OUT}')


def report(pass_only=True):
    d = pd.read_csv(OUT)
    print(f'anchor·채널 {len(d)}행 · 방문 {d.subject.nunique()} · '
          f'trial {d.groupby(["subject","session"]).ngroups}')
    if pass_only:
        n0 = len(d)
        d = d[d.quality == 'PASS']
        print(f'품질 PASS만: {n0} → {len(d)}행')
    a, b = d.a_matched.astype(bool), d.b_matched.astype(bool)
    print(f'\n=== 매칭 ===')
    print(f'  -0.5 (변경 전): {int(a.sum())} / {len(d)} ({a.mean()*100:.2f}%)')
    print(f'  -0.8 (현재)   : {int(b.sum())} / {len(d)} ({b.mean()*100:.2f}%)')
    rec = int((~a & b).sum())
    lost = int((a & ~b).sum())
    chg = int((a & b & (np.abs(d.a_onset - d.b_onset) > 1e-6)).sum())
    print(f'\n  회수 (미매칭→매칭) : {rec}')
    print(f'  손실 (매칭→미매칭) : {lost}        ← 0이어야 한다')
    print(f'  귀속 변경          : {chg}')
    if chg:
        c = d[a & b & (np.abs(d.a_onset - d.b_onset) > 1e-6)]
        print(f'    |지연| 줄어든 것 : {int((c.b_lat.abs() < c.a_lat.abs()).sum())}/{len(c)}')
        print(f'    지연 중위 {c.a_lat.median():+.3f} → {c.b_lat.median():+.3f} s')
        print(f'    |진폭| 중위 {c.a_amp.abs().median():.1f} → {c.b_amp.abs().median():.1f} µV')
    print(f'\n=== latency (§4: 0에 가까워야 한다) ===')
    for lab, col, m in (('-0.5', 'a_lat', a), ('-0.8', 'b_lat', b)):
        s = d.loc[m, col]
        print(f'  {lab}: 중위 {s.median():+.4f}  평균 {s.mean():+.4f}  '
              f'[5%..95%] [{s.quantile(.05):+.3f} .. {s.quantile(.95):+.3f}]')
    r = d[~a & b]
    if len(r):
        print(f'  회수된 것만: 중위 {r.b_lat.median():+.4f}  n={len(r)}')
    print(f'\n=== 진폭 ===')
    print(f'  -0.5: |amp| 중위 {d.loc[a,"a_amp"].abs().median():.1f} µV')
    print(f'  -0.8: |amp| 중위 {d.loc[b,"b_amp"].abs().median():.1f} µV')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='매칭 창 짝비교 (-0.5 vs -0.8)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--all-quality', action='store_true')
    a = ap.parse_args()
    if not a.report:
        run(a.limit)
    report(pass_only=not a.all_quality)
