"""창 하한 변경(-0.5 → -0.8) 전후 비교 — 재추출 결과가 예측과 맞는지 확인한다.

`tools/window_sensitivity.py`가 예측한 것: 매칭 +941, 손실 0, 기존 매칭 116건 귀속 변경,
회수되는 띠의 dose 기울기가 주 반응과 같음. 재추출 결과가 그 예측을 재현해야 한다.
재현되지 않으면 바꾼 곳이 의도한 경로가 아니거나 다른 것이 섞여 들어간 것이다.

`REANALYSIS.md §4`의 검증 항목도 같이 본다 — 이벤트 수가 늘었는가, dose-response
기울기의 방향·규모가 유지되는가.

    python3 tools/compare_lat08.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy.stats as sps

BEFORE = REPO / 'result' / '_pre_lat08' / 'phase1_params' / 'grf_triggered_params.csv'
AFTER = REPO / 'result' / 'phase1_params' / 'grf_triggered_params.csv'
KEY = ['subject', 'session', 'channel', 'trans_id']


def load(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False)


def desc(s: pd.Series, label: str, unit: str = ''):
    s = s.dropna()
    if not len(s):
        print(f'  {label:26s} (없음)')
        return
    print(f'  {label:26s} n={len(s):6d}  중위 {s.median():+8.4f}{unit}  '
          f'평균 {s.mean():+8.4f}  [5%..95%] [{s.quantile(.05):+.3f} .. {s.quantile(.95):+.3f}]')


def main():
    if not BEFORE.exists():
        raise SystemExit(f'변경 전 파일 없음: {BEFORE}')
    if not AFTER.exists():
        raise SystemExit(f'재추출 결과 없음: {AFTER} — parameter_extractor.py --batch 먼저')

    b, a = load(BEFORE), load(AFTER)
    print(f'변경 전 {len(b)}행 / 변경 후 {len(a)}행')

    print('\n=== 1. 매칭 수 ===')
    for name, d in (('변경 전', b), ('변경 후', a)):
        m = d['matched'].astype(bool)
        print(f'  {name}: matched {int(m.sum())} / {len(d)} ({m.mean()*100:.2f}%)')
    print(f"  증가: {int(a['matched'].astype(bool).sum() - b['matched'].astype(bool).sum()):+d}"
          f"  (예측 +941, PASS 기준)")

    # 같은 anchor를 키로 맞춰 귀속 변화를 센다
    if set(KEY) <= set(b.columns) and set(KEY) <= set(a.columns):
        bb = b.set_index(KEY)
        aa = a.set_index(KEY)
        common = bb.index.intersection(aa.index)
        bb, aa = bb.loc[common], aa.loc[common]
        mb = bb['matched'].astype(bool)
        ma = aa['matched'].astype(bool)
        print(f'\n  공통 anchor {len(common)}개')
        print(f'    회수 (미매칭 → 매칭): {int((~mb & ma).sum())}')
        print(f'    손실 (매칭 → 미매칭): {int((mb & ~ma).sum())}   ← 0이어야 한다')
        both = mb & ma
        chg = both & (np.abs(bb['onset_time'] - aa['onset_time']) > 1e-6)
        print(f'    둘 다 매칭인데 다른 edge: {int(chg.sum())}   (예측 116)')
        if chg.sum():
            dl = (aa.loc[chg, 'latency'].abs() - bb.loc[chg, 'latency'].abs())
            print(f'      그중 |지연|이 줄어든 것: {int((dl<0).sum())}/{int(chg.sum())}')
            print(f'      진폭 변화 중위: {(aa.loc[chg,"amplitude"].abs() - bb.loc[chg,"amplitude"].abs()).median():+.1f} µV')

    print('\n=== 2. latency 분포 (ANNOTATION_PROTOCOL §4: 0에 가까워야 한다) ===')
    desc(b[b.matched.astype(bool)]['latency'], '변경 전', ' s')
    desc(a[a.matched.astype(bool)]['latency'], '변경 후', ' s')
    if set(KEY) <= set(a.columns):
        rec = aa[(~mb & ma)]
        desc(rec['latency'], '└ 회수된 것만', ' s')

    print('\n=== 3. 주 지표 (REANALYSIS §4-3: 방향·규모가 유지되는가) ===')
    for name, d in (('변경 전', b), ('변경 후', a)):
        m = d[d.matched.astype(bool)].copy()
        print(f'  {name}:')
        print(f'    |amplitude| 중위 {m.amplitude.abs().median():7.1f} µV   '
              f'transition_time 중위 {m.transition_time.median():.3f} s')
        if 'grf_step' in m.columns:
            x, y = m['grf_step'].abs(), m['amplitude'].abs()
            ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 100:
                lr = sps.linregress(x[ok], y[ok])
                print(f'    dose: |amp| ~ |grf_step|  기울기 {lr.slope:+.2f}  '
                      f'r²={lr.rvalue**2:.3f}  n={int(ok.sum())}')

    print('\n=== 4. 세션·채널 커버리지 ===')
    for name, d in (('변경 전', b), ('변경 후', a)):
        m = d[d.matched.astype(bool)]
        print(f'  {name}: 방문 {d.subject.nunique()} · trial·채널 '
              f'{d.groupby(["subject","session","channel"]).ngroups} · '
              f'매칭 있는 trial·채널 {m.groupby(["subject","session","channel"]).ngroups}')


if __name__ == '__main__':
    main()
