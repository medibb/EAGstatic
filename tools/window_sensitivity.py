"""매칭 창(lat_lo) 민감도 + 창 밖 edge의 dose 의존성 측정 — 측정만, 아무것도 고치지 않는다.

두 가지를 한 번의 데이터 로드로 같이 센다.

**(1) 창 민감도.** `match_edge`의 창은 현재 `[-0.5, +1.5]`초다. 실측에서 미검출 anchor의
7.1%는 창 **밖**(주로 -0.5~-0.9초)에 방향 규칙을 만족하는 edge를 갖고 있었다. 하한을
넓히면 몇 개가 회수되고, 기존 매칭이 몇 개 바뀌는지(=재추출 시 결과가 달라지는 범위)를
전 세션에서 센다. 2세션 표본에서는 `[-0.8,+1.5]`가 29개를 회수하면서 기존 매칭을 하나도
바꾸지 않았다 — 그 결과가 유지되는지 확인하는 것이 목적이다.

**(2) 창 밖 edge가 주 반응인가 APA인가.** 이게 더 중요하다. `parameter_extractor.py`의
`extract_phase2_apa`는 `pre_window=2.0`초로 "onset 전 변곡점"을 APA로 이미 측정하고
있고(`result/phase2_params/apa_params.csv`), 그 apa_latency 중위는 -0.240초, -0.9~-0.5초
구간에 16.4%(5,120건)가 들어 있다. 즉 창을 넓히면 **같은 현상이 주 dose-response와 APA
양쪽에 집계될 수 있다.**

구분 기준은 **부하 의존성**이다. 주 반응의 진폭은 체중부하율(load_pct)에 비례해야 하고
(Ch.4 §4.2.8의 dose 축), APA는 '움직이려는 준비'이므로 부하 크기에 그렇게 비례할 이유가
없다. 창 밖 edge의 amp~load_pct 기울기를 매칭된 주 반응의 기울기와 비교하면 답이 나온다.

실행:
    python3 tools/window_sensitivity.py --out result/window_sens      # 전체 (약 1시간)
    python3 tools/window_sensitivity.py --limit-visits 2              # 빠른 확인
    python3 tools/window_sensitivity.py --report result/window_sens   # 집계만 다시

중단되면 같은 명령으로 재실행하면 이어서 한다(완료한 trial·채널은 건너뛴다).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

WINDOWS = {'base': (-0.5, 1.5), 'w08': (-0.8, 1.5),
           'w10': (-1.0, 1.5), 'w12': (-1.2, 1.5)}
WANT = {'on': 'fall', 'off': 'rise'}          # EXPECTED_DIR과 같아야 한다
OOW_LO, OOW_HI = -3.0, 3.5                    # '창 밖'을 찾아볼 범위

FIELDS = (['subject', 'session', 'channel', 'quality', 'source', 'cycle', 'kind',
           'load_pct', 'n_edges']
          + [f'{w}_{k}' for w in WINDOWS for k in ('idx', 'lat', 'ncand', 'amp')]
          + ['oow_lat', 'oow_amp', 'oow_noise'])


def _dirn(e) -> str:
    return 'rise' if (e['offset_amp'] - e['onset_amp']) > 0 else 'fall'


def _cands(edges, t, want, lo, hi):
    """창 안에서 방향 규칙을 만족하는 후보 [(지연, 인덱스), …]."""
    return [(e['onset_time'] - t, i) for i, e in enumerate(edges)
            if lo <= e['onset_time'] - t <= hi and _dirn(e) == want]


def quality_map() -> dict:
    """(subject, session, channel) → channel_quality. 본 분석과 같은 PASS 규칙.

    `stats_grf_eag.load_pooled`가 품질을 병합해 오는 그 파일을 그대로 읽는다.
    없으면 빈 dict (모두 'UNKNOWN'으로 기록되고, 집계에서 품질 필터를 못 쓴다).
    """
    p = REPO / 'result' / 'phase1_params' / 'eag_grf_cross_params.csv'
    if not p.exists():
        return {}
    q = pd.read_csv(p, usecols=['subject', 'session', 'channel', 'channel_quality'])
    q = q.drop_duplicates(['subject', 'session', 'channel'])
    return {(r.subject, r.session, int(r.channel)): r.channel_quality
            for r in q.itertuples()}


def cohort_visits() -> set:
    """분석대상 방문. `result/cohort_manual.csv`의 include=True.

    코호트 표기에는 꼬리 괄호 주석('(분석안됨)' 등)이 붙어 있고 data_flat 폴더명에는
    없으므로 떼고 맞춘다.
    """
    import re
    p = REPO / 'result' / 'cohort_manual.csv'
    if not p.exists():
        return set()
    tail = re.compile(r'\((?:분석안됨|분석어려움)\)\s*$')
    with open(p, encoding='utf-8-sig') as f:
        return {tail.sub('', r['visit']).strip() for r in csv.DictReader(f)
                if str(r['include']).strip().lower() in ('true', '1', 'yes')}


def trials(limit_visits: int = 0) -> list:
    """[(visit, trial_tag, dir)] — 분석대상 방문의 모든 trial."""
    import edge_app  # noqa  (get_data_dir 경유로 data_flat을 가리킨다)
    from eag_analyzer import get_data_dir
    base = Path(get_data_dir())
    keep = cohort_visits()
    out = []
    for v in sorted(os.listdir(base)):
        if keep and v not in keep:
            continue
        if not (base / v).is_dir():
            continue
        for t in sorted(os.listdir(base / v)):
            if (base / v / t).is_dir():
                out.append((v, t.rsplit('-', 1)[-1], str(base / v / t)))
    if limit_visits:
        vs = sorted({v for v, _, _ in out})[:limit_visits]
        out = [r for r in out if r[0] in vs]
    return out


def measure(out_dir: Path, limit_visits: int, channels: range):
    import edge_app
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'anchors.csv'
    qmap = quality_map()

    done = set()
    if path.exists():
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                done.add((r['subject'], r['session'], int(r['channel'])))
        print(f'이어서 진행: 이미 완료한 trial·채널 {len(done)}개')

    todo = trials(limit_visits)
    new = not path.exists()
    f = open(path, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    t0 = time.time()
    n_done = n_fail = 0
    total = len(todo) * len(channels)
    for ti, (visit, tag, sdir) in enumerate(todo, 1):
        for ch in channels:
            if (visit, tag, ch) in done:
                continue
            try:
                d = edge_app.build_data(sdir, ch)
            except Exception as e:
                n_fail += 1
                print(f'  실패 {visit}/{tag} ch{ch}: {type(e).__name__}: {e}', flush=True)
                continue
            E = d['edges']
            pct = {c['id']: c['load_pct'] for c in d['cycles']}
            nz = set(d['noise_idx'] or [])
            for a in d['anchors']:
                row = {'subject': d['subject'], 'session': d['session'], 'channel': ch,
                       'quality': qmap.get((d['subject'], d['session'], ch), 'UNKNOWN'),
                       'source': d['source'], 'cycle': a['cycle'], 'kind': a['kind'],
                       'load_pct': pct.get(a['cycle']), 'n_edges': len(E)}
                want = WANT[a['kind']]
                for name, (lo, hi) in WINDOWS.items():
                    c = _cands(E, a['t'], want, lo, hi)
                    row[f'{name}_ncand'] = len(c)
                    if c:
                        lat, i = min(c, key=lambda x: abs(x[0]))
                        row[f'{name}_idx'] = i
                        row[f'{name}_lat'] = round(lat, 4)
                        row[f'{name}_amp'] = round(E[i]['offset_amp'] - E[i]['onset_amp'], 2)
                    else:
                        row[f'{name}_idx'] = -1
                        row[f'{name}_lat'] = ''
                        row[f'{name}_amp'] = ''
                # 현행 창에서 미검출인 anchor의 '창 밖' 최근접 edge (APA 비교용)
                if row['base_idx'] == -1:
                    far = [(e['onset_time'] - a['t'], i) for i, e in enumerate(E)
                           if OOW_LO <= e['onset_time'] - a['t'] <= OOW_HI
                           and _dirn(e) == want
                           and not (WINDOWS['base'][0] <= e['onset_time'] - a['t']
                                    <= WINDOWS['base'][1])]
                    if far:
                        lat, i = min(far, key=lambda x: abs(x[0]))
                        row['oow_lat'] = round(lat, 4)
                        row['oow_amp'] = round(E[i]['offset_amp'] - E[i]['onset_amp'], 2)
                        row['oow_noise'] = int(i in nz)
                w.writerow(row)
            n_done += 1
        f.flush()
        el = time.time() - t0
        rate = n_done / max(1e-9, el)
        print(f'[{ti}/{len(todo)}] {visit}/{tag}  처리 {n_done}  실패 {n_fail}  '
              f'{rate:.2f}/s  남은 약 {(total - n_done - len(done)) / max(1e-9, rate) / 60:.0f}분',
              flush=True)
    f.close()
    print(f'\n완료: trial·채널 {n_done}개 (실패 {n_fail}) → {path}')


# ==================== 집계 ====================

def report(out_dir: Path, pass_only: bool = True):
    path = out_dir / 'anchors.csv'
    if not path.exists():
        raise SystemExit(f'{path} 없음 — 먼저 측정을 실행하세요')
    d = pd.read_csv(path)
    print(f'anchor {len(d)}행 · 방문 {d.subject.nunique()} · trial·채널 '
          f'{d.groupby(["subject","session","channel"]).ngroups}개')
    if pass_only and (d['quality'] == 'PASS').any():
        n0 = len(d)
        d = d[d['quality'] == 'PASS'].copy()
        print(f'품질 PASS만: {n0} → {len(d)}행 '
              f'(본 분석과 같은 분모)')
    n = len(d)

    print('\n=== (1) 창 민감도 ===')
    print(f"{'창':<16}{'매칭':>14}{'회수':>8}{'매칭바뀜':>10}{'손실':>8}{'모호(2+)':>11}")
    for name, (lo, hi) in WINDOWS.items():
        m = (d[f'{name}_idx'] >= 0)
        amb = (d[f'{name}_ncand'] >= 2)
        if name == 'base':
            rec = chg = lost = '-'
        else:
            b = (d['base_idx'] >= 0)
            rec = int((~b & m).sum())
            chg = int((b & m & (d['base_idx'] != d[f'{name}_idx'])).sum())
            lost = int((b & ~m).sum())
        print(f'[{lo:+.1f},{hi:+.1f}]'.ljust(16)
              + f'{int(m.sum()):>7} ({m.mean()*100:4.1f}%)'
              + f'{rec:>8}{chg:>10}{lost:>8}'
              + f'{int(amb.sum()):>6} ({amb.mean()*100:4.1f}%)')

    print('\n=== (2) 창 밖 edge: 주 반응인가 APA인가 ===')
    oow = d[d['oow_lat'].notna()].copy()
    print(f'현행 창에서 미검출: {int((d.base_idx<0).sum())}행 '
          f'({(d.base_idx<0).mean()*100:.1f}%) · 그중 창 밖에 후보 있음: {len(oow)}행 '
          f'({len(oow)/n*100:.1f}%)')
    if len(oow):
        print(f"  지연 중위 {oow.oow_lat.median():+.3f}s  "
              f"[5%..95%] [{oow.oow_lat.quantile(.05):+.3f} .. "
              f"{oow.oow_lat.quantile(.95):+.3f}]  "
              f"음(-)쪽 {int((oow.oow_lat<0).sum())}/{len(oow)}")
        print(f"  노이즈 후보로 분류돼 `노이즈 삭제`에 지워지는 것: "
              f"{int(oow.oow_noise.fillna(0).sum())}행")

    # 부하 의존성 — 주 반응이면 amp가 load_pct에 비례해야 한다
    import scipy.stats as sps

    def slope(x, y, label):
        ok = x.notna() & y.notna()
        if ok.sum() < 20:
            print(f'  {label:28s} n={int(ok.sum())} (부족)')
            return None
        lr = sps.linregress(x[ok], y[ok])
        print(f'  {label:28s} n={int(ok.sum()):6d}  기울기 {lr.slope:+8.3f} µV/%BW  '
              f'r²={lr.rvalue**2:.3f}  p={lr.pvalue:.1e}')
        return lr

    print('\n  진폭 ~ 체중부하율(load_pct) 회귀:')
    base_m = d[d.base_idx >= 0]
    slope(base_m['load_pct'], base_m['base_amp'].abs(), '매칭된 주 반응 (현행 창)')
    slope(oow['load_pct'], oow['oow_amp'].abs(), '창 밖 edge')

    # APA와 진폭 비교
    apa = REPO / 'result' / 'phase2_params' / 'apa_params.csv'
    if apa.exists():
        a = pd.read_csv(apa)
        a = a[a.has_apa == True]
        print(f"\n  진폭 비교 (절대값 중위):")
        print(f"    매칭된 주 반응   {base_m['base_amp'].abs().median():7.1f} µV")
        if len(oow):
            print(f"    창 밖 edge      {oow['oow_amp'].abs().median():7.1f} µV")
        print(f"    기존 APA 측정치  {a['apa_amplitude'].abs().median():7.1f} µV  "
              f"(apa_latency 중위 {a['apa_latency'].median():+.3f}s)")
        band = a[(a.apa_latency >= -0.9) & (a.apa_latency <= -0.5)]
        print(f"    └ 그중 -0.9~-0.5초 띠: {len(band)}행 ({len(band)/len(a)*100:.1f}%), "
              f"진폭 중위 {band['apa_amplitude'].abs().median():.1f} µV")

    # 지연대별 부하 의존성 — 띠마다 성질이 다른지
    if len(oow) > 40:
        print('\n  창 밖 edge를 지연대로 나눈 부하 의존성:')
        for lo, hi in [(-1.2, -0.8), (-0.9, -0.5), (1.5, 3.5)]:
            g = oow[(oow.oow_lat >= lo) & (oow.oow_lat < hi)]
            if len(g) >= 20:
                slope(g['load_pct'], g['oow_amp'].abs(), f'  {lo:+.1f}~{hi:+.1f}s')


def main():
    ap = argparse.ArgumentParser(description='매칭 창 민감도 + 창 밖 edge 성질 (측정만)')
    ap.add_argument('--out', default='result/window_sens')
    ap.add_argument('--limit-visits', type=int, default=0, help='앞의 N개 방문만 (빠른 확인)')
    ap.add_argument('--channels', default='1-8')
    ap.add_argument('--report', metavar='DIR', help='측정 없이 집계만')
    ap.add_argument('--all-quality', action='store_true', help='PASS 필터 없이 집계')
    a = ap.parse_args()

    if a.report:
        return report(Path(a.report), pass_only=not a.all_quality)

    lo, hi = (a.channels.split('-') + [a.channels])[:2]
    measure(Path(a.out), a.limit_visits, range(int(lo), int(hi) + 1))
    print()
    report(Path(a.out), pass_only=not a.all_quality)


if __name__ == '__main__':
    main()
