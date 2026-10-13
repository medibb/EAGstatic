"""주석 저장소 3종의 건수 점검 — 코드를 고치기 전후로 라벨링이 보존됐는지 확인한다.

GUI나 저장 계층을 손댈 때 가장 무서운 것은 "조용한 유실"이다. 저장은 성공했는데
키 하나가 날아가거나, 스키마를 바꾸면서 옛 엔트리가 읽히지 않게 되는 식이다.
파일 크기만 봐서는 알 수 없고, 사람이 수백 건을 눈으로 셀 수도 없다.

그래서 **건수와 지문(fingerprint)을 찍어두고 비교한다.**

    python3 tools/check_stores.py                      # 현재 상태 출력
    python3 tools/check_stores.py --save               # 기준선으로 저장
    python3 tools/check_stores.py --compare            # 기준선과 비교 (줄어들면 실패)

`--compare`는 건수가 **줄어들면** 종료코드 1로 실패한다. 늘어나는 것은 정상이다
(연구원이 계속 주석하고 있으므로). 분석대상 분모는 `result/cohort_manual.csv`의
include=True 방문으로 센다.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(REPO, 'result', '_store_baseline.json')

# 코호트 표기의 꼬리 괄호 주석. data_flat 폴더명에는 없으므로 떼고 맞춘다.
_TAIL = re.compile(r'\((?:분석안됨|분석어려움)\)\s*$')


def _load(name: str) -> dict:
    p = os.path.join(REPO, 'result', name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _cohort() -> set:
    """분석대상 방문 {폴더명}. 없으면 빈 set (= 분모 없이 전체만 센다)."""
    p = os.path.join(REPO, 'result', 'cohort_manual.csv')
    if not os.path.exists(p):
        return set()
    with open(p, encoding='utf-8-sig') as f:
        return {_TAIL.sub('', r['visit']).strip() for r in csv.DictReader(f)
                if str(r['include']).strip().lower() in ('true', '1', 'yes')}


def _fingerprint(keys) -> str:
    """키 집합의 지문. 건수가 같아도 내용이 뒤바뀌면 달라진다."""
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(('\x1f'.join(str(x) for x in k) + '\x1e').encode('utf-8'))
    return h.hexdigest()[:12]


def snapshot() -> dict:
    cohort = _cohort()
    off, edg, exc = (_load('manual_offsets.json'), _load('manual_edges.json'),
                     _load('exclusions.json'))

    off_keys = {(s, t) for s, ts in off.items() for t in ts}
    # edge는 채널 단위가 작업 단위다 (한 trial에 여러 채널).
    edge_keys = {(s, t, c) for s, ts in edg.items() for t, cs in ts.items() for c in cs}
    n_edges = sum(len(e.get('edges', []))
                  for ts in edg.values() for cs in ts.values() for e in cs.values())
    # channel '0' = trial 전체 제외, 그 외 = 채널만 제외 (exclusion_store 참조)
    exc_trial = {(s, t) for s, ts in exc.items() for t, cs in ts.items() if '0' in cs}
    exc_chan = {(s, t, c) for s, ts in exc.items() for t, cs in ts.items()
                for c in cs if c != '0'}

    return {
        'offset_total': len(off_keys),
        'offset_cohort': len({k for k in off_keys if k[0] in cohort}) if cohort else None,
        'offset_fp': _fingerprint(off_keys),
        'edge_channels': len(edge_keys),
        'edge_points': n_edges,
        'edge_fp': _fingerprint(edge_keys),
        'excl_trial': len(exc_trial),
        'excl_channel': len(exc_chan),
        'excl_fp': _fingerprint(exc_trial | exc_chan),
    }


COUNTS = ('offset_total', 'offset_cohort', 'edge_channels', 'edge_points',
          'excl_trial', 'excl_channel')
LABEL = {'offset_total': 'offset 확정 (전체)',
         'offset_cohort': 'offset 확정 (분석대상)',
         'edge_channels': 'edge 확정 채널',
         'edge_points': 'edge 점 수',
         'excl_trial': '제외 (trial 전체)',
         'excl_channel': '제외 (채널만)'}


def show(s: dict):
    for k in COUNTS:
        v = s[k]
        print(f"  {LABEL[k]:22s} {'-' if v is None else v}")
    print(f"  지문                   offset={s['offset_fp']} "
          f"edge={s['edge_fp']} excl={s['excl_fp']}")


def main():
    ap = argparse.ArgumentParser(description='주석 저장소 건수 점검')
    ap.add_argument('--save', action='store_true', help='현재 상태를 기준선으로 저장')
    ap.add_argument('--compare', action='store_true', help='기준선과 비교 (감소하면 실패)')
    a = ap.parse_args()

    cur = snapshot()
    print('현재:')
    show(cur)

    if a.save:
        with open(BASELINE, 'w', encoding='utf-8') as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
        print(f'\n기준선 저장: {BASELINE}')
        return 0

    if a.compare:
        if not os.path.exists(BASELINE):
            print(f'\n기준선 없음: {BASELINE} — 먼저 --save 를 실행하세요', file=sys.stderr)
            return 2
        with open(BASELINE, encoding='utf-8') as f:
            base = json.load(f)
        print('\n기준선:')
        show(base)
        lost = []
        for k in COUNTS:
            b, c = base.get(k), cur.get(k)
            if b is None or c is None:
                continue
            if c < b:
                lost.append(f'{LABEL[k]}: {b} → {c} ({c - b})')
        if lost:
            print('\n❌ 라벨링이 줄었습니다:', file=sys.stderr)
            for m in lost:
                print(f'  {m}', file=sys.stderr)
            return 1
        grew = [f'{LABEL[k]}: {base[k]} → {cur[k]}' for k in COUNTS
                if base.get(k) is not None and cur.get(k) is not None
                and cur[k] > base[k]]
        print('\n✅ 감소 없음' + (' (증가: ' + ', '.join(grew) + ')' if grew else ' (변화 없음)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
