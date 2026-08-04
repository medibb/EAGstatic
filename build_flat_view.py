#!/usr/bin/env python3
"""data/(4단계 중첩)를 파이프라인이 기대하는 평면 구조로 심볼릭 링크 미러링.

sync_analyzer.find_session_pair()는 OpenBCISession 폴더의 '부모 폴더명'을 subject_name으로
쓴다. 중첩 구조에서는 그 부모가 조건 폴더('1. Side')라 44명이 3개로 뭉개지고
manual_offsets/manual_edges 키(방문 폴더명)와도 어긋난다. 평면 미러를 만들어 부모가
방문 폴더가 되게 한다.

기본 정책: *_Temp(폐기/재측정 테이크) 제외, 연습 세션(st/ft/ct) 제외.
"""
import os, re, csv, argparse, shutil

SRC = "/workspace/research/EAGstatic/data"
DST = "/workspace/research/EAGstatic/data_flat"
COND = {"1. Side": "s", "2. Front": "f", "3. Crutch": "c"}
QC_SUFFIX = re.compile(r'\((?:분석안됨|분석어려움|노이즈|데이터적음|드리프트|drift)[^)]*\)')
# 피험자 폴더는 '<번호>. <이름>' 형식. SRC가 data/ 직속이 되면서 그 외 폴더가
# 섞여 들어올 수 있으므로 형식으로 거른다.
SUBJ_DIR = re.compile(r'^\d+\.')


def strip_qc(name):
    """방문/조건 폴더명에서 연구원이 붙인 QC 접미사를 제거하고 플래그를 돌려준다."""
    flags = QC_SUFFIX.findall(name)
    return QC_SUFFIX.sub('', name).strip(), flags


def annotate_cohort(manifest, min_takes=3):
    """방문×조건별 분석가능 테이크 수를 세어 축별 적격 여부를 각 행에 붙인다.

    포함 기준은 방문 단위다. 한 방문에서 어떤 조건의 테이크가 min_takes 미만이면
    그 조건의 dose-response를 추정할 수 없으므로, 그 조건이 필요한 축에서 방문째 뺀다.
    축을 나누는 이유는 c(목발) 결손 방문을 Axis A(s vs f)에서까지 버릴 이유가 없기 때문이다.

      axis_a  s,f >= min_takes          (부하 방향 대비)
      axis_b  s,f,c 모두 >= min_takes   (목발 부분체중부하 포함)
    """
    n = {}
    for r in manifest:
        if r['analysable']:
            n.setdefault(r['visit'], {}).setdefault(r['condition'], 0)
            n[r['visit']][r['condition']] += 1
    for r in manifest:
        c = n.get(r['visit'], {})
        ok = lambda k: c.get(k, 0) >= min_takes
        r['n_takes_cond'] = c.get(r['condition'], 0)
        r['axis_a'] = ok('s') and ok('f')
        r['axis_b'] = r['axis_a'] and ok('c')
    return manifest


def build(include_temp=False, include_practice=False, dry_run=False, min_takes=3):
    if os.path.exists(DST) and not dry_run:
        shutil.rmtree(DST)
    manifest = []
    for subj in sorted(os.listdir(SRC)):
        sp = os.path.join(SRC, subj)
        if not os.path.isdir(sp) or subj.startswith('.') or not SUBJ_DIR.match(subj):
            continue
        subj_clean, subj_flags = strip_qc(subj)
        for visit in sorted(os.listdir(sp)):
            vp = os.path.join(sp, visit)
            if not os.path.isdir(vp) or visit.startswith('.'):
                continue
            visit_clean, visit_flags = strip_qc(visit)
            for cond_dir in sorted(os.listdir(vp)):
                cp = os.path.join(vp, cond_dir)
                if not os.path.isdir(cp) or cond_dir.startswith('.'):
                    continue
                cond_clean, cond_flags = strip_qc(cond_dir)
                cond = COND.get(cond_clean, '?')
                for entry in sorted(os.listdir(cp)):
                    ep = os.path.join(cp, entry)
                    if not os.path.isdir(ep) or entry.startswith('.'):
                        continue
                    if entry.endswith('_Temp'):
                        if not include_temp:
                            continue
                        sessions = [(os.path.join(ep, s), s) for s in sorted(os.listdir(ep))
                                    if os.path.isdir(os.path.join(ep, s)) and not s.startswith('.')]
                        in_temp = True
                    else:
                        sessions, in_temp = [(ep, entry)], False
                    for path, name in sessions:
                        tag = name.rsplit('-', 1)[-1]
                        if not include_practice and re.fullmatch(r'[sfc]t\d*', tag):
                            continue
                        files = os.listdir(path)
                        has_eag = any(f.startswith('BrainFlow-RAW') and f.endswith('.csv') for f in files)
                        has_grf = any(f.endswith('.csv') and not f.startswith('BrainFlow') for f in files)
                        link = os.path.join(DST, visit_clean, name)
                        manifest.append(dict(
                            subject=subj_clean, subject_no=subj.split('.')[0], visit=visit_clean,
                            condition=cond, session=tag, in_temp=in_temp,
                            has_eag=has_eag, has_grf=has_grf, analysable=has_eag and has_grf,
                            qc_flags='|'.join(subj_flags + visit_flags + cond_flags),
                            src=path, link=link))
                        if not dry_run:
                            # 세션 폴더는 실제 디렉터리로 만들고 파일만 링크한다.
                            # os.walk()는 심볼릭 링크 디렉터리로 내려가지 않으므로
                            # 폴더째 링크하면 find_all_pairs()가 아무것도 찾지 못한다.
                            os.makedirs(link, exist_ok=True)
                            for f in files:
                                if f.startswith('.'):
                                    continue
                                tgt = os.path.join(link, f)
                                if not os.path.lexists(tgt):
                                    os.symlink(os.path.join(path, f), tgt)
    annotate_cohort(manifest, min_takes)
    if not dry_run:
        with open(os.path.join(DST, 'manifest.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader(); w.writerows(manifest)
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--include-temp', action='store_true', help='*_Temp 폐기 테이크도 포함')
    ap.add_argument('--include-practice', action='store_true', help='연습 세션(st/ft/ct) 포함')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--min-takes', type=int, default=3,
                    help='방문×조건별 최소 분석가능 테이크 수 (기본 3)')
    a = ap.parse_args()
    m = build(a.include_temp, a.include_practice, a.dry_run, a.min_takes)
    from collections import Counter
    print(f"방문 폴더 {len({r['visit'] for r in m})}개 · 피험자 {len({r['subject'] for r in m})}명")
    print(f"세션 {len(m)}개 (분석가능 {sum(r['analysable'] for r in m)}, "
          f"GRF누락 {sum(not r['has_grf'] for r in m)}, EAG누락 {sum(not r['has_eag'] for r in m)})")
    print("조건별:", dict(Counter(r['condition'] for r in m)))
    print("QC 플래그 세션:", sum(1 for r in m if r['qc_flags']))
    for ax in ('axis_a', 'axis_b'):
        sel = [r for r in m if r[ax] and r['analysable']]
        print(f"{ax} (조건당 >={a.min_takes}): 방문 {len({r['visit'] for r in sel})} · "
              f"피험자 {len({r['subject'] for r in sel})}명 · 테이크 {len(sel)}")
    if not a.dry_run:
        print("->", DST)
