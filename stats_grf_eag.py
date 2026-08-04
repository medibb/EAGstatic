"""
EAG-GRF 단계별 체중부하 통계분석 (dose-response).

STATS_PLAN.md의 절차를 구현. GRF 전이 step 크기(dose) 대비 EAG 반응(amplitude)의
용량-반응, 방향 비대칭(부하 rise vs 이탈 fall), 동시성(latency≈0), 채널 특이성.

입력 : result/phase1_params/grf_triggered_params_*.csv  (+ eag_grf_cross_params_*.csv 품질)
출력 : result/stats/  (pooled csv, 요약 csv, 혼합모형 txt, plot png)

⚠️ manual review(offset/edge 확정) + parameter_extractor 재실행 후에 돌릴 것.
   (grf_step 컬럼과 manual offset/edge가 반영된 최신 파라미터 기준)

실행:
  python3 stats_grf_eag.py                 # PASS 채널, 전체 세션
  python3 stats_grf_eag.py --all-channels  # 품질 필터 없이
  python3 stats_grf_eag.py --exclude-review  # needs_review offset 세션 제외(민감도)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps
try:
    from eag_analyzer import setup_korean_font
    setup_korean_font()
except Exception:
    pass

IN_DIR = Path('result/phase1_params')
OUT_DIR = Path('result/stats')
STEP_BINS = [0, 0.5, 1.0, 1.5, 2.01]
STEP_LABELS = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0']


# ==================== 로드·정제 ====================

def _inputs(stem: str):
    """입력 파일 선택: batch 통합(무접미사) 파일 우선, 없으면 단일 피험자(_이름) glob.

    parameter_extractor.py --batch는 `{stem}.csv`(전체 피험자)를,
    --subject X는 `{stem}_X.csv`를 쓴다. 통합 파일이 있으면 그것만 읽어
    단일 실행 잔재와의 중복/구버전 혼입을 피한다.
    """
    agg = IN_DIR / f'{stem}.csv'
    if agg.exists():
        return [agg]
    return sorted(IN_DIR.glob(f'{stem}_*.csv'))


def attach_cohort(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """manifest.csv를 붙여 사람(subject_id)·조건(condition)·코호트 적격을 채운다.

    파라미터 CSV의 `subject` 컬럼은 사람이 아니라 **방문 폴더명**이다(data_flat이
    방문을 최상위로 펼치기 때문). 한 사람이 1·2차 두 번 방문하므로 이걸 그대로
    군집 단위로 쓰면 같은 사람의 두 방문이 서로 독립 개체로 취급되어 표준오차가
    과소추정된다. manifest에서 사람 이름을 끌어와 `subject_id`로 두고, 방문은
    `visit_id`로 남겨 중첩 랜덤효과 (1|subject/visit)를 쓸 수 있게 한다.

    axis: 'none' 필터 없음 / 'a' s·f 모두 >=min_takes / 'b' s·f·c 모두 >=min_takes
    """
    df['visit_id'] = df['subject']
    if 'condition' not in df.columns:
        df['condition'] = df['session'].astype(str).str[0].str.lower()

    man = Path('data_flat/manifest.csv')
    if not man.exists():
        print("  manifest.csv 없음 → subject_id=visit_id로 대체(build_flat_view.py 먼저 실행)")
        df['subject_id'] = df['visit_id']
        return df

    m = pd.read_csv(man)
    vis = (m[['visit', 'subject', 'subject_no', 'axis_a', 'axis_b']]
           .drop_duplicates('visit')
           .rename(columns={'visit': 'visit_id', 'subject': 'subject_id'}))
    df = df.drop(columns=['subject_no'], errors='ignore').merge(vis, on='visit_id', how='left')

    n_miss = int(df['subject_id'].isna().sum())
    if n_miss:
        print(f"  manifest에 없는 방문 {n_miss}행 → visit_id를 subject_id로 대체")
        df['subject_id'] = df['subject_id'].fillna(df['visit_id'])

    if axis in ('a', 'b'):
        col = f'axis_{axis}'
        n0, v0 = len(df), df['visit_id'].nunique()
        df = df[df[col].fillna(False).astype(bool)].copy()
        print(f"  코호트 axis_{axis} 적용: {n0 - len(df)}행 제외 "
              f"(방문 {v0}→{df['visit_id'].nunique()}, 남은 {len(df)}행)")
    return df


def load_pooled(exclude_review: bool, all_channels: bool,
                axis: str = 'none') -> pd.DataFrame:
    files = _inputs('grf_triggered_params')
    if not files:
        raise FileNotFoundError(f"{IN_DIR}/grf_triggered_params.csv 없음. "
                                "parameter_extractor.py --batch 먼저 실행.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    need = {'matched', 'amplitude', 'grf_step', 'eag_direction', 'channel',
            'latency', 'subject', 'session'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing}. 최신 스키마로 재추출 필요 "
                         "(grf_step 등은 review 후 재실행 때 생성).")

    df = df[df['matched'] == True].copy()

    # 분석 제외 라벨(exclusions.json)이 붙은 세션/채널 제거.
    # 컬럼이 없으면(구 스키마) 건너뛴다 — 재추출하면 생긴다.
    if 'excluded' in df.columns:
        n0 = len(df)
        df = df[~df['excluded'].fillna(False).astype(bool)].copy()
        if n0 != len(df):
            print(f"  분석제외 라벨로 {n0 - len(df)}행 제외 (남은 {len(df)}행)")

    # 방향 오염으로 한쪽만 채택된 행 중 폐기된 쪽 제거
    if 'accepted' in df.columns:
        n0 = len(df)
        df = df[df['accepted'].fillna(True).astype(bool)].copy()
        if n0 != len(df):
            print(f"  방향 불일치로 폐기된 {n0 - len(df)}행 제외 (남은 {len(df)}행)")

    # 품질 병합 (cross_params)
    qfiles = _inputs('eag_grf_cross_params')
    if qfiles and not all_channels:
        q = pd.concat([pd.read_csv(f) for f in qfiles], ignore_index=True)
        if {'channel_quality', 'subject', 'session', 'channel'} <= set(q.columns):
            # 구버전 cross_params는 channel이 0-based라 1-based인 파라미터 CSV와
            # 한 칸 어긋난다(ch8은 짝이 없어 소멸). 재추출 전 파일도 읽히도록 보정.
            if q['channel'].min() == 0:
                print("  cross_params가 0-based → 전극번호(1-based)로 보정")
                q = q.copy()
                q['channel'] = q['channel'] + 1
            q = q[['subject', 'session', 'channel', 'channel_quality',
                   'channel_snr_db']].drop_duplicates(['subject', 'session', 'channel'])
            df = df.merge(q, on=['subject', 'session', 'channel'], how='left')
            df = df[df['channel_quality'] == 'PASS'].copy()

    # needs_review offset 세션 플래그 (worklist)
    wl = Path('result/offset_review/worklist.csv')
    review_keys = set()
    if wl.exists():
        w = pd.read_csv(wl)
        if {'subject', 'session'} <= set(w.columns):
            review_keys = set(zip(w['subject'], w['session']))
    df['offset_review_flag'] = [ (s, ss) in review_keys
                                 for s, ss in zip(df['subject'], df['session']) ]
    if exclude_review:
        df = df[~df['offset_review_flag']].copy()

    df = attach_cohort(df, axis)

    # 파생
    df['abs_amp'] = df['amplitude'].abs()
    df['abs_step'] = df['grf_step'].abs()
    df['step_bin'] = pd.cut(df['abs_step'], STEP_BINS, labels=STEP_LABELS,
                            include_lowest=True)
    df = df[np.isfinite(df['abs_amp']) & np.isfinite(df['abs_step'])].copy()
    return df


# ==================== 분석 ====================

def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(['eag_direction', 'step_bin'], observed=True)
    out = g.agg(n=('abs_amp', 'size'),
                amp_mean=('abs_amp', 'mean'), amp_sd=('abs_amp', 'std'),
                lat_median=('latency', 'median'),
                step_mean=('abs_step', 'mean')).reset_index()
    return out


def dose_response(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, sub in list(df.groupby('eag_direction')) + [('ALL', df)]:
        x, y = sub['abs_step'].values, sub['abs_amp'].values
        if len(x) < 3:
            continue
        sl, ic, r, p, se = sps.linregress(x, y)
        rows.append({'direction': direction, 'n': len(x),
                     'slope_uV_per_step': round(sl, 2), 'intercept': round(ic, 2),
                     'r2': round(r ** 2, 3), 'p': f'{p:.2e}', 'slope_se': round(se, 2)})
    return pd.DataFrame(rows)


def latency_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, sub in list(df.groupby('eag_direction')) + [('ALL', df)]:
        lat = sub['latency'].dropna().values
        if len(lat) < 3:
            continue
        try:
            w, p = sps.wilcoxon(lat)
        except Exception:
            w, p = np.nan, np.nan
        rows.append({'direction': direction, 'n': len(lat),
                     'lat_median': round(float(np.median(lat)), 3),
                     'lat_iqr': round(float(np.subtract(*np.percentile(lat, [75, 25]))), 3),
                     'wilcoxon_vs0_p': f'{p:.2e}' if np.isfinite(p) else 'NA'})
    return pd.DataFrame(rows)


def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('channel')
    out = g.agg(n=('abs_amp', 'size'), amp_mean=('abs_amp', 'mean'),
                amp_sd=('abs_amp', 'std'), lat_median=('latency', 'median')).reset_index()
    # 채널별 dose 기울기
    slopes = []
    for ch, sub in g:
        if len(sub) >= 3:
            sl = sps.linregress(sub['abs_step'], sub['abs_amp']).slope
        else:
            sl = np.nan
        slopes.append(round(sl, 2))
    out['dose_slope'] = slopes
    # Kruskal-Wallis (채널간 |amp| 차이)
    groups = [s['abs_amp'].values for _, s in g if len(s) >= 3]
    if len(groups) >= 2:
        h, p = sps.kruskal(*groups)
        out.attrs['kruskal_H'] = round(float(h), 2)
        out.attrs['kruskal_p'] = float(p)
    return out


def mixed_model(df: pd.DataFrame, out_path: Path):
    try:
        import statsmodels.formula.api as smf
    except Exception:
        out_path.write_text("statsmodels 미설치 → subject별 OLS 기울기 집단검정으로 대체.\n"
                            + _per_subject_slope_test(df), encoding='utf-8')
        return
    d = df.copy()
    d['direction'] = (d['eag_direction'] == 'rise').astype(int)  # rise=1
    # 군집은 사람(subject_id). 같은 사람의 방문은 그 안에 중첩시킨다 —
    # 방문을 최상위 군집으로 두면 1·2차 방문이 독립 개체로 취급된다.
    grp = 'subject_id' if 'subject_id' in d.columns else 'subject'
    vc = {'visit': '0 + C(visit_id)'} if 'visit_id' in d.columns else None
    try:
        md = smf.mixedlm("abs_amp ~ abs_step * direction + C(channel)", d,
                         groups=d[grp], re_formula="~1", vc_formula=vc)
        res = md.fit(method='lbfgs', maxiter=200)
        head = (f"군집: {grp} (n={d[grp].nunique()})"
                + (f" / 중첩: visit_id (n={d['visit_id'].nunique()})\n\n" if vc else "\n\n"))
        out_path.write_text(head + str(res.summary()), encoding='utf-8')
    except Exception as e:
        out_path.write_text(f"MixedLM 실패({e}) → 대체.\n" + _per_subject_slope_test(df),
                           encoding='utf-8')


def _per_subject_slope_test(df: pd.DataFrame) -> str:
    slopes = []
    key = 'subject_id' if 'subject_id' in df.columns else 'subject'
    for subj, s in df.groupby(key):
        if len(s) >= 5:
            slopes.append(sps.linregress(s['abs_step'], s['abs_amp']).slope)
    slopes = np.array(slopes)
    if len(slopes) < 3:
        return "피험자 수 부족."
    t, p = sps.ttest_1samp(slopes, 0)
    return (f"subject별 dose 기울기 (n_subj={len(slopes)}): "
            f"mean={slopes.mean():.2f} µV/step, sd={slopes.std():.2f}, "
            f"1-sample t vs 0: t={t:.2f}, p={p:.2e}")


# ==================== plot ====================

def plots(df: pd.DataFrame):
    # dose-response 산점 + 방향별 적합
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'rise': '#d62728', 'fall': '#2ca02c'}
    for direction, sub in df.groupby('eag_direction'):
        ax.scatter(sub['abs_step'], sub['abs_amp'], s=8, alpha=0.25,
                   color=colors.get(direction, 'gray'), label=f'{direction} (n={len(sub)})')
        if len(sub) >= 3:
            sl, ic, r, p, _ = sps.linregress(sub['abs_step'], sub['abs_amp'])
            xs = np.linspace(sub['abs_step'].min(), sub['abs_step'].max(), 50)
            ax.plot(xs, sl * xs + ic, color=colors.get(direction, 'gray'), lw=2,
                    label=f'{direction} fit: {sl:.0f}uV/step, R2={r**2:.2f}')
    ax.set_xlabel('|GRF step| (load dose)'); ax.set_ylabel('|EAG amplitude| (uV)')
    ax.set_title('Dose-response: EAG response vs graded load step')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'dose_response.png', dpi=120); plt.close(fig)

    # latency 히스토그램
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['latency'].dropna(), bins=60, color='#1f77b4', alpha=0.8)
    ax.axvline(0, color='red', ls='--', label='0 (simultaneous)')
    ax.axvline(df['latency'].median(), color='k', ls=':',
               label=f"median={df['latency'].median():.3f}s")
    ax.set_xlabel('latency = EAG onset - GRF transition (s)'); ax.set_ylabel('count')
    ax.set_title('EAG-GRF simultaneity'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'latency_hist.png', dpi=120); plt.close(fig)

    # 채널별 box
    fig, ax = plt.subplots(figsize=(9, 5))
    chs = sorted(df['channel'].unique())
    ax.boxplot([df[df['channel'] == c]['abs_amp'].values for c in chs],
               showfliers=False)
    ax.set_xticks(range(1, len(chs) + 1))
    ax.set_xticklabels(chs)
    ax.set_xlabel('EAG channel'); ax.set_ylabel('|EAG amplitude| (uV)')
    ax.set_title('Per-channel response'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'channel_box.png', dpi=120); plt.close(fig)


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description='EAG-GRF 단계별 부하 dose-response 통계')
    ap.add_argument('--all-channels', action='store_true', help='품질(PASS) 필터 끔')
    ap.add_argument('--exclude-review', action='store_true',
                    help='needs_review offset 세션 제외(민감도)')
    ap.add_argument('--axis', choices=['none', 'a', 'b'], default='b',
                    help="코호트 필터: a=s·f 충족, b=s·f·c 충족(기본), none=필터 없음")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pooled(args.exclude_review, args.all_channels, args.axis)
    print(f"관측(matched): {len(df)}  | 피험자 {df['subject_id'].nunique()}명 "
          f"· 방문 {df['visit_id'].nunique()} · 테이크 "
          f"{df.groupby(['visit_id','session']).ngroups}  "
          f"| axis={args.axis} review제외={args.exclude_review} "
          f"PASS만={not args.all_channels}")

    df.to_csv(OUT_DIR / 'grf_eag_pooled.csv', index=False)
    desc = descriptive(df); desc.to_csv(OUT_DIR / 'descriptive_by_bin.csv', index=False)
    dr = dose_response(df); dr.to_csv(OUT_DIR / 'dose_response_by_direction.csv', index=False)
    lat = latency_summary(df); lat.to_csv(OUT_DIR / 'latency_summary.csv', index=False)
    chs = channel_summary(df); chs.to_csv(OUT_DIR / 'channel_summary.csv', index=False)
    mixed_model(df, OUT_DIR / 'mixedmodel_summary.txt')
    plots(df)

    print("\n=== Dose-response (|EAG amp| ~ |GRF step|) ===")
    print(dr.to_string(index=False))
    print("\n=== Latency (동시성 vs 0) ===")
    print(lat.to_string(index=False))
    print(f"\n=== 채널 (Kruskal p={chs.attrs.get('kruskal_p', float('nan')):.2e}) ===")
    print(chs.to_string(index=False))
    print(f"\n산출물: {OUT_DIR}/  (pooled/descriptive/dose_response/latency/channel csv, "
          "mixedmodel_summary.txt, dose_response.png, latency_hist.png, channel_box.png)")


if __name__ == '__main__':
    main()
