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
import warnings
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
        res = fit_lmm(md, maxiter=200, label='mixed_model (주 dose-response)')
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


# ==================== load_pct 주 분석 · 조건 대비 ====================

COND_NAME = {'s': 'mediolateral', 'f': 'anteroposterior', 'c': 'crutch'}


def load_dose_response(df: pd.DataFrame) -> pd.DataFrame:
    """주 노출은 실측 체중부하율(load_pct)이다.

    grf_step은 GRF 불균형의 '단계 크기'라 프로토콜 내부 지표에 가깝고,
    4.2.8에서 dose 축으로 규정한 것은 검사측 다리가 실제로 받은 체중 비율이다.
    설계값(20/50/80/100)이 아니라 실측을 쓰는 이유는 실측이 일관되게 낮아
    설계값을 쓰면 노출을 계통적으로 잘못 라벨링하기 때문이다.
    """
    rows = []
    d = df.dropna(subset=['load_pct'])
    for cond, sub in list(d.groupby('condition')) + [('ALL', d)]:
        for direction, s2 in list(sub.groupby('eag_direction')) + [('ALL', sub)]:
            if len(s2) < 3:
                continue
            sl, ic, r, p, se = sps.linregress(s2['load_pct'], s2['abs_amp'])
            rows.append({'condition': cond, 'direction': direction, 'n': len(s2),
                         'slope_uV_per_pctBW': round(sl, 3), 'intercept': round(ic, 2),
                         'r2': round(r ** 2, 3), 'p': f'{p:.2e}', 'slope_se': round(se, 3)})
    return pd.DataFrame(rows)


def per_subject_condition_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """피험자×조건별 load_pct 기울기. 축 대비는 이 요약치를 대응표본으로 검정한다.

    이벤트 단위로 조건을 비교하면 n이 수만이 되어 사소한 차이도 유의해진다.
    독립 표본 단위는 피험자이므로, 피험자 안에서 조건별 기울기를 하나씩 뽑아
    대응 검정하는 쪽이 설계와 일치한다.
    """
    rows = []
    d = df.dropna(subset=['load_pct'])
    for (subj, cond), s in d.groupby(['subject_id', 'condition']):
        if len(s) >= 8 and s['load_pct'].nunique() >= 3:
            lr = sps.linregress(s['load_pct'], s['abs_amp'])
            rows.append({'subject_id': subj, 'condition': cond, 'n': len(s),
                         'slope': lr.slope, 'amp_mean': s['abs_amp'].mean()})
    return pd.DataFrame(rows)


def axis_contrasts(ps: pd.DataFrame) -> pd.DataFrame:
    """Axis A(s vs f)와 Axis B(s·f vs c)를 피험자 대응표본으로 검정."""
    rows = []
    w = ps.pivot_table(index='subject_id', columns='condition',
                       values=['slope', 'amp_mean'])
    for metric in ('slope', 'amp_mean'):
        m = w[metric]
        for label, a, b in [('Axis A: s vs f', 's', 'f'),
                            ('Axis B: s vs c', 's', 'c'),
                            ('Axis B: f vs c', 'f', 'c')]:
            if a not in m or b not in m:
                continue
            pair = m[[a, b]].dropna()
            if len(pair) < 5:
                continue
            diff = pair[a] - pair[b]
            t, p_t = sps.ttest_rel(pair[a], pair[b])
            try:
                _, p_w = sps.wilcoxon(pair[a], pair[b])
            except Exception:
                p_w = float('nan')
            rows.append({'metric': metric, 'contrast': label, 'n_subj': len(pair),
                         f'mean_{a}': round(pair[a].mean(), 3),
                         f'mean_{b}': round(pair[b].mean(), 3),
                         'mean_diff': round(diff.mean(), 3),
                         'sd_diff': round(diff.std(), 3),
                         'cohen_dz': round(diff.mean() / diff.std(), 3) if diff.std() else float('nan'),
                         't': round(float(t), 3), 'p_ttest': f'{p_t:.3e}',
                         'p_wilcoxon': f'{p_w:.3e}'})
    return pd.DataFrame(rows)


# 임상 앵커로 고정한 등가 마진(µV per %BW).
# 기울기 β로 보정한 모델을 실제 기울기 β+Δ인 조건에 쓰면 부하 L에서의 추정오차는
# 약 L·Δ/β다. 전체 기울기 ~1.42에서 Δ=0.20이면 50%BW 판정점에서 7%BW,
# 100%BW에서 14%BW 오차 — 부분체중부하 처방 구간(20/50/80%)을 가르지 않는 크기다.
# 재현성(SEM 0.64, MDC95 1.78) 기반 마진은 기울기 자체보다 커서 쓸 수 없다.
TOST_MARGIN = 0.20
TOST_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """대응표본 TOST. 두 단측검정이 모두 유의해야 등가.

    비유의(p>0.05)는 '차이 없음'의 근거가 아니다. 등가를 주장하려면 차이가
    사전 마진 안에 있음을 적극적으로 보여야 하므로 TOST를 쓴다.
    90% CI가 ±margin 안에 완전히 들어가는 것과 동치다.
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    dfree = n - 1
    t_lo = (m + margin) / se           # H0: diff <= -margin
    t_hi = (m - margin) / se           # H0: diff >= +margin
    p_lo = 1 - sps.t.cdf(t_lo, dfree)
    p_hi = sps.t.cdf(t_hi, dfree)
    p_tost = max(p_lo, p_hi)
    crit = sps.t.ppf(0.95, dfree)      # 90% CI
    lo, hi = m - crit * se, m + crit * se
    return {'n': n, 'mean_diff': round(m, 4), 'ci90_lo': round(lo, 4),
            'ci90_hi': round(hi, 4), 'margin': margin,
            'p_tost': p_tost, 'equivalent': bool(p_tost < 0.05),
            'p_diff': float(sps.ttest_rel(a, b).pvalue)}


def tost_contrasts(ps: pd.DataFrame, margins=None) -> pd.DataFrame:
    """축 대비별 TOST를 여러 마진에 대해 훑는다(마진 민감도 곡선).

    확립된 앵커가 없는 영역에서 마진 하나만 제시하면 그 값 자체가 쟁점이 된다.
    '어느 마진부터 등가가 성립하는가'를 함께 보이는 편이 방어에 유리하다.
    """
    margins = margins or TOST_SWEEP
    w = ps.pivot_table(index='subject_id', columns='condition', values='slope')
    rows = []
    for label, a, b in [('Axis A: s vs f', 's', 'f'),
                        ('Axis B: s vs c', 's', 'c'),
                        ('Axis B: f vs c', 'f', 'c')]:
        if a not in w or b not in w:
            continue
        pair = w[[a, b]].dropna()
        if len(pair) < 5:
            continue
        for mg in margins:
            r = tost_paired(pair[a].values, pair[b].values, mg)
            r.update({'contrast': label, 'primary': mg == TOST_MARGIN})
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out = out[['contrast', 'margin', 'primary', 'n', 'mean_diff',
                   'ci90_lo', 'ci90_hi', 'p_tost', 'equivalent', 'p_diff']]
    return out


def fit_lmm(md, maxiter=400, label=''):
    """수렴할 때까지 최적화기를 바꿔가며 MixedLM을 적합한다.

    lbfgs 단독은 랜덤기울기+분산성분을 함께 추정할 때 분산성분 쪽 곡면이 평평해
    반복을 소진하고 멈추는 경우가 있다(|grad| 수백 대). 고정효과와 그 SE는 이미
    안정 지점에 있어 추정치는 거의 바뀌지 않지만, 경고가 남으면 보고할 때
    '수렴 실패한 모형의 계수'가 되므로 미리 정리한다.

    powell은 미분을 쓰지 않아 평평한 구간을 잘 빠져나오므로 lbfgs 앞에 붙인다.
    """
    last = None
    for method in (['lbfgs'], ['powell', 'lbfgs'], ['nm'], ['cg']):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = md.fit(method=method, maxiter=maxiter)
        except Exception:
            continue
        last = res
        if getattr(res, 'converged', False):
            if method != ['lbfgs']:
                print(f"    [수렴] {label or md.__class__.__name__}: "
                      f"lbfgs 실패 → {'+'.join(method)}로 재적합")
            return res
    if last is not None:
        print(f"    [경고] {label}: 모든 최적화기에서 수렴 실패. 계수 인용 주의")
    return last


def tost_lmm(df: pd.DataFrame) -> pd.DataFrame:
    """LMM 고정효과 상호작용에 대한 TOST — 축 대비의 주 분석.

    셀별 OLS 기울기는 이벤트 8~16개로 추정해 잡음이 크고(ICC 0.465), 그 잡음이
    대비의 SE를 부풀린다. random slope LMM은 개인차를 분산성분으로 흡수하므로
    load_pct:condition 고정효과가 곧 '조건 간 기울기 차이'의 편향 없는 추정치다.
    축소추정(BLUP)을 대비에 직접 쓰면 대비 자체가 0으로 당겨져 등가가 인위적으로
    쉬워지지만, 고정효과는 축소되지 않아 그 편향이 없다.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return pd.DataFrame()
    d = df.dropna(subset=['load_pct']).copy()
    rows = []
    for label, a, b in [('Axis A: s vs f', 's', 'f'),
                        ('Axis B: s vs c', 's', 'c'),
                        ('Axis B: f vs c', 'f', 'c')]:
        sub = d[d['condition'].isin([a, b])].copy()
        if sub['condition'].nunique() < 2:
            continue
        # 기준범주를 a로 고정해 상호작용이 곧 (b - a) 기울기 차가 되게 한다
        sub['cond'] = pd.Categorical(sub['condition'], categories=[a, b])
        try:
            md = smf.mixedlm("abs_amp ~ load_pct * cond + C(channel)", sub,
                             groups=sub['subject_id'], re_formula="~load_pct",
                             vc_formula={'visit': '0 + C(visit_id)'})
            res = fit_lmm(md, maxiter=400, label=f'tost_lmm {label}')
        except Exception as e:
            rows.append({'contrast': label, 'error': str(e)[:80]})
            continue
        key = [k for k in res.params.index if k.startswith('load_pct:cond')]
        if not key:
            continue
        est, se = float(res.params[key[0]]), float(res.bse[key[0]])
        lo, hi = est - 1.645 * se, est + 1.645 * se     # 90% CI
        for mg in TOST_SWEEP:
            p_tost = max(1 - sps.norm.cdf((est + mg) / se),
                         sps.norm.cdf((est - mg) / se))
            # 상호작용은 기준범주 a 대비 b의 기울기 차이 = slope(b) - slope(a).
            # 피험자단위 대비(a - b)와 부호가 반대이므로 라벨에 방향을 박아둔다.
            rows.append({'contrast': f'{label} [{b}-{a}]',
                         'n_subj': sub['subject_id'].nunique(),
                         'n_obs': len(sub), 'slope_diff': round(est, 4),
                         'se': round(se, 4), 'ci90_lo': round(lo, 4),
                         'ci90_hi': round(hi, 4), 'margin': mg,
                         'primary': mg == TOST_MARGIN,
                         'p_tost': p_tost, 'equivalent': bool(p_tost < 0.05),
                         'converged': bool(res.converged)})
    return pd.DataFrame(rows)


def shrunk_condition_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """조건별 random-slope LMM의 BLUP으로 피험자 기울기를 축소추정한다.

    민감도용. 대비에 쓰면 차이가 0으로 당겨져 등가 쪽으로 편향되므로
    주 분석(tost_lmm)의 보조로만 본다. 재현성 지표에도 같은 이유로 쓰지 않는다.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return pd.DataFrame()
    d = df.dropna(subset=['load_pct'])
    rows = []
    for cond, sub in d.groupby('condition'):
        try:
            res = fit_lmm(smf.mixedlm("abs_amp ~ load_pct", sub,
                                      groups=sub['subject_id'], re_formula="~load_pct"),
                          maxiter=400, label=f'shrunk slopes cond={cond}')
        except Exception:
            continue
        fe = float(res.fe_params['load_pct'])
        for subj, re_ in res.random_effects.items():
            rows.append({'subject_id': subj, 'condition': cond,
                         'slope': fe + float(re_.get('load_pct', 0.0))})
    return pd.DataFrame(rows)


# 전극 몽타주 (EAG 연구방법.docx 2.1). 3장(동적연구)과 배치가 다르므로 혼용 금지.
#   내측: 1 MPFL · 2 슬개골 근위 내측(시상면 중심선 인접) · 3 내측 전방 · 4 내측 후방
#   외측: 5 LPFL · 6 슬개골 근위 외측(시상면 중심선 인접) · 7 외측 전방 · 8 외측 후방
#   기준전극 = 슬개골 하방 10 cm 경골, 접지 = 그로부터 5~6 cm 하방
#
# 인접 그래프는 무릎을 한 바퀴 도는 고리다. 2와 6은 시상면 중심선을 사이에 두고
# 전면에서 만나고, 4와 8은 슬와부 뒤에서 만난다. 1(MPFL)·5(LPFL)는 관절선보다
# 근위라 같은 쪽 근위/전방 전극에만 인접한 것으로 둔다.
#   4 — 3 — 2 — 6 — 7 — 8 — 4   (고리)
#           └1┘   └5┘           (슬개대퇴 인대)
CHANNEL_ADJ = {1: [2, 3], 2: [1, 3, 6], 3: [1, 2, 4], 4: [3, 8],
               5: [6, 7], 6: [2, 5, 7], 7: [5, 6, 8], 8: [4, 7]}

# 해석·표기용 채널 라벨
CHANNEL_SITE = {1: 'MPFL (medial)', 2: 'proximal-medial', 3: 'medial-anterior',
                4: 'medial-posterior', 5: 'LPFL (lateral)', 6: 'proximal-lateral',
                7: 'lateral-anterior', 8: 'lateral-posterior'}


def _clusters(sig_channels, adj=CHANNEL_ADJ):
    """유의 채널들을 인접성으로 묶어 연결 성분을 만든다."""
    todo, out = set(sig_channels), []
    while todo:
        seed = todo.pop()
        comp, stack = {seed}, [seed]
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb in todo:
                    todo.discard(nb); comp.add(nb); stack.append(nb)
        out.append(sorted(comp))
    return out


def cluster_permutation(df: pd.DataFrame, metric='slope', n_perm=5000,
                        alpha=0.05, seed=20260804) -> pd.DataFrame:
    """채널×조건 대비의 cluster-based permutation (Maris & Oostenveld).

    채널 8개를 각각 검정하면 다중비교가 되고, Bonferroni는 인접 전극이 같은
    생리 신호를 공유한다는 사실을 무시해 지나치게 보수적이다. 인접 채널에서
    함께 나타나는 효과를 하나의 클러스터로 묶어 검정하면 공간 구조를 살리면서
    family-wise error를 통제할 수 있다.

    대응설계이므로 귀무분포는 피험자별 차이의 부호를 무작위로 뒤집어 만든다.
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=['load_pct'])
    cells = []
    for (subj, cond, ch), g in d.groupby(['subject_id', 'condition', 'channel']):
        if len(g) >= 6 and g['load_pct'].nunique() >= 3:
            val = (sps.linregress(g['load_pct'], g['abs_amp']).slope
                   if metric == 'slope' else g['abs_amp'].mean())
            cells.append({'subject_id': subj, 'condition': cond,
                          'channel': int(ch), 'val': val})
    cell = pd.DataFrame(cells)
    rows = []
    for label, a, b in [('Axis A: s vs f', 's', 'f'),
                        ('Axis B: s vs c', 's', 'c'),
                        ('Axis B: f vs c', 'f', 'c')]:
        w = cell[cell['condition'].isin([a, b])].pivot_table(
            index=['subject_id', 'channel'], columns='condition', values='val')
        if a not in w or b not in w:
            continue
        w = w.dropna().reset_index()
        w['diff'] = w[a] - w[b]
        M = w.pivot(index='subject_id', columns='channel', values='diff').dropna()
        if len(M) < 5 or M.shape[1] < 2:
            continue
        chans = list(M.columns)
        X = M.values
        n = len(X)
        tcrit = sps.t.ppf(1 - alpha / 2, n - 1)

        def cluster_stats(mat):
            t = mat.mean(axis=0) / (mat.std(axis=0, ddof=1) / np.sqrt(len(mat)))
            sig = [chans[i] for i in range(len(chans)) if abs(t[i]) > tcrit]
            best = 0.0
            comps = []
            for comp in _clusters(sig):
                s = float(sum(t[chans.index(c)] for c in comp))
                comps.append((comp, s))
                best = max(best, abs(s))
            return t, comps, best

        t_obs, comps_obs, _ = cluster_stats(X)
        null = np.empty(n_perm)
        for i in range(n_perm):
            flip = rng.choice([-1.0, 1.0], size=(n, 1))
            null[i] = cluster_stats(X * flip)[2]
        for comp, s in comps_obs:
            p = float((null >= abs(s)).mean())
            rows.append({'contrast': f'{label} [{a}-{b}]', 'metric': metric,
                         'cluster': '+'.join(f'Ch{c}' for c in comp),
                         'n_subj': n, 'cluster_t_sum': round(s, 2),
                         'p_cluster': round(p, 4),
                         'significant': bool(p < alpha)})
        if not comps_obs:
            rows.append({'contrast': f'{label} [{a}-{b}]', 'metric': metric,
                         'cluster': '(없음)', 'n_subj': n, 'cluster_t_sum': 0.0,
                         'p_cluster': 1.0, 'significant': False})
    return pd.DataFrame(rows)


def _icc21(M: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single measure.

    방문(=평가 시점)을 무작위 효과로 보고 절대일치를 본다. consistency가 아니라
    agreement를 쓰는 이유는, 1·2차 방문 사이의 계통적 이동(bias)도 재현성 실패로
    세어야 임상에서 같은 값을 읽는다고 말할 수 있기 때문이다.
    """
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(axis=1) - gm) ** 2).sum() / (n - 1)          # 피험자 간
    msc = n * ((M.mean(axis=0) - gm) ** 2).sum() / (k - 1)          # 방문 간
    sse = ((M - M.mean(axis=1, keepdims=True) - M.mean(axis=0) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom else float('nan')


def retest_reliability(df: pd.DataFrame, n_boot=2000, seed=20260804) -> pd.DataFrame:
    """방문 간 재검사 신뢰도. 전체 및 조건별로 ICC(2,1)·SEM·MDC95·Bland-Altman.

    ICC 신뢰구간은 부트스트랩으로 낸다. F분포 근사식도 있으나 셀 수가 조건별로
    30 안팎이라 재표본이 가정에 덜 기댄다.
    """
    rng = np.random.default_rng(seed)
    rows = []
    d = df.dropna(subset=['load_pct'])
    for (s, v, c), g in d.groupby(['subject_id', 'visit_id', 'condition']):
        if len(g) >= 8 and g['load_pct'].nunique() >= 3:
            rows.append({'subject_id': s, 'visit_id': v, 'condition': c,
                         'slope': sps.linregress(g['load_pct'], g['abs_amp']).slope,
                         'amp_mean': g['abs_amp'].mean()})
    r = pd.DataFrame(rows)
    out = []
    for metric in ('slope', 'amp_mean'):
        for cond in ['ALL', 's', 'f', 'c']:
            rr = r if cond == 'ALL' else r[r['condition'] == cond]
            pairs = []
            for (s, c), g in rr.groupby(['subject_id', 'condition']):
                if g['visit_id'].nunique() >= 2:
                    pairs.append(g.sort_values('visit_id')[metric].values[:2])
            if len(pairs) < 5:
                continue
            M = np.array(pairs, dtype=float)
            diff = M[:, 0] - M[:, 1]
            n = len(M)
            sd_d = diff.std(ddof=1)
            sem = sd_d / np.sqrt(2)
            icc = _icc21(M)
            boot = np.array([_icc21(M[rng.integers(0, n, n)]) for _ in range(n_boot)])
            boot = boot[np.isfinite(boot)]
            lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
            # LoA 자체의 불확실성(Bland & Altman 1999)
            se_loa = sd_d * np.sqrt(3 / n)
            out.append({'metric': metric, 'condition': cond, 'n_pairs': n,
                        'icc21': round(icc, 3),
                        'icc_ci_lo': round(float(lo), 3), 'icc_ci_hi': round(float(hi), 3),
                        'bias': round(float(diff.mean()), 3),
                        'loa_lo': round(float(diff.mean() - 1.96 * sd_d), 3),
                        'loa_hi': round(float(diff.mean() + 1.96 * sd_d), 3),
                        'loa_se': round(float(se_loa), 3),
                        'sem': round(float(sem), 3),
                        'mdc95': round(float(1.96 * np.sqrt(2) * sem), 3),
                        'sem_pct': round(float(sem / M.mean() * 100), 1)})
    return pd.DataFrame(out)


def plot_bland_altman(df: pd.DataFrame):
    """방문 1 vs 방문 2 Bland-Altman. 지표별로 한 패널씩."""
    rows = []
    d = df.dropna(subset=['load_pct'])
    for (s, v, c), g in d.groupby(['subject_id', 'visit_id', 'condition']):
        if len(g) >= 8 and g['load_pct'].nunique() >= 3:
            rows.append({'subject_id': s, 'visit_id': v, 'condition': c,
                         'slope': sps.linregress(g['load_pct'], g['abs_amp']).slope,
                         'amp_mean': g['abs_amp'].mean()})
    r = pd.DataFrame(rows)
    if r.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'s': '#1f77b4', 'f': '#ff7f0e', 'c': '#2ca02c'}
    for ax, metric, unit in zip(axes, ('slope', 'amp_mean'),
                                ('uV per %BW', 'uV')):
        pts = []
        for (s, c), g in r.groupby(['subject_id', 'condition']):
            if g['visit_id'].nunique() >= 2:
                v = g.sort_values('visit_id')[metric].values[:2]
                pts.append((v.mean(), v[0] - v[1], c))
        if not pts:
            continue
        mean_v = np.array([p[0] for p in pts])
        diff_v = np.array([p[1] for p in pts])
        for c in ('s', 'f', 'c'):
            m = [i for i, p in enumerate(pts) if p[2] == c]
            if m:
                ax.scatter(mean_v[m], diff_v[m], s=28, alpha=0.7,
                           color=colors.get(c, 'gray'), label=f'{c} (n={len(m)})')
        bias, sd = diff_v.mean(), diff_v.std(ddof=1)
        ax.axhline(bias, color='k', lw=1.5, label=f'bias={bias:.2f}')
        for y, lab in [(bias + 1.96 * sd, '+1.96 SD'), (bias - 1.96 * sd, '-1.96 SD')]:
            ax.axhline(y, color='r', ls='--', lw=1)
            ax.text(ax.get_xlim()[1], y, f' {y:.2f}', va='center', fontsize=8, color='r')
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_xlabel(f'Mean of visit 1 and 2 ({unit})')
        ax.set_ylabel(f'Visit 1 - Visit 2 ({unit})')
        ax.set_title(f'Bland-Altman: {metric}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'bland_altman.png', dpi=120)
    plt.close(fig)


def channel_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """채널×조건 지형도. Axis A의 '방향별 지형 차이'를 조작적으로 보는 표."""
    d = df.dropna(subset=['load_pct'])
    rows = []
    for (ch, cond), s in d.groupby(['channel', 'condition']):
        lr = sps.linregress(s['load_pct'], s['abs_amp']) if len(s) >= 3 else None
        rows.append({'channel': ch, 'condition': cond, 'n': len(s),
                     'amp_mean': round(s['abs_amp'].mean(), 2),
                     'slope_uV_per_pctBW': round(lr.slope, 3) if lr else float('nan')})
    return pd.DataFrame(rows)


def condition_model(df: pd.DataFrame, out_path: Path):
    """조건 모델: |amp| ~ load_pct * condition + C(channel) + channel:condition.

    condition×load 상호작용은 오프로딩 방식에 따라 dose-response 기울기가
    다른지를, channel×condition은 방향 특이적 지형을 검정한다.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        out_path.write_text("statsmodels 미설치.", encoding='utf-8')
        return
    d = df.dropna(subset=['load_pct']).copy()
    d['channel'] = d['channel'].astype(int)
    txt = []
    for label, sub in [('Axis A+B (s/f/c 전체)', d),
                       ('Axis A (s vs f)', d[d['condition'].isin(['s', 'f'])])]:
        if sub['condition'].nunique() < 2:
            continue
        try:
            md = smf.mixedlm("abs_amp ~ load_pct * C(condition) + C(channel) "
                             "+ C(channel):C(condition)", sub,
                             groups=sub['subject_id'], re_formula="~1",
                             vc_formula={'visit': '0 + C(visit_id)'})
            res = fit_lmm(md, maxiter=300, label=f'condition_model {label}')
            txt.append(f"### {label}  (피험자 {sub['subject_id'].nunique()}명 · "
                       f"관측 {len(sub)})\n{res.summary()}\n")
        except Exception as e:
            txt.append(f"### {label} — 적합 실패: {e}\n")
    out_path.write_text('\n'.join(txt), encoding='utf-8')


def plots_condition(df: pd.DataFrame):
    d = df.dropna(subset=['load_pct'])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'s': '#1f77b4', 'f': '#ff7f0e', 'c': '#2ca02c'}
    for cond, sub in d.groupby('condition'):
        b = sub.groupby(pd.cut(sub['load_pct'], [0, 30, 60, 85, 101]),
                        observed=True)['abs_amp'].agg(['mean', 'sem', 'size'])
        x = [i.mid for i in b.index]
        ax.errorbar(x, b['mean'], yerr=b['sem'], marker='o', capsize=3,
                    color=colors.get(cond, 'gray'),
                    label=f"{cond} ({COND_NAME.get(cond, cond)}, n={len(sub)})")
        lr = sps.linregress(sub['load_pct'], sub['abs_amp'])
        xs = np.linspace(5, 100, 30)
        ax.plot(xs, lr.slope * xs + lr.intercept, ls='--', lw=1,
                color=colors.get(cond, 'gray'))
    ax.set_xlabel('Measured load on tested limb (%BW)')
    ax.set_ylabel('|EAG amplitude| (uV)')
    ax.set_title('Dose-response by off-loading condition (matched dose)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'dose_by_condition.png', dpi=120)
    plt.close(fig)


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description='EAG-GRF 단계별 부하 dose-response 통계')
    ap.add_argument('--all-channels', action='store_true', help='품질(PASS) 필터 끔')
    ap.add_argument('--exclude-review', action='store_true',
                    help='needs_review offset 세션 제외(민감도)')
    ap.add_argument('--axis', choices=['none', 'a', 'b'], default='b',
                    help="코호트 필터: a=s·f 충족, b=s·f·c 충족(기본), none=필터 없음")
    ap.add_argument('--tag', default='',
                    help='출력 폴더 접미사. 민감도 비교본을 덮어쓰지 않고 나란히 둘 때 사용')
    args = ap.parse_args()

    global OUT_DIR
    if args.tag:
        OUT_DIR = OUT_DIR.with_name(f'{OUT_DIR.name}_{args.tag}')
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

    # 주 분석: 실측 체중부하율(load_pct) + 조건 대비
    ldr = load_dose_response(df)
    ldr.to_csv(OUT_DIR / 'dose_response_loadpct.csv', index=False)
    ps = per_subject_condition_slopes(df)
    ps.to_csv(OUT_DIR / 'per_subject_condition_slopes.csv', index=False)
    ax_c = axis_contrasts(ps) if len(ps) else pd.DataFrame()
    if len(ax_c):
        ax_c.to_csv(OUT_DIR / 'axis_contrasts.csv', index=False)
    tost = tost_contrasts(ps) if len(ps) else pd.DataFrame()
    if len(tost):
        tost.to_csv(OUT_DIR / 'tost_equivalence.csv', index=False)
    tl = tost_lmm(df)
    if len(tl):
        tl.to_csv(OUT_DIR / 'tost_equivalence_lmm.csv', index=False)
    sh = shrunk_condition_slopes(df)
    tost_sh = tost_contrasts(sh) if len(sh) else pd.DataFrame()
    if len(tost_sh):
        tost_sh.to_csv(OUT_DIR / 'tost_equivalence_shrunk.csv', index=False)
    cperm = pd.concat([cluster_permutation(df, 'slope'),
                       cluster_permutation(df, 'amp_mean')], ignore_index=True)
    if len(cperm):
        cperm.to_csv(OUT_DIR / 'cluster_permutation.csv', index=False)
    rel = retest_reliability(df)
    if len(rel):
        rel.to_csv(OUT_DIR / 'retest_reliability.csv', index=False)
    plot_bland_altman(df)
    chc = channel_by_condition(df)
    chc.to_csv(OUT_DIR / 'channel_by_condition.csv', index=False)
    condition_model(df, OUT_DIR / 'condition_model_summary.txt')
    plots_condition(df)

    n_nodose = int(df['load_pct'].isna().sum())
    print(f"\n=== [주 분석] Dose-response (|EAG amp| ~ load_pct %BW) ===")
    print(f"  load_pct 결측 {n_nodose}행 제외 (protocol_ok=False 세션)")
    print(ldr[ldr['direction'] == 'ALL'].to_string(index=False))
    if len(ax_c):
        print("\n=== 축 대비 (피험자 단위 대응표본) ===")
        print(ax_c.to_string(index=False))
    if len(tost):
        pr = tost[tost['primary']]
        print(f"\n=== TOST 등가성 (주 마진 ±{TOST_MARGIN} µV/%BW) ===")
        print(pr[['contrast', 'n', 'mean_diff', 'ci90_lo', 'ci90_hi',
                  'p_tost', 'equivalent']].to_string(index=False))
        print("\n  마진 민감도 (equivalent 여부):")
        print(tost.pivot(index='contrast', columns='margin',
                         values='equivalent').to_string())
    if len(tl) and 'margin' in tl.columns:
        pr = tl[tl['primary'] == True]
        print(f"\n=== [주] LMM 고정효과 TOST (마진 ±{TOST_MARGIN} µV/%BW) ===")
        print(pr[['contrast', 'n_subj', 'n_obs', 'slope_diff', 'se',
                  'ci90_lo', 'ci90_hi', 'p_tost', 'equivalent']].to_string(index=False))
        print("\n  마진 민감도:")
        print(tl.pivot(index='contrast', columns='margin',
                       values='equivalent').to_string())
    if len(tost_sh):
        print("\n  [민감도] 축소추정 기울기 TOST (등가 쪽 편향 주의):")
        print(tost_sh[tost_sh['primary']][
            ['contrast', 'mean_diff', 'ci90_lo', 'ci90_hi', 'equivalent']
        ].to_string(index=False))
    if len(cperm):
        print("\n=== 채널 cluster-based permutation (FWE 통제) ===")
        print(cperm.to_string(index=False))
    if len(rel):
        print("\n=== 방문 간 재검사 신뢰도 ===")
        print(rel.to_string(index=False))
    print("\n=== 채널 × 조건 (load_pct 기울기) ===")
    pv = chc.pivot(index='channel', columns='condition',
                   values='slope_uV_per_pctBW').round(3)
    pv.insert(0, 'site', [CHANNEL_SITE.get(int(c), '') for c in pv.index])
    print(pv.to_string())

    print("\n=== [보조] Dose-response (|EAG amp| ~ |GRF step|) ===")
    print(dr.to_string(index=False))
    print("\n=== Latency (동시성 vs 0) ===")
    print(lat.to_string(index=False))
    print(f"\n=== 채널 (Kruskal p={chs.attrs.get('kruskal_p', float('nan')):.2e}) ===")
    print(chs.to_string(index=False))
    print(f"\n산출물: {OUT_DIR}/  (pooled/descriptive/dose_response/latency/channel csv, "
          "mixedmodel_summary.txt, dose_response.png, latency_hist.png, channel_box.png)")


if __name__ == '__main__':
    main()
