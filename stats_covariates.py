#!/usr/bin/env python3
"""성별·체성분 공변량 분석 — 학위논문 1.2의 사전 계획(성별 subgroup) 이행 + 기전 탐색.

**설계 원칙: 표본을 쪼개지 않는다.**

성별을 다루는 방법이 두 가지다. 표본을 남/여로 나눠 각각 분석하면 n=22/19이 되고,
LOSO 디코더에서는 21명 학습·1명 평가가 되어 성능 차이가 성별 효과인지 표본 크기
효과인지 분리되지 않는다. 대신 성별을 **모형의 항**으로 넣으면 41명 전원과 46,772
관측을 그대로 쓴다. 그래서

  * 통계: LMM에 sex 주효과와 load_pct × sex 상호작용을 넣는다(S1). 피험자 단위
    지표의 성별 비교는 독립 단위가 사람인 지점에서만 한다(S2~S4).
  * ML: 층화 학습을 하지 않는다. 이미 산출된 LOSO 예측의 피험자별 오차가 성별에
    따라 다른지만 본다(S8). 일반화·공정성 점검이며 부분군 모형이 아니다.

**주효과와 상호작용을 반드시 분리한다.** 진폭 이득(µV)은 전극-연골 사이 조직
두께에 좌우되므로 성별 주효과가 있을 것으로 예상된다. 임상적으로 중요한 것은
부하 민감도(µV/%BW)가 다른지, 즉 **상호작용**이다. 전자가 유의해도 후자가 등가면
"교정 상수는 사람마다 다르나 부하-전위 관계의 기울기는 같다"가 되고, 이는 4.3.2의
조건 불변성 논법과 같은 구조다. 그래서 상호작용도 동일한 ±0.20 마진으로 TOST한다.

**교란을 숨기지 않는다.** 이 코호트에서 성별과 체성분은 강하게 얽혀 있다
(체지방률 남 22.4% 대 여 32.5%). 41명으로는 분리할 수 없다. 따라서 성별은 1.2에
사전 명시된 확인적 분석으로, 체성분은 기전 가설 하나(피하조직 감쇠)에 대한
탐색적 후속으로 역할을 나눠 보고한다. 연령은 20~35세로 범위가 좁아 공변량으로
쓰지 않고 한계로 서술한다.

사용:
  python3 stats_covariates.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings('ignore')

INBODY = Path('인바디/인바디정리(26_EAG).xlsx')
OUT_DIR = Path('result/stats')
TOST_MARGIN = 0.20          # stats_grf_eag와 동일한 임상 앵커 마진
SEX = {'남': 'M', '여': 'F'}


def load_inbody() -> pd.DataFrame:
    """인바디 시트를 subject_id 키로 정규화한다.

    manifest의 subject_id는 '<연번>. <성명>' 형식이므로 같은 규칙으로 키를 만든다.
    """
    ib = pd.read_excel(INBODY)
    ib['subject_id'] = ib['연번'].astype(str) + '. ' + ib['성명'].astype(str)
    ib = ib.rename(columns={'성별': 'sex', '연령': 'age', '신장(cm)': 'height',
                            '체중(kg)': 'weight', 'BMI': 'bmi',
                            '골격근량(kg)': 'muscle', '체지방량(kg)': 'fat_mass',
                            '체지방률(%)': 'fat_pct', '체수분(L)': 'water'})
    ib['sex'] = ib['sex'].map(SEX)
    cols = ['subject_id', 'sex', 'age', 'height', 'weight', 'bmi',
            'muscle', 'fat_mass', 'fat_pct', 'water']
    return ib[cols]


def per_subject_metrics(d: pd.DataFrame) -> pd.DataFrame:
    """피험자별 용량-반응 기울기와 평균 진폭. 독립 단위가 사람인 지표."""
    rows = []
    for s, g in d.groupby('subject_id'):
        r = sps.linregress(g['load_pct'], g['abs_amp'])
        rec = {'subject_id': s, 'n': len(g), 'slope': r.slope,
               'intercept': r.intercept, 'amp_mean': g['abs_amp'].mean()}
        # 조건별 기울기 (S3용)
        for c, h in g.groupby('condition'):
            if len(h) >= 20:
                rec[f'slope_{c}'] = sps.linregress(h['load_pct'], h['abs_amp']).slope
        rows.append(rec)
    ps = pd.DataFrame(rows)
    for a, b in (('f', 's'), ('c', 's')):
        ka, kb = f'slope_{a}', f'slope_{b}'
        if ka in ps and kb in ps:
            ps[f'diff_{a}_{b}'] = ps[ka] - ps[kb]
    return ps


def _tost(est, se, margin=TOST_MARGIN):
    return max(sps.norm.sf((est + margin) / se), sps.norm.cdf((est - margin) / se))


def _cmp(a, b, label, unit=''):
    """두 군 비교를 Welch t, Mann-Whitney, Hedges g로 보고한다."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    g *= 1 - 3 / (4 * (na + nb) - 9)                       # Hedges 소표본 보정
    t = sps.ttest_ind(a, b, equal_var=False)
    return {'metric': label, 'unit': unit, 'n_M': na, 'n_F': nb,
            'mean_M': round(a.mean(), 4), 'mean_F': round(b.mean(), 4),
            'sd_M': round(a.std(ddof=1), 4), 'sd_F': round(b.std(ddof=1), 4),
            'diff_M_minus_F': round(a.mean() - b.mean(), 4),
            'hedges_g': round(g, 3), 'p_welch': t.pvalue,
            'p_mannwhitney': sps.mannwhitneyu(a, b).pvalue}


# ==================== S1. LMM ====================

def s1_lmm(d: pd.DataFrame, out) -> None:
    """주 LMM에 sex 주효과와 load_pct × sex를 추가한다. 표본은 쪼개지 않는다."""
    import statsmodels.formula.api as smf
    from stats_grf_eag import fit_lmm
    d = d.dropna(subset=['load_pct', 'sex']).copy()
    d['applying'] = (d['event_kind'] == 'on').astype(int)
    d['male'] = (d['sex'] == 'M').astype(int)
    md = smf.mixedlm("abs_amp ~ load_pct * applying + load_pct * male + C(channel)",
                     d, groups=d['subject_id'], re_formula="~load_pct",
                     vc_formula={'visit': '0 + C(visit_id)'})
    res = fit_lmm(md, maxiter=800, label='S1 sex-adjusted LMM')
    out.append("=== S1. 성별 보정 주 LMM (표본 분할 없음) ===")
    out.append(f"  관측 {len(d)} · 피험자 {d.subject_id.nunique()} "
               f"(M {d[d.male==1].subject_id.nunique()} / "
               f"F {d[d.male==0].subject_id.nunique()}) · 수렴 {res.converged}")
    rows = []
    for k, lab in (('male', 'sex 주효과 (M-F, µV)'),
                   ('load_pct:male', 'load × sex 상호작용 (µV/%BW)'),
                   ('load_pct', 'load 기울기 (해제, µV/%BW)'),
                   ('load_pct:applying', 'load × 적용')):
        if k not in res.params.index:
            continue
        b, se = float(res.params[k]), float(res.bse[k])
        r = {'term': lab, 'est': round(b, 4), 'se': round(se, 4),
             'ci90_lo': round(b - 1.645 * se, 4), 'ci90_hi': round(b + 1.645 * se, 4),
             'p': res.pvalues[k]}
        if k == 'load_pct:male':                    # 상호작용만 등가검정 대상
            p_t = _tost(b, se)
            r.update({'p_tost': p_t, 'equivalent_0.20': bool(p_t < 0.05)})
            r['margin_sweep'] = ';'.join(
                f"{m}:{'Y' if _tost(b, se, m) < 0.05 else 'N'}"
                for m in (0.10, 0.15, 0.20, 0.25, 0.30))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / 'cov_s1_lmm_sex.csv', index=False)
    out.append(t.to_string(index=False))
    (OUT_DIR / 'cov_s1_lmm_sex_summary.txt').write_text(str(res.summary()),
                                                        encoding='utf-8')


# ==================== S2~S4, S8 ====================

def s2_subject_level(ps: pd.DataFrame, out) -> None:
    out.append("\n=== S2. 피험자 단위 지표의 성별 비교 (1.2 사전 계획) ===")
    rows = [_cmp(ps[ps.sex == 'M'][k], ps[ps.sex == 'F'][k], lab, u)
            for k, lab, u in (('slope', '용량-반응 기울기', 'µV/%BW'),
                              ('amp_mean', '평균 진폭', 'µV'),
                              ('intercept', '절편', 'µV'))]
    t = pd.DataFrame(rows); t.to_csv(OUT_DIR / 'cov_s2_subject_by_sex.csv', index=False)
    out.append(t.to_string(index=False))


def s3_condition_by_sex(ps: pd.DataFrame, out) -> None:
    out.append("\n=== S3. 조건 대비의 성별 차이 (탐색적, 군당 약 20명) ===")
    rows = []
    for k, lab in (('diff_f_s', 'Axis A [f-s]'), ('diff_c_s', 'Axis B [c-s]')):
        if k not in ps:
            continue
        rows.append(_cmp(ps[ps.sex == 'M'][k], ps[ps.sex == 'F'][k], lab, 'µV/%BW'))
        for sx in ('M', 'F'):
            v = ps[ps.sex == sx][k].dropna()
            se = v.std(ddof=1) / np.sqrt(len(v))
            p_t = _tost(v.mean(), se)
            rows.append({'metric': f'  {lab} {sx} 단독 등가검정', 'n_M': len(v),
                         'mean_M': round(v.mean(), 4), 'p_welch': np.nan,
                         'p_mannwhitney': p_t,
                         'hedges_g': np.nan, 'unit': f'p_TOST={p_t:.4f}'})
    t = pd.DataFrame(rows); t.to_csv(OUT_DIR / 'cov_s3_condition_by_sex.csv', index=False)
    out.append(t.to_string(index=False))
    out.append("  주의: 군당 약 20명이므로 확인적 해석 불가. 부호와 크기만 참고.")


def s4_task_by_sex(cov: pd.DataFrame, out) -> None:
    out.append("\n=== S4. 체중부하 과제 수행의 성별 차이 ===")
    h = pd.read_csv(OUT_DIR / 'hold_steadiness.csv')
    from stats_grf_eag import load_pooled
    d = load_pooled(False, False, 'b')
    smap = d.drop_duplicates('subject').set_index('subject')['subject_id'].to_dict()
    h = h[h.subject.isin(smap)].copy(); h['subject_id'] = h.subject.map(smap)
    TG = {0: 20.0, 1: 50.0, 2: 80.0}
    h = h[h.cycle_id < 3].copy()
    h['abserr'] = (h.load_pct - h.cycle_id.map(TG)).abs()
    q = h.groupby('subject_id').agg(acc=('abserr', 'mean'), cv=('hold_cv', 'mean'),
                                    sd=('hold_sd', 'mean')).reset_index()
    q = q.merge(cov[['subject_id', 'sex']], on='subject_id')
    rows = [_cmp(q[q.sex == 'M'][k], q[q.sex == 'F'][k], lab, u)
            for k, lab, u in (('acc', '표적 정확도(절대오차)', '%BW'),
                              ('cv', '유지 안정성(CV)', '%'),
                              ('sd', '유지 안정성(SD)', '%BW'))]
    t = pd.DataFrame(rows); t.to_csv(OUT_DIR / 'cov_s4_task_by_sex.csv', index=False)
    out.append(t.to_string(index=False))
    return q


def s8_decoder_by_sex(cov: pd.DataFrame, out) -> None:
    """디코더 오차의 성별 비교. 층화 학습이 아니라 기존 LOSO 예측의 점검."""
    out.append("\n=== S8. 디코더 오차의 성별 비교 (층화 학습 아님) ===")
    rows, keep = [], None
    for f, lab in ((Path('result/ml/loso_per_subject.csv'), 'LightGBM LOSO MAE'),
                   (Path('result/dl/eegnet_load_per_subject.csv'), 'EEGNet LOSO MAE')):
        if not f.exists():
            continue
        m = pd.read_csv(f).merge(cov[['subject_id', 'sex']], on='subject_id')
        rows.append(_cmp(m[m.sex == 'M']['mae'], m[m.sex == 'F']['mae'], lab, '%BW'))
        if keep is None:
            keep = m[['subject_id', 'mae', 'sex']].rename(columns={'mae': 'mae_lgb'})
    t = pd.DataFrame(rows); t.to_csv(OUT_DIR / 'cov_s8_decoder_by_sex.csv', index=False)
    out.append(t.to_string(index=False))
    out.append("  특징 집합에 성별·인구학 변수는 포함되지 않았다(events.csv 확인).")
    return keep


# ==================== B6, B7 ====================

def b_body_composition(ps: pd.DataFrame, mae: pd.DataFrame, out) -> None:
    """체성분과의 연관. 사전 지정 주 가설은 체지방률(피하조직 감쇠)."""
    out.append("\n=== B6/B7. 체성분 연관 (탐색적; 주 가설 = 체지방률) ===")
    rows = []
    targets = [('slope', '기울기'), ('amp_mean', '평균 진폭')]
    if mae is not None:
        ps = ps.merge(mae[['subject_id', 'mae_lgb']], on='subject_id', how='left')
        targets.append(('mae_lgb', '디코더 MAE'))
    for tk, tl in targets:
        for ck, cl, primary in (('fat_pct', '체지방률', True), ('bmi', 'BMI', False),
                                ('muscle', '골격근량', False), ('weight', '체중', False)):
            sub = ps[[tk, ck, 'sex']].dropna()
            if len(sub) < 10:
                continue
            r = sps.spearmanr(sub[tk], sub[ck])
            # 성별을 통제한 부분상관 (성별-체성분 교란 때문에 반드시 병기)
            res_t = sub[tk] - sub.groupby('sex')[tk].transform('mean')
            res_c = sub[ck] - sub.groupby('sex')[ck].transform('mean')
            rp = sps.spearmanr(res_t, res_c)
            rows.append({'target': tl, 'covariate': cl, 'primary': primary,
                         'n': len(sub), 'rho': round(r.statistic, 3),
                         'p': r.pvalue,
                         'rho_sex_adj': round(rp.statistic, 3), 'p_sex_adj': rp.pvalue})
    t = pd.DataFrame(rows); t.to_csv(OUT_DIR / 'cov_b_bodycomp.csv', index=False)
    out.append(t.to_string(index=False))
    out.append("  rho_sex_adj = 성별 내 편차로 계산한 부분상관. 성별과 체성분이"
               " 교란되어 있으므로(체지방률 M 22.4% vs F 32.5%) 원상관 단독 해석 금지.")


def main():
    from stats_grf_eag import load_pooled
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cov = load_inbody()
    d = load_pooled(False, False, 'b').dropna(subset=['load_pct'])
    d = d.merge(cov, on='subject_id', how='left')
    n_miss = d[d.sex.isna()].subject_id.nunique()

    out = [f"인바디 결합: 피험자 {d.subject_id.nunique()} 중 성별 결측 {n_miss}",
           "성별 분포: " + str(cov[cov.subject_id.isin(d.subject_id)]['sex']
                            .value_counts().to_dict())]
    ps = per_subject_metrics(d).merge(cov, on='subject_id', how='left')
    ps.to_csv(OUT_DIR / 'cov_per_subject.csv', index=False)

    s1_lmm(d, out)
    s2_subject_level(ps, out)
    s3_condition_by_sex(ps, out)
    s4_task_by_sex(cov, out)
    mae = s8_decoder_by_sex(cov, out)
    b_body_composition(ps, mae, out)

    txt = "\n".join(str(x) for x in out)
    (OUT_DIR / 'covariates_report.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
