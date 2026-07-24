"""
EAG-GRF Statistical Analysis Module

Phase 1/2/3 CSV 결과를 읽어 통계 분석 수행:
1. 세션 타입별 비교 (s/f/c) — Repeated Measures ANOVA
2. 채널별 APA 분석 — 발생률, latency 비교
3. Test-retest 신뢰도 — ICC(2,1)
4. Effect size — partial η², Cohen's d

Usage:
    python3 stats_analyzer.py                    # 전체 분석
    python3 stats_analyzer.py --quality-filter   # PASS 채널만 사용
    python3 stats_analyzer.py --analysis anova   # 특정 분석만
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pingouin as pg
from scipy import stats
from pathlib import Path


RESULT_DIR = Path(os.path.dirname(__file__)) / 'result'
STATS_DIR = RESULT_DIR / 'statistics'


def load_data(quality_filter: bool = False):
    """Phase 1/2/3 CSV를 로드하고 세션 타입을 추출."""
    data = {}

    # Phase 1
    data['session'] = pd.read_csv(RESULT_DIR / 'phase1_params' / 'session_params.csv')
    data['cross'] = pd.read_csv(RESULT_DIR / 'phase1_params' / 'eag_grf_cross_params.csv')
    data['grf_event'] = pd.read_csv(RESULT_DIR / 'phase1_params' / 'grf_event_params.csv')

    # Phase 2
    p2_dir = RESULT_DIR / 'phase2_params'
    if (p2_dir / 'eag_channel_params.csv').exists():
        data['eag_ch'] = pd.read_csv(p2_dir / 'eag_channel_params.csv')
    if (p2_dir / 'apa_params.csv').exists():
        data['apa'] = pd.read_csv(p2_dir / 'apa_params.csv')
    if (p2_dir / 'coactivation_params.csv').exists():
        data['coact'] = pd.read_csv(p2_dir / 'coactivation_params.csv')

    # Phase 3
    p3_dir = RESULT_DIR / 'phase3_params'
    if (p3_dir / 'coherence.csv').exists():
        data['coherence'] = pd.read_csv(p3_dir / 'coherence.csv')

    # 세션 타입 추출 (s, f, c — trial 세션 제외)
    for key in data:
        df = data[key]
        if 'session' in df.columns:
            df['session_type'] = df['session'].str.extract(r'^([a-zA-Z]+)')
            # trial 세션(st, ft, ct) 제외 → s, f, c만
            df = df[df['session_type'].isin(['s', 'f', 'c'])].copy()
            data[key] = df

    # quality_filter 적용
    if quality_filter:
        for key in ['cross', 'eag_ch', 'apa']:
            if key in data and 'channel_quality' in data[key].columns:
                before = len(data[key])
                data[key] = data[key][data[key]['channel_quality'] == 'PASS'].copy()
                after = len(data[key])
                print(f"  Quality filter [{key}]: {before} → {after} rows "
                      f"({before - after} removed)")

    return data


# ==================== 1. Session Type Comparison (RM ANOVA) ====================

def run_session_type_anova(data: dict, save_dir: Path):
    """세션 타입(s/f/c)별 주요 파라미터 RM ANOVA."""
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # --- Session-level parameters ---
    session_df = data['session'].copy()

    # 피험자별 세션타입 평균 (반복측정 구조)
    params = ['symmetry_index', 'sway_variability', 'event_frequency',
              'mean_time_to_peak', 'mean_recovery_time']

    for param in params:
        agg = session_df.groupby(['subject', 'session_type'])[param].mean().reset_index()
        agg = agg.dropna(subset=[param])

        # 3그룹 모두 있는 피험자만
        valid_subjects = agg.groupby('subject')['session_type'].nunique()
        valid_subjects = valid_subjects[valid_subjects == 3].index
        agg = agg[agg['subject'].isin(valid_subjects)]

        if len(valid_subjects) < 3:
            results.append({'parameter': param, 'test': 'RM ANOVA',
                            'F': np.nan, 'p': np.nan, 'eta2': np.nan,
                            'note': f'insufficient subjects ({len(valid_subjects)})'})
            continue

        try:
            aov = pg.rm_anova(data=agg, dv=param, within='session_type',
                              subject='subject', detailed=True)
            F_val = aov['F'].iloc[0]
            p_val = aov['p_unc'].iloc[0]
            eta2 = aov['ng2'].iloc[0]

            result = {
                'parameter': param,
                'test': 'RM ANOVA',
                'F': round(F_val, 3),
                'p': round(p_val, 4),
                'eta2': round(eta2, 3),
                'n_subjects': len(valid_subjects),
                'significance': '***' if p_val < 0.001 else '**' if p_val < 0.01
                                else '*' if p_val < 0.05 else 'ns',
            }

            # Post-hoc (Bonferroni)
            if p_val < 0.05:
                posthoc = pg.pairwise_tests(data=agg, dv=param,
                                            within='session_type',
                                            subject='subject',
                                            padjust='bonf')
                posthoc_str = '; '.join(
                    f"{r['A']}-{r['B']}: p={r['p_corr']:.4f}"
                    for _, r in posthoc.iterrows())
                result['posthoc'] = posthoc_str

            results.append(result)
        except Exception as e:
            results.append({'parameter': param, 'test': 'RM ANOVA',
                            'F': np.nan, 'p': np.nan, 'eta2': np.nan,
                            'note': str(e)})

    # --- Cross params (FSI latency, MI latency) by session type ---
    if 'cross' in data:
        cross_df = data['cross'].copy()
        cross_params = ['fsi_latency', 'mi_latency']

        for param in cross_params:
            agg = cross_df.groupby(['subject', 'session_type'])[param].mean().reset_index()
            agg = agg.dropna(subset=[param])
            valid_subjects = agg.groupby('subject')['session_type'].nunique()
            valid_subjects = valid_subjects[valid_subjects == 3].index
            agg = agg[agg['subject'].isin(valid_subjects)]

            if len(valid_subjects) < 3:
                results.append({'parameter': param, 'test': 'RM ANOVA',
                                'F': np.nan, 'p': np.nan, 'eta2': np.nan,
                                'note': f'insufficient subjects ({len(valid_subjects)})'})
                continue

            try:
                aov = pg.rm_anova(data=agg, dv=param, within='session_type',
                                  subject='subject', detailed=True)
                result = {
                    'parameter': param,
                    'test': 'RM ANOVA',
                    'F': round(aov['F'].iloc[0], 3),
                    'p': round(aov['p_unc'].iloc[0], 4),
                    'eta2': round(aov['ng2'].iloc[0], 3),
                    'n_subjects': len(valid_subjects),
                    'significance': '***' if aov['p_unc'].iloc[0] < 0.001
                                    else '**' if aov['p_unc'].iloc[0] < 0.01
                                    else '*' if aov['p_unc'].iloc[0] < 0.05 else 'ns',
                }
                if aov['p_unc'].iloc[0] < 0.05:
                    posthoc = pg.pairwise_tests(data=agg, dv=param,
                                                within='session_type',
                                                subject='subject',
                                                padjust='bonf')
                    result['posthoc'] = '; '.join(
                        f"{r['A']}-{r['B']}: p={r['p_corr']:.4f}"
                        for _, r in posthoc.iterrows())
                results.append(result)
            except Exception as e:
                results.append({'parameter': param, 'test': 'RM ANOVA',
                                'note': str(e)})

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / 'rm_anova_session_type.csv', index=False)
    print(f"\n=== RM ANOVA (세션 타입별) ===")
    show_cols = [c for c in ['parameter', 'F', 'p', 'eta2', 'significance'] if c in results_df.columns]
    print(results_df[show_cols].to_string(index=False))

    # 시각화: 세션 타입별 boxplot
    _plot_session_type_boxplots(data, params, save_dir)

    return results_df


def _plot_session_type_boxplots(data, params, save_dir):
    """세션 타입별 파라미터 boxplot."""
    session_df = data['session'].copy()
    session_df = session_df[session_df['session_type'].isin(['s', 'f', 'c'])]

    n_params = len(params)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        agg = session_df.groupby(['subject', 'session_type'])[param].mean().reset_index()
        agg = agg.dropna(subset=[param])

        bp_data = [agg[agg['session_type'] == t][param].values for t in ['s', 'f', 'c']]
        bp = ax.boxplot(bp_data, labels=['Side (s)', 'Front (f)', 'Crutch (c)'],
                        patch_artist=True)
        colors = ['#2196F3', '#FF9800', '#4CAF50']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.set_title(param, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Session Type Comparison (subject-averaged)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'session_type_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==================== 2. Channel-wise APA Analysis ====================

def run_apa_analysis(data: dict, save_dir: Path):
    """채널별 APA 발생률 및 latency 분석."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if 'apa' not in data:
        print("APA 데이터 없음 (Phase 2 미실행)")
        return None

    apa_df = data['apa'].copy()
    apa_df = apa_df[apa_df['session_type'].isin(['s', 'f', 'c'])]

    # 채널별 APA 발생률
    apa_rate = apa_df.groupby(['channel', 'session_type'])['has_apa'].mean().reset_index()
    apa_rate.columns = ['channel', 'session_type', 'apa_rate']

    # 채널별 APA latency (has_apa == True만)
    apa_latency = apa_df[apa_df['has_apa'] == True].groupby(
        ['channel', 'session_type'])['apa_latency'].agg(['mean', 'std', 'count']).reset_index()

    # Pivot table
    rate_pivot = apa_rate.pivot(index='channel', columns='session_type', values='apa_rate')
    rate_pivot.to_csv(save_dir / 'apa_rate_by_channel.csv')

    print(f"\n=== APA 발생률 (채널 × 세션타입) ===")
    print(rate_pivot.round(3).to_string())

    # 채널별 RM ANOVA (APA latency ~ session_type)
    results = []
    apa_true = apa_df[apa_df['has_apa'] == True].copy()
    for ch in range(8):
        ch_data = apa_true[apa_true['channel'] == ch]
        agg = ch_data.groupby(['subject', 'session_type'])['apa_latency'].mean().reset_index()
        agg = agg.dropna()
        valid = agg.groupby('subject')['session_type'].nunique()
        valid = valid[valid == 3].index
        agg = agg[agg['subject'].isin(valid)]

        if len(valid) < 3:
            results.append({'channel': f'Ch{ch+1}', 'F': np.nan, 'p': np.nan, 'eta2': np.nan})
            continue

        try:
            aov = pg.rm_anova(data=agg, dv='apa_latency', within='session_type',
                              subject='subject', detailed=True)
            results.append({
                'channel': f'Ch{ch+1}',
                'F': round(aov['F'].iloc[0], 3),
                'p': round(aov['p_unc'].iloc[0], 4),
                'eta2': round(aov['ng2'].iloc[0], 3),
            })
        except Exception:
            results.append({'channel': f'Ch{ch+1}', 'F': np.nan, 'p': np.nan, 'eta2': np.nan})

    apa_anova_df = pd.DataFrame(results)
    apa_anova_df.to_csv(save_dir / 'apa_latency_anova_by_channel.csv', index=False)

    print(f"\n=== APA Latency RM ANOVA (채널별) ===")
    print(apa_anova_df.to_string(index=False))

    # 시각화: 채널별 APA rate heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    rate_matrix = rate_pivot.values
    im = ax.imshow(rate_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(rate_pivot.columns)))
    ax.set_xticklabels(rate_pivot.columns)
    ax.set_yticks(range(len(rate_pivot.index)))
    ax.set_yticklabels([f'Ch{i+1}' for i in rate_pivot.index])
    ax.set_xlabel('Session Type')
    ax.set_ylabel('Channel')
    ax.set_title('APA Detection Rate by Channel × Session Type')
    plt.colorbar(im, label='APA Rate')
    for i in range(rate_matrix.shape[0]):
        for j in range(rate_matrix.shape[1]):
            ax.text(j, i, f'{rate_matrix[i, j]:.2f}', ha='center', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / 'apa_rate_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return apa_rate


# ==================== 3. Test-Retest Reliability (ICC) ====================

def run_icc_analysis(data: dict, save_dir: Path):
    """1차/2차 측정 간 test-retest 신뢰도 (ICC)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    session_df = data['session'].copy()

    # 피험자명에서 측정 회차 추출: "(02.02_10)주창민_1" → _1 = 1차, _2 = 2차
    session_df['measurement'] = session_df['subject'].str.extract(r'_(\d+)$').astype(float)

    # 2차 측정이 있는 피험자 확인
    has_2nd = session_df[session_df['measurement'] == 2]['subject'].unique()
    if len(has_2nd) == 0:
        print("\n=== ICC 분석 ===")
        print("2차 측정 데이터가 없습니다 (모두 _1 = 1차 측정).")
        print("2차 측정 데이터가 수집되면 ICC 분석이 가능합니다.")

        # ICC 분석 구조만 저장
        pd.DataFrame({
            'parameter': ['symmetry_index', 'sway_variability', 'event_frequency',
                           'mean_time_to_peak', 'mean_recovery_time'],
            'icc': [np.nan] * 5,
            'note': ['No 2nd measurement data'] * 5,
        }).to_csv(save_dir / 'icc_test_retest.csv', index=False)
        return None

    # 피험자 기본명 추출 (회차 제거)
    session_df['subject_base'] = session_df['subject'].str.replace(r'_\d+$', '', regex=True)

    params = ['symmetry_index', 'sway_variability', 'event_frequency',
              'mean_time_to_peak', 'mean_recovery_time']

    results = []
    for param in params:
        # 세션타입별로 평균 → 피험자별 1개 값
        agg = session_df.groupby(['subject_base', 'measurement'])[param].mean().reset_index()
        agg = agg.dropna()

        # 1차와 2차 모두 있는 피험자
        both = agg.groupby('subject_base')['measurement'].nunique()
        both = both[both == 2].index
        agg = agg[agg['subject_base'].isin(both)]

        if len(both) < 3:
            results.append({'parameter': param, 'icc': np.nan,
                            'note': f'insufficient paired subjects ({len(both)})'})
            continue

        try:
            icc_result = pg.intraclass_corr(
                data=agg, targets='subject_base', raters='measurement',
                ratings=param)
            # ICC(2,1) — Two-way random, single measures, absolute agreement
            icc_21 = icc_result[icc_result['Type'] == 'ICC2'].iloc[0]
            results.append({
                'parameter': param,
                'icc': round(icc_21['ICC'], 3),
                'ci95_low': round(icc_21['CI95%'][0], 3),
                'ci95_high': round(icc_21['CI95%'][1], 3),
                'p': round(icc_21['pval'], 4),
                'n_pairs': len(both),
                'interpretation': _icc_interpretation(icc_21['ICC']),
            })
        except Exception as e:
            results.append({'parameter': param, 'icc': np.nan, 'note': str(e)})

    icc_df = pd.DataFrame(results)
    icc_df.to_csv(save_dir / 'icc_test_retest.csv', index=False)

    print(f"\n=== ICC Test-Retest Reliability ===")
    print(icc_df.to_string(index=False))

    return icc_df


def _icc_interpretation(icc_val):
    """ICC 해석 기준 (Koo & Li, 2016)."""
    if np.isnan(icc_val):
        return '-'
    if icc_val < 0.50:
        return 'poor'
    elif icc_val < 0.75:
        return 'moderate'
    elif icc_val < 0.90:
        return 'good'
    else:
        return 'excellent'


# ==================== 4. Descriptive Statistics ====================

def run_descriptive(data: dict, save_dir: Path):
    """기술통계: 세션 타입별 mean ± SD 테이블."""
    save_dir.mkdir(parents=True, exist_ok=True)

    session_df = data['session'].copy()
    session_df = session_df[session_df['session_type'].isin(['s', 'f', 'c'])]

    params = ['symmetry_index', 'sway_variability', 'event_frequency',
              'mean_duration', 'mean_time_to_peak', 'mean_recovery_time']

    rows = []
    for param in params:
        for stype in ['s', 'f', 'c']:
            vals = session_df[session_df['session_type'] == stype][param].dropna()
            rows.append({
                'parameter': param,
                'session_type': stype,
                'n': len(vals),
                'mean': round(vals.mean(), 4),
                'sd': round(vals.std(), 4),
                'median': round(vals.median(), 4),
                'min': round(vals.min(), 4),
                'max': round(vals.max(), 4),
            })

    desc_df = pd.DataFrame(rows)
    desc_df.to_csv(save_dir / 'descriptive_session_type.csv', index=False)

    # Cross params
    if 'cross' in data:
        cross_df = data['cross'].copy()
        cross_df = cross_df[cross_df['session_type'].isin(['s', 'f', 'c'])]

        cross_rows = []
        for param in ['fsi_latency', 'mi_latency', 'fsi_amplitude', 'mi_amplitude']:
            for stype in ['s', 'f', 'c']:
                vals = cross_df[cross_df['session_type'] == stype][param].dropna()
                cross_rows.append({
                    'parameter': param,
                    'session_type': stype,
                    'n': len(vals),
                    'mean': round(vals.mean(), 4),
                    'sd': round(vals.std(), 4),
                    'median': round(vals.median(), 4),
                })
        cross_desc_df = pd.DataFrame(cross_rows)
        cross_desc_df.to_csv(save_dir / 'descriptive_cross_params.csv', index=False)

    print(f"\n=== Descriptive Statistics (세션 타입별) ===")
    pivot = desc_df.pivot_table(index='parameter', columns='session_type',
                                values=['mean', 'sd'])
    print(pivot.round(4).to_string())

    return desc_df


# ==================== 5. Coherence Comparison ====================

def run_coherence_comparison(data: dict, save_dir: Path):
    """세션 타입별 × 채널별 coherence 비교."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if 'coherence' not in data:
        print("\nCoherence 데이터 없음 (Phase 3 미실행)")
        return None

    coh_df = data['coherence'].copy()
    coh_df = coh_df[coh_df['session_type'].isin(['s', 'f', 'c'])]

    # 채널별 × 세션타입별 RM ANOVA (mean_coh_0_2hz)
    results = []
    for ch in coh_df['channel'].unique():
        ch_data = coh_df[coh_df['channel'] == ch]
        agg = ch_data.groupby(['subject', 'session_type'])['mean_coh_0_2hz'].mean().reset_index()
        agg = agg.dropna()
        valid = agg.groupby('subject')['session_type'].nunique()
        valid = valid[valid == 3].index
        agg = agg[agg['subject'].isin(valid)]

        if len(valid) < 3:
            results.append({'channel': f'Ch{ch}', 'F': np.nan, 'p': np.nan, 'eta2': np.nan})
            continue

        try:
            aov = pg.rm_anova(data=agg, dv='mean_coh_0_2hz', within='session_type',
                              subject='subject', detailed=True)
            results.append({
                'channel': f'Ch{ch}',
                'F': round(aov['F'].iloc[0], 3),
                'p': round(aov['p_unc'].iloc[0], 4),
                'eta2': round(aov['ng2'].iloc[0], 3),
                'significance': '***' if aov['p_unc'].iloc[0] < 0.001
                                else '**' if aov['p_unc'].iloc[0] < 0.01
                                else '*' if aov['p_unc'].iloc[0] < 0.05 else 'ns',
            })
        except Exception:
            results.append({'channel': f'Ch{ch}', 'F': np.nan, 'p': np.nan, 'eta2': np.nan})

    coh_anova_df = pd.DataFrame(results)
    coh_anova_df.to_csv(save_dir / 'coherence_anova_by_channel.csv', index=False)

    print(f"\n=== Coherence (0-2Hz) RM ANOVA (채널별) ===")
    print(coh_anova_df.to_string(index=False))

    # 시각화: 채널별 coherence boxplot (session type별)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for idx, ch in enumerate(sorted(coh_df['channel'].unique())[:8]):
        ax = axes[idx // 4, idx % 4]
        ch_data = coh_df[coh_df['channel'] == ch]
        bp_data = [ch_data[ch_data['session_type'] == t]['mean_coh_0_2hz'].dropna().values
                   for t in ['s', 'f', 'c']]
        bp = ax.boxplot(bp_data, labels=['s', 'f', 'c'], patch_artist=True)
        colors = ['#2196F3', '#FF9800', '#4CAF50']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.set_title(f'Ch{ch}', fontweight='bold')
        ax.set_ylabel('Coherence (0-2Hz)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('EAG-GRF Coherence (0-2Hz) by Session Type × Channel',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'coherence_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return coh_anova_df


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='EAG-GRF Statistical Analysis')
    parser.add_argument('--quality-filter', action='store_true',
                        help='PASS 채널만 사용 (quality_flag 기반)')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['all', 'descriptive', 'anova', 'apa', 'icc', 'coherence'],
                        help='실행할 분석 선택')
    args = parser.parse_args()

    print("=" * 60)
    print("EAG-GRF Statistical Analysis")
    print("=" * 60)

    if args.quality_filter:
        print("Quality filter: ON (PASS 채널만 사용)")
    else:
        print("Quality filter: OFF (모든 채널 포함)")

    data = load_data(quality_filter=args.quality_filter)
    print(f"\nLoaded data: {', '.join(f'{k}({len(v)})' for k, v in data.items())}")

    analyses = {
        'descriptive': run_descriptive,
        'anova': run_session_type_anova,
        'apa': run_apa_analysis,
        'icc': run_icc_analysis,
        'coherence': run_coherence_comparison,
    }

    if args.analysis == 'all':
        for name, func in analyses.items():
            try:
                func(data, STATS_DIR)
            except Exception as e:
                print(f"\n[ERROR] {name}: {e}")
    else:
        analyses[args.analysis](data, STATS_DIR)

    print(f"\n{'=' * 60}")
    print(f"결과 저장: {STATS_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
