#!/usr/bin/env python3
"""
Validation script for event-based synchronization across ALL 86 sessions.

Tests the updated _detect_initial_event_grf() and _detect_initial_event_eag()
from sync_analyzer.py with the new parameters:
  - GRF: body_weight parameter, imb_threshold=0.18, search_sec=10.0, step-on filtering
  - EAG: unchanged (search_sec=5.0)

Outputs a comprehensive summary table with per-subject breakdown,
offset statistics, and failure analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import traceback
from collections import defaultdict

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sync_analyzer import SyncAnalyzer, find_all_pairs
from grf_viewer import load_grf_data, extract_body_weight
from eag_analyzer import EAGAnalyzer, FilterConfig, SAMPLE_RATE


def run_validation():
    """Run validation across all session pairs."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    pairs = find_all_pairs(data_dir)

    print(f"{'='*80}")
    print(f"  EVENT-BASED SYNCHRONIZATION VALIDATION")
    print(f"  Total sessions to test: {len(pairs)}")
    print(f"  GRF params: search_sec=10.0, imb_threshold=0.18, step-on filter ON")
    print(f"  EAG params: search_sec=5.0 (unchanged)")
    print(f"{'='*80}\n")

    # Results storage
    results = []

    for i, pair in enumerate(pairs, 1):
        session_label = f"{pair.subject_name}/{pair.session_name}"
        result = {
            'idx': i,
            'subject': pair.subject_name,
            'session': pair.session_name,
            'label': session_label,
            'grf_onset': None,
            'eag_onset': None,
            'offset': None,
            'body_weight': None,
            'grf_status': 'FAIL',
            'eag_status': 'FAIL',
            'sync_status': 'FAIL',
            'grf_fail_reason': '',
            'eag_fail_reason': '',
            'grf_stepon_skipped': False,
            'grf_max_imbalance': None,
        }

        # ---- Load GRF ----
        try:
            grf_df = load_grf_data(pair.grf_filepath)
            body_weight = extract_body_weight(pair.grf_filepath)
            result['body_weight'] = body_weight
        except Exception as e:
            result['grf_fail_reason'] = f"GRF load error: {str(e)[:80]}"
            results.append(result)
            print(f"  [{i:2d}/{len(pairs)}] {session_label:40s} GRF_LOAD_FAIL")
            continue

        # ---- Detect GRF initial event ----
        try:
            # Run with extra diagnostics: inspect the internals
            grf_onset = SyncAnalyzer._detect_initial_event_grf(
                grf_df, body_weight, search_sec=10.0, imb_threshold=0.18
            )
            result['grf_onset'] = grf_onset
            result['grf_status'] = 'OK'

            # Check if step-on events were skipped by running a quick analysis
            # We look at the raw imbalance data to check for earlier events
            time = grf_df['time'].values
            left = grf_df['left_grf'].values
            right = grf_df['right_grf'].values
            mask = time <= 10.0
            t = time[mask]
            total = left[mask] + right[mask]
            safe_total = np.maximum(total, 0.1)
            imbalance = np.abs(left[mask] - right[mask]) / safe_total
            result['grf_max_imbalance'] = float(np.max(imbalance))

            # Check if there were earlier imbalance events that got skipped
            above = np.where(imbalance > 0.18)[0]
            if len(above) > 0:
                first_above_time = float(t[above[0]])
                if grf_onset > first_above_time + 0.5:
                    # The detected onset is significantly later than the first threshold crossing
                    # This means earlier events were skipped (likely step-on)
                    result['grf_stepon_skipped'] = True

        except ValueError as e:
            result['grf_fail_reason'] = str(e)[:120]
            # Still try to get max imbalance for diagnostics
            try:
                time = grf_df['time'].values
                left = grf_df['left_grf'].values
                right = grf_df['right_grf'].values
                mask = time <= 10.0
                total = left[mask] + right[mask]
                safe_total = np.maximum(total, 0.1)
                imbalance = np.abs(left[mask] - right[mask]) / safe_total
                result['grf_max_imbalance'] = float(np.max(imbalance))
            except Exception:
                pass
        except Exception as e:
            result['grf_fail_reason'] = f"Unexpected: {str(e)[:80]}"

        # ---- Load EAG and detect initial event ----
        try:
            config = FilterConfig()
            config.start_time = 0.0
            eag = EAGAnalyzer(pair.eag_filepath, config=config)
            eag_onset = SyncAnalyzer._detect_initial_event_eag(eag.eeg_data, search_sec=5.0)
            result['eag_onset'] = eag_onset
            result['eag_status'] = 'OK'
        except ValueError as e:
            result['eag_fail_reason'] = str(e)[:120]
        except Exception as e:
            result['eag_fail_reason'] = f"Unexpected: {str(e)[:80]}"

        # ---- Compute offset ----
        if result['grf_status'] == 'OK' and result['eag_status'] == 'OK':
            result['offset'] = result['eag_onset'] - result['grf_onset']
            result['sync_status'] = 'OK'

        # ---- Print per-session result ----
        grf_str = f"{result['grf_onset']:.3f}s" if result['grf_onset'] is not None else "FAIL"
        eag_str = f"{result['eag_onset']:.3f}s" if result['eag_onset'] is not None else "FAIL"
        off_str = f"{result['offset']:+.3f}s" if result['offset'] is not None else "N/A"
        skip_str = " [step-on skipped]" if result['grf_stepon_skipped'] else ""
        status = result['sync_status']

        print(f"  [{i:2d}/{len(pairs)}] {session_label:40s} GRF={grf_str:>8s}  EAG={eag_str:>8s}  "
              f"offset={off_str:>9s}  {status}{skip_str}")

        results.append(result)

    # ==================== SUMMARY ====================
    print_summary(results, len(pairs))


def print_summary(results, total):
    """Print comprehensive summary tables."""
    df = pd.DataFrame(results)

    # ---- Overall statistics ----
    n_grf_ok = (df['grf_status'] == 'OK').sum()
    n_eag_ok = (df['eag_status'] == 'OK').sum()
    n_sync_ok = (df['sync_status'] == 'OK').sum()
    n_stepon = df['grf_stepon_skipped'].sum()

    print(f"\n{'='*80}")
    print(f"  SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"\n  --- Overall Success Rates ---")
    print(f"  Total sessions:          {total}")
    print(f"  GRF detection OK:        {n_grf_ok}/{total} ({100*n_grf_ok/total:.1f}%)")
    print(f"  EAG detection OK:        {n_eag_ok}/{total} ({100*n_eag_ok/total:.1f}%)")
    print(f"  Both OK (sync success):  {n_sync_ok}/{total} ({100*n_sync_ok/total:.1f}%)")
    print(f"  Step-on events skipped:  {n_stepon} sessions")

    # ---- Per-subject breakdown ----
    print(f"\n  --- Per-Subject Breakdown ---")
    print(f"  {'Subject':<35s} {'Total':>5s} {'GRF_OK':>6s} {'EAG_OK':>6s} {'Sync':>5s} {'Skip':>5s} {'Rate':>6s}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")

    subjects = df.groupby('subject')
    for subj_name, subj_df in subjects:
        n_total = len(subj_df)
        n_grf = (subj_df['grf_status'] == 'OK').sum()
        n_eag = (subj_df['eag_status'] == 'OK').sum()
        n_sync = (subj_df['sync_status'] == 'OK').sum()
        n_skip = subj_df['grf_stepon_skipped'].sum()
        rate = 100 * n_sync / n_total if n_total > 0 else 0
        print(f"  {subj_name:<35s} {n_total:>5d} {n_grf:>6d} {n_eag:>6d} {n_sync:>5d} {n_skip:>5d} {rate:>5.1f}%")

    # ---- Offset statistics (for successful syncs) ----
    sync_ok = df[df['sync_status'] == 'OK']
    if len(sync_ok) > 0:
        offsets = sync_ok['offset'].values
        print(f"\n  --- Offset Statistics (n={len(sync_ok)}) ---")
        print(f"  Mean:    {np.mean(offsets):+.3f} s")
        print(f"  Std:     {np.std(offsets):.3f} s")
        print(f"  Median:  {np.median(offsets):+.3f} s")
        print(f"  Min:     {np.min(offsets):+.3f} s")
        print(f"  Max:     {np.max(offsets):+.3f} s")
        print(f"  Range:   {np.max(offsets) - np.min(offsets):.3f} s")

        # Per-subject offset stats
        print(f"\n  --- Per-Subject Offset Stats ---")
        print(f"  {'Subject':<35s} {'N':>3s} {'Mean':>8s} {'Std':>7s} {'Min':>8s} {'Max':>8s}")
        print(f"  {'-'*35} {'-'*3} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

        for subj_name, subj_df in sync_ok.groupby('subject'):
            offs = subj_df['offset'].values
            n = len(offs)
            print(f"  {subj_name:<35s} {n:>3d} {np.mean(offs):>+7.3f} {np.std(offs):>7.3f} "
                  f"{np.min(offs):>+7.3f} {np.max(offs):>+7.3f}")

        # Suspicious offsets (|offset| > 2.0 seconds)
        suspicious = sync_ok[sync_ok['offset'].abs() > 2.0]
        if len(suspicious) > 0:
            print(f"\n  --- SUSPICIOUS OFFSETS (|offset| > 2.0s): {len(suspicious)} sessions ---")
            for _, row in suspicious.iterrows():
                print(f"  !! {row['label']:<40s} offset={row['offset']:+.3f}s "
                      f"GRF={row['grf_onset']:.3f}s  EAG={row['eag_onset']:.3f}s")
        else:
            print(f"\n  --- No suspicious offsets (all |offset| <= 2.0s) ---")

    # ---- Failures detail ----
    grf_failures = df[df['grf_status'] == 'FAIL']
    eag_failures = df[df['eag_status'] == 'FAIL']

    if len(grf_failures) > 0:
        print(f"\n  --- GRF Detection Failures ({len(grf_failures)}) ---")
        for _, row in grf_failures.iterrows():
            max_imb = f"max_imb={row['grf_max_imbalance']:.3f}" if row['grf_max_imbalance'] is not None else ""
            print(f"  FAIL {row['label']:<40s} {max_imb}")
            print(f"       Reason: {row['grf_fail_reason']}")

    if len(eag_failures) > 0:
        print(f"\n  --- EAG Detection Failures ({len(eag_failures)}) ---")
        for _, row in eag_failures.iterrows():
            print(f"  FAIL {row['label']:<40s}")
            print(f"       Reason: {row['eag_fail_reason']}")

    if len(grf_failures) == 0 and len(eag_failures) == 0:
        print(f"\n  --- No failures! All {total} sessions detected successfully ---")

    # ---- Step-on skip detail ----
    stepon_sessions = df[df['grf_stepon_skipped'] == True]
    if len(stepon_sessions) > 0:
        print(f"\n  --- Sessions where step-on events were skipped ({len(stepon_sessions)}) ---")
        for _, row in stepon_sessions.iterrows():
            print(f"  {row['label']:<40s} GRF_onset={row['grf_onset']:.3f}s  "
                  f"BW={row['body_weight']:.1f}kg")

    print(f"\n{'='*80}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_validation()
