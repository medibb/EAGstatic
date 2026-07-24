#!/usr/bin/env python3
"""
Check initial 5-second event patterns across ALL sessions.
Evaluates whether event-based synchronization works universally.
"""

import os
import sys
import numpy as np
import pandas as pd
import traceback

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eag_analyzer import EAGAnalyzer, FilterConfig, SAMPLE_RATE, EEG_CHANNELS
from grf_viewer import load_grf_data, find_forceplate_csvs_in_dir
from sync_analyzer import (
    find_all_pairs, find_session_pair, SyncAnalyzer,
)


def check_grf_initial_event(grf_filepath, search_sec=5.0):
    """Check if GRF has an initial weight-shift event in first N seconds.

    Returns dict with:
        found: bool
        onset_time: float or None
        max_imbalance: float
        error: str or None
    """
    result = {
        'found': False,
        'onset_time': None,
        'max_imbalance': 0.0,
        'error': None,
        'details': '',
    }
    try:
        grf_df = load_grf_data(grf_filepath)
        time = grf_df['time'].values
        left = grf_df['left_grf'].values
        right = grf_df['right_grf'].values

        mask = time <= search_sec
        t = time[mask]
        total = left[mask] + right[mask]
        safe_total = np.maximum(total, 0.1)
        imbalance = np.abs(left[mask] - right[mask]) / safe_total

        max_imb = float(np.max(imbalance))
        max_imb_time = float(t[np.argmax(imbalance)])
        result['max_imbalance'] = max_imb

        # Check threshold
        above = np.where(imbalance > 0.30)[0]
        if len(above) > 0:
            result['found'] = True
            result['onset_time'] = float(t[above[0]])
            result['details'] = (
                f"onset={result['onset_time']:.3f}s, "
                f"max_imb={max_imb:.3f} @ {max_imb_time:.3f}s"
            )
        else:
            result['details'] = f"max_imb={max_imb:.3f} @ {max_imb_time:.3f}s (below 0.30 threshold)"

    except Exception as e:
        result['error'] = str(e)
        result['details'] = f"ERROR: {e}"

    return result


def check_eag_initial_event(eag_filepath, search_sec=5.0):
    """Check if EAG has an initial gradient-based activity event in first N seconds.

    Returns dict with:
        found: bool
        onset_time: float or None
        peak_time: float or None
        peak_grad_value: float
        error: str or None
    """
    result = {
        'found': False,
        'onset_time': None,
        'peak_time': None,
        'peak_grad_value': 0.0,
        'error': None,
        'details': '',
    }
    try:
        config = FilterConfig()
        config.start_time = 0.0
        eag = EAGAnalyzer(eag_filepath, config=config)
        eeg_data = eag.eeg_data

        n_search = int(search_sec * SAMPLE_RATE)
        start_idx = 1 if eeg_data[0, 0] == 0 else 0
        end_idx = min(start_idx + n_search, len(eeg_data))
        seg = eeg_data[start_idx:end_idx].astype(float)

        # Sum of |gradient| across all channels
        grad_sum = np.zeros(len(seg) - 1)
        for ch in range(min(EEG_CHANNELS, seg.shape[1])):
            grad_sum += np.abs(np.diff(seg[:, ch]))

        # Smoothing (50 samples = 0.2s window)
        kernel = np.ones(50) / 50
        grad_smooth = np.convolve(grad_sum, kernel, mode='same')

        peak_idx = np.argmax(grad_smooth)
        peak_value = float(grad_smooth[peak_idx])
        peak_time = (start_idx + peak_idx) / SAMPLE_RATE

        # Onset backtracking
        baseline = np.median(grad_smooth[:int(0.5 * SAMPLE_RATE)])
        threshold = baseline + 0.3 * (grad_smooth[peak_idx] - baseline)
        onset_idx = peak_idx
        for i in range(peak_idx, -1, -1):
            if grad_smooth[i] < threshold:
                onset_idx = i
                break
        onset_time = (start_idx + onset_idx) / SAMPLE_RATE

        # Signal-to-noise ratio: peak / baseline
        snr = peak_value / max(baseline, 0.001)

        result['found'] = True  # EAG always finds a "peak" - question is whether it's meaningful
        result['onset_time'] = onset_time
        result['peak_time'] = peak_time
        result['peak_grad_value'] = peak_value
        result['baseline_value'] = float(baseline)
        result['snr'] = snr
        result['details'] = (
            f"onset={onset_time:.3f}s, peak={peak_time:.3f}s, "
            f"grad_peak={peak_value:.1f}, baseline={baseline:.1f}, SNR={snr:.1f}x"
        )

        # Mark as "not found" if SNR is very low (just picking noise)
        if snr < 2.0:
            result['found'] = False
            result['details'] += " [LOW SNR - likely noise]"

    except Exception as e:
        result['error'] = str(e)
        result['details'] = f"ERROR: {e}"

    return result


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    print("=" * 100)
    print("INITIAL 5-SECOND EVENT PATTERN CHECK - ALL SESSIONS")
    print("=" * 100)
    print()

    # Find all session pairs
    all_pairs = find_all_pairs(data_dir)
    print(f"Found {len(all_pairs)} session pairs with both EAG + GRF files")
    print()

    # Results storage
    results = []

    for i, pair in enumerate(all_pairs, 1):
        label = f"{pair.subject_name}/{pair.session_name}"
        print(f"[{i:2d}/{len(all_pairs)}] {label} ...")

        # Suppress print output from EAGAnalyzer
        import io
        from contextlib import redirect_stdout

        # Check GRF
        grf_result = check_grf_initial_event(pair.grf_filepath)

        # Check EAG (suppress verbose output)
        with redirect_stdout(io.StringIO()):
            eag_result = check_eag_initial_event(pair.eag_filepath)

        # Compute offset
        offset = None
        if grf_result['found'] and eag_result['found'] and \
           grf_result['onset_time'] is not None and eag_result['onset_time'] is not None:
            offset = eag_result['onset_time'] - grf_result['onset_time']

        both_found = grf_result['found'] and eag_result['found']

        results.append({
            'idx': i,
            'label': label,
            'subject': pair.subject_name,
            'session': pair.session_name,
            'grf_found': grf_result['found'],
            'grf_onset': grf_result['onset_time'],
            'grf_max_imb': grf_result['max_imbalance'],
            'grf_error': grf_result['error'],
            'grf_details': grf_result['details'],
            'eag_found': eag_result['found'],
            'eag_onset': eag_result['onset_time'],
            'eag_peak_time': eag_result.get('peak_time'),
            'eag_snr': eag_result.get('snr', 0),
            'eag_error': eag_result['error'],
            'eag_details': eag_result['details'],
            'offset': offset,
            'both_found': both_found,
        })

        # Compact status line
        grf_status = f"GRF: {grf_result['onset_time']:.3f}s" if grf_result['found'] else "GRF: NOT FOUND"
        eag_status = f"EAG: {eag_result['onset_time']:.3f}s" if eag_result['found'] else "EAG: NOT FOUND"
        offset_status = f"offset={offset:+.3f}s" if offset is not None else "offset=N/A"
        sync_ok = "OK" if both_found else "FAIL"
        print(f"       {grf_status}  |  {eag_status}  |  {offset_status}  [{sync_ok}]")

    # ======================== SUMMARY TABLE ========================
    print()
    print("=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    print(f"{'#':>3} {'Session':<45} {'GRF Onset':>10} {'EAG Onset':>10} {'Offset':>10} {'GRF MaxImb':>10} {'EAG SNR':>8} {'Status':>8}")
    print("-" * 130)

    for r in results:
        grf_str = f"{r['grf_onset']:.3f}s" if r['grf_onset'] is not None else "---"
        eag_str = f"{r['eag_onset']:.3f}s" if r['eag_onset'] is not None else "---"
        off_str = f"{r['offset']:+.3f}s" if r['offset'] is not None else "---"
        imb_str = f"{r['grf_max_imb']:.3f}"
        snr_str = f"{r['eag_snr']:.1f}x" if r['eag_snr'] else "---"
        status = "OK" if r['both_found'] else "FAIL"
        print(f"{r['idx']:3d} {r['label']:<45} {grf_str:>10} {eag_str:>10} {off_str:>10} {imb_str:>10} {snr_str:>8} {status:>8}")

    # ======================== STATISTICS ========================
    print()
    print("=" * 100)
    print("STATISTICS")
    print("=" * 100)

    total = len(results)
    grf_ok = sum(1 for r in results if r['grf_found'])
    eag_ok = sum(1 for r in results if r['eag_found'])
    both_ok = sum(1 for r in results if r['both_found'])

    print(f"  Total sessions:           {total}")
    print(f"  GRF event found:          {grf_ok}/{total} ({100*grf_ok/total:.1f}%)")
    print(f"  EAG event found:          {eag_ok}/{total} ({100*eag_ok/total:.1f}%)")
    print(f"  Both events found (sync): {both_ok}/{total} ({100*both_ok/total:.1f}%)")
    print()

    # Offset statistics for successful pairs
    offsets = [r['offset'] for r in results if r['offset'] is not None]
    if offsets:
        offsets_arr = np.array(offsets)
        print(f"  Offset stats (EAG - GRF), N={len(offsets)}:")
        print(f"    Mean:   {np.mean(offsets_arr):+.3f}s")
        print(f"    Median: {np.median(offsets_arr):+.3f}s")
        print(f"    Std:    {np.std(offsets_arr):.3f}s")
        print(f"    Min:    {np.min(offsets_arr):+.3f}s")
        print(f"    Max:    {np.max(offsets_arr):+.3f}s")
        print(f"    Range:  {np.ptp(offsets_arr):.3f}s")

    # ======================== FAILURE ANALYSIS ========================
    grf_failures = [r for r in results if not r['grf_found']]
    eag_failures = [r for r in results if not r['eag_found']]

    if grf_failures:
        print()
        print(f"  GRF FAILURES ({len(grf_failures)} sessions):")
        for r in grf_failures:
            print(f"    - {r['label']}: {r['grf_details']}")

    if eag_failures:
        print()
        print(f"  EAG FAILURES ({len(eag_failures)} sessions):")
        for r in eag_failures:
            print(f"    - {r['label']}: {r['eag_details']}")

    # ======================== PER-SUBJECT BREAKDOWN ========================
    print()
    print("=" * 100)
    print("PER-SUBJECT BREAKDOWN")
    print("=" * 100)

    subjects = {}
    for r in results:
        subj = r['subject']
        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(r)

    for subj, subj_results in subjects.items():
        n = len(subj_results)
        ok = sum(1 for r in subj_results if r['both_found'])
        grf_fail = sum(1 for r in subj_results if not r['grf_found'])
        eag_fail = sum(1 for r in subj_results if not r['eag_found'])
        subj_offsets = [r['offset'] for r in subj_results if r['offset'] is not None]

        print(f"\n  {subj}: {ok}/{n} sessions sync OK (GRF fail: {grf_fail}, EAG fail: {eag_fail})")
        if subj_offsets:
            arr = np.array(subj_offsets)
            print(f"    Offsets: mean={np.mean(arr):+.3f}s, std={np.std(arr):.3f}s, "
                  f"range=[{np.min(arr):+.3f}, {np.max(arr):+.3f}]")

        for r in subj_results:
            status = "OK" if r['both_found'] else "FAIL"
            off_str = f"{r['offset']:+.3f}s" if r['offset'] is not None else "N/A"
            print(f"    [{status}] {r['session']:>6}  GRF={r['grf_onset'] if r['grf_onset'] is not None else 'N/A':>8}  "
                  f"EAG={r['eag_onset'] if r['eag_onset'] is not None else 'N/A':>8}  offset={off_str}")

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == '__main__':
    main()
