#!/usr/bin/env python3
"""
Comprehensive validation of 2-stage sync (event + xcorr fallback) across all 86 sessions.

For each session:
  - Creates SyncAnalyzer with FilterConfig defaults
  - Records: sync_method, time_offset, overlap_duration, xcorr_confidence
  - Catches all exceptions and continues

Prints a detailed summary at the end.
"""

import sys
import os
import time
import traceback
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sync_analyzer import find_all_pairs, SyncAnalyzer, SessionPair
from eag_analyzer import FilterConfig, get_data_dir


def main():
    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}")
    pairs = find_all_pairs(data_dir)
    print(f"Total session pairs found: {len(pairs)}\n")

    # Result storage
    results = []
    failures = []

    total = len(pairs)
    t_start_all = time.time()

    for i, pair in enumerate(pairs):
        idx = i + 1
        label = f"{pair.subject_name} / {pair.session_name}"
        print(f"[{idx:3d}/{total}] {label} ... ", end="", flush=True)

        t0 = time.time()
        try:
            config = FilterConfig()
            analyzer = SyncAnalyzer(pair, config=config)

            method = getattr(analyzer, 'sync_method', 'unknown')
            offset = analyzer.time_offset
            overlap = analyzer.overlap_duration
            xcorr_conf = getattr(analyzer, '_xcorr_confidence', 0.0)

            elapsed = time.time() - t0
            print(f"OK  method={method:5s}  offset={offset:+7.3f}s  "
                  f"overlap={overlap:6.1f}s  xcorr_conf={xcorr_conf:.4f}  "
                  f"({elapsed:.1f}s)")

            results.append({
                'idx': idx,
                'subject': pair.subject_name,
                'session': pair.session_name,
                'label': label,
                'method': method,
                'offset': offset,
                'overlap': overlap,
                'xcorr_conf': xcorr_conf,
                'error': None,
            })

        except Exception as e:
            elapsed = time.time() - t0
            err_msg = f"{type(e).__name__}: {e}"
            print(f"FAIL  ({elapsed:.1f}s)  {err_msg}")
            failures.append({
                'idx': idx,
                'subject': pair.subject_name,
                'session': pair.session_name,
                'label': label,
                'error': err_msg,
                'traceback': traceback.format_exc(),
            })
            results.append({
                'idx': idx,
                'subject': pair.subject_name,
                'session': pair.session_name,
                'label': label,
                'method': 'FAIL',
                'offset': float('nan'),
                'overlap': float('nan'),
                'xcorr_conf': float('nan'),
                'error': err_msg,
            })

    total_elapsed = time.time() - t_start_all

    # ======================== SUMMARY ========================
    print("\n")
    print("=" * 80)
    print("  COMPREHENSIVE SYNC VALIDATION SUMMARY")
    print("=" * 80)

    success = [r for r in results if r['error'] is None]
    event_sessions = [r for r in success if r['method'] == 'event']
    xcorr_sessions = [r for r in success if r['method'] == 'xcorr']

    print(f"\nTotal sessions:       {total}")
    print(f"Successful:           {len(success)}")
    print(f"  Event-sync:         {len(event_sessions)}")
    print(f"  XCorr-fallback:     {len(xcorr_sessions)}")
    print(f"Failed (exceptions):  {len(failures)}")
    print(f"Total time:           {total_elapsed:.1f}s  "
          f"(avg {total_elapsed/total:.1f}s per session)")

    # ---- Offset statistics: event vs xcorr ----
    if event_sessions:
        ev_offsets = [r['offset'] for r in event_sessions]
        print(f"\n--- Event-sync offset statistics (n={len(event_sessions)}) ---")
        print(f"  Mean:   {np.mean(ev_offsets):+.4f}s")
        print(f"  Median: {np.median(ev_offsets):+.4f}s")
        print(f"  Std:    {np.std(ev_offsets):.4f}s")
        print(f"  Min:    {np.min(ev_offsets):+.4f}s")
        print(f"  Max:    {np.max(ev_offsets):+.4f}s")

    if xcorr_sessions:
        xc_offsets = [r['offset'] for r in xcorr_sessions]
        xc_confs = [r['xcorr_conf'] for r in xcorr_sessions]
        print(f"\n--- XCorr-fallback offset statistics (n={len(xcorr_sessions)}) ---")
        print(f"  Mean:   {np.mean(xc_offsets):+.4f}s")
        print(f"  Median: {np.median(xc_offsets):+.4f}s")
        print(f"  Std:    {np.std(xc_offsets):.4f}s")
        print(f"  Min:    {np.min(xc_offsets):+.4f}s")
        print(f"  Max:    {np.max(xc_offsets):+.4f}s")
        print(f"\n  XCorr confidence:")
        print(f"    Mean:   {np.mean(xc_confs):.4f}")
        print(f"    Median: {np.median(xc_confs):.4f}")
        print(f"    Min:    {np.min(xc_confs):.4f}")
        print(f"    Max:    {np.max(xc_confs):.4f}")

    # ---- Overlap statistics ----
    if success:
        overlaps = [r['overlap'] for r in success]
        print(f"\n--- Overlap duration statistics (n={len(success)}) ---")
        print(f"  Mean:   {np.mean(overlaps):.1f}s")
        print(f"  Median: {np.median(overlaps):.1f}s")
        print(f"  Min:    {np.min(overlaps):.1f}s")
        print(f"  Max:    {np.max(overlaps):.1f}s")

    # ---- Per-subject breakdown ----
    print(f"\n--- Per-subject breakdown ---")
    subjects = sorted(set(r['subject'] for r in results))
    print(f"{'Subject':<35s} {'Total':>5s} {'Event':>5s} {'XCorr':>5s} {'Fail':>5s}")
    print("-" * 60)
    for subj in subjects:
        subj_results = [r for r in results if r['subject'] == subj]
        n_total = len(subj_results)
        n_event = len([r for r in subj_results if r['method'] == 'event'])
        n_xcorr = len([r for r in subj_results if r['method'] == 'xcorr'])
        n_fail = len([r for r in subj_results if r['error'] is not None])
        print(f"{subj:<35s} {n_total:>5d} {n_event:>5d} {n_xcorr:>5d} {n_fail:>5d}")

    # ---- List all xcorr sessions ----
    if xcorr_sessions:
        print(f"\n--- All XCorr-fallback sessions (n={len(xcorr_sessions)}) ---")
        print(f"{'#':>3s}  {'Subject / Session':<45s} {'Offset':>9s} {'Conf':>8s} {'Overlap':>9s}")
        print("-" * 78)
        for r in xcorr_sessions:
            flag = " ***" if abs(r['offset']) > 2.0 else ""
            print(f"{r['idx']:3d}  {r['label']:<45s} {r['offset']:+9.3f} {r['xcorr_conf']:8.4f} "
                  f"{r['overlap']:8.1f}s{flag}")

    # ---- Flag |offset| > 2.0s (any method) ----
    large_offset = [r for r in success if abs(r['offset']) > 2.0]
    if large_offset:
        print(f"\n--- WARNING: Sessions with |offset| > 2.0s (n={len(large_offset)}) ---")
        print(f"{'#':>3s}  {'Subject / Session':<45s} {'Method':>6s} {'Offset':>9s} {'Conf':>8s}")
        print("-" * 75)
        for r in large_offset:
            print(f"{r['idx']:3d}  {r['label']:<45s} {r['method']:>6s} {r['offset']:+9.3f} "
                  f"{r['xcorr_conf']:8.4f}")
    else:
        print(f"\n--- No sessions with |offset| > 2.0s (all within range) ---")

    # ---- Failures detail ----
    if failures:
        print(f"\n--- FAILURES detail (n={len(failures)}) ---")
        for f in failures:
            print(f"\n  [{f['idx']}] {f['label']}")
            print(f"      {f['error']}")
            # Print last 3 lines of traceback
            tb_lines = f['traceback'].strip().split('\n')
            for line in tb_lines[-3:]:
                print(f"      {line}")
    else:
        print(f"\n--- No failures: all {total} sessions processed successfully ---")

    # ---- Final verdict ----
    print(f"\n{'=' * 80}")
    success_rate = len(success) / total * 100 if total > 0 else 0
    event_rate = len(event_sessions) / len(success) * 100 if success else 0
    xcorr_rate = len(xcorr_sessions) / len(success) * 100 if success else 0
    print(f"  SUCCESS RATE: {len(success)}/{total} ({success_rate:.1f}%)")
    print(f"  Event-sync:   {len(event_sessions)}/{len(success)} ({event_rate:.1f}% of successful)")
    print(f"  XCorr-fb:     {len(xcorr_sessions)}/{len(success)} ({xcorr_rate:.1f}% of successful)")
    print(f"  Failures:     {len(failures)}/{total}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
