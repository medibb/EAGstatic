"""
Debug script v2: Focused analysis on sync events in 0-10s window
- EAG: skip sample 0, zoom into actual signal variations (not artifact scale)
- GRF: detailed weight-shift event characterization
- Precise event timing for alignment calculation
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, '/workspace/research/EAGstatic')
from grf_viewer import load_grf_data, extract_body_weight
from eag_analyzer import SAMPLE_RATE, EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS

# === File paths ===
SESSION_DIR = "/workspace/research/EAGstatic/data/(02.28_13)황진성_2/OpenBCISession_2026-02-28_13-41-42-s1"
EAG_FILE = os.path.join(SESSION_DIR, "BrainFlow-RAW_2026-02-28_13-41-42-s1_0.csv")
GRF_FILE = os.path.join(SESSION_DIR, "양발_자세_황_진성2__282월26_13_41_58_282월26_13_43_39.csv")
OUTPUT_DIR = "/workspace/research/EAGstatic/results/debug_sync"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LOAD DATA ====================
# EAG
eag_data = pd.read_csv(EAG_FILE, header=None, sep='\t', quotechar='"')
if len(eag_data.columns) == 1:
    eag_data = eag_data.iloc[:, 0].str.split('\t', expand=True)
    eag_data = eag_data.astype(float)

eag_raw_all = eag_data.iloc[:, 1:9].values.astype(float)
eag_timestamps = eag_data.iloc[:, 22].values.astype(float)

# Skip sample 0 (artifact)
eag_raw = eag_raw_all[1:]  # skip sample 0
eag_time = np.arange(len(eag_raw)) / SAMPLE_RATE  # time from sample 1

# GRF
grf_df = load_grf_data(GRF_FILE)
body_weight = extract_body_weight(GRF_FILE)
grf_time = grf_df['time'].values - grf_df['time'].values[0]  # relative from 0
grf_left = grf_df['left_grf'].values
grf_right = grf_df['right_grf'].values
grf_total = grf_df['total_grf'].values

# ==================== PLOT: EAG zoomed (skip sample 0) first 10s ====================
print("Creating EAG zoomed plot (first 10s, sample 0 skipped)...")

eag_10s_mask = eag_time <= 10.0
eag_time_10s = eag_time[eag_10s_mask]
eag_raw_10s = eag_raw[eag_10s_mask]

fig, axes = plt.subplots(EEG_CHANNELS, 1, figsize=(20, 2.5 * EEG_CHANNELS), sharex=True)

for ch in range(EEG_CHANNELS):
    ax = axes[ch]
    data_ch = eag_raw_10s[:, ch]
    ax.plot(eag_time_10s, data_ch, linewidth=0.5, color=CHANNEL_COLORS[ch])
    ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(raw uV)', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Highlight windows
    ax.axvspan(2.0, 3.5, alpha=0.15, color='red', label='2-3.5s')
    ax.axvspan(0, 0.5, alpha=0.1, color='blue', label='0-0.5s init')

    # Auto-scale to signal variation (exclude outliers)
    p5, p95 = np.percentile(data_ch, [2, 98])
    margin = (p95 - p5) * 0.3
    ax.set_ylim(p5 - margin, p95 + margin)

    # Mark the signal range
    ax.text(0.02, 0.95, f'range: {p5:.0f} - {p95:.0f}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Time from recording start (seconds, sample 0 skipped)', fontsize=10)
fig.suptitle(f'EAG RAW Data (sample 0 skipped, zoomed to signal range)\nFirst 10 Seconds',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eag_raw_zoomed_10s.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ==================== PLOT: Side-by-side 0-5s with fine detail ====================
print("Creating detailed comparison plot (0-5s)...")

fig, axes = plt.subplots(4, 1, figsize=(20, 16), gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5]})

# --- Panel 1: EAG all channels (mean-subtracted, zoomed) ---
ax = axes[0]
eag_5s_mask = eag_time <= 5.0
eag_time_5s = eag_time[eag_5s_mask]
eag_5s = eag_raw[eag_5s_mask]

for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch]
    data_demean = data_ch - np.mean(data_ch)
    ax.plot(eag_time_5s, data_demean, linewidth=0.6, color=CHANNEL_COLORS[ch],
            label=CHANNEL_NAMES[ch], alpha=0.8)

ax.set_ylabel('EAG (mean-subtracted, uV)', fontsize=10)
ax.set_title('EAG 8 Channels (0-5s from EAG recording start)', fontsize=12, fontweight='bold')
ax.axvspan(2.0, 3.5, alpha=0.12, color='red')
ax.axvline(x=2.37, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='t=2.37s (max jump)')
ax.legend(loc='upper right', fontsize=7, ncol=5)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# --- Panel 2: EAG RMS across all channels (event detector) ---
ax = axes[1]
# Compute rolling RMS (window = 50ms = 12.5 samples)
window_samples = 13
rms_per_ch = np.zeros((len(eag_5s), EEG_CHANNELS))
for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch]
    data_demean = data_ch - np.mean(data_ch)
    # Rolling std as proxy for local RMS
    rms_per_ch[:, ch] = pd.Series(np.abs(np.gradient(data_demean))).rolling(
        window_samples, center=True, min_periods=1).mean().values

rms_mean = np.mean(rms_per_ch, axis=1)
ax.plot(eag_time_5s, rms_mean, color='black', linewidth=1.0)
ax.set_ylabel('EAG mean |gradient|\n(activity detector)', fontsize=10)
ax.set_title('EAG Activity Level (mean absolute gradient across channels)', fontsize=11)
ax.axvspan(2.0, 3.5, alpha=0.12, color='red')
ax.axvline(x=2.37, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# Find peak activity time
peak_activity_idx = np.argmax(rms_mean)
peak_activity_time = eag_time_5s[peak_activity_idx]
ax.annotate(f'Peak: t={peak_activity_time:.3f}s',
            xy=(peak_activity_time, rms_mean[peak_activity_idx]),
            xytext=(peak_activity_time + 0.3, rms_mean[peak_activity_idx] * 0.8),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red', fontweight='bold')

# --- Panel 3: GRF Left/Right ---
ax = axes[2]
grf_5s_mask = grf_time <= 5.0
grf_time_5s = grf_time[grf_5s_mask]
grf_left_5s = grf_left[grf_5s_mask]
grf_right_5s = grf_right[grf_5s_mask]
grf_total_5s = grf_total[grf_5s_mask]

ax.plot(grf_time_5s, grf_left_5s, color='#2196F3', linewidth=1.0, label='Left GRF')
ax.plot(grf_time_5s, grf_right_5s, color='#F44336', linewidth=1.0, label='Right GRF')
ax.plot(grf_time_5s, grf_total_5s, color='green', linewidth=1.5, alpha=0.5, label='Total GRF')
ax.axhline(y=body_weight/2, color='gray', linestyle=':', alpha=0.4, label=f'BW/2={body_weight/2:.1f}')
ax.set_ylabel('GRF (kg)', fontsize=10)
ax.set_title('GRF Left/Right/Total (0-5s from GRF recording start)', fontsize=12, fontweight='bold')
ax.axvspan(2.0, 3.5, alpha=0.12, color='red')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# --- Panel 4: GRF derivative (event onset detector) ---
ax = axes[3]
grf_left_diff = np.gradient(grf_left_5s, grf_time_5s)
grf_right_diff = np.gradient(grf_right_5s, grf_time_5s)
ax.plot(grf_time_5s, grf_left_diff, color='#2196F3', linewidth=0.8, label='Left dGRF/dt')
ax.plot(grf_time_5s, grf_right_diff, color='#F44336', linewidth=0.8, label='Right dGRF/dt')
ax.set_ylabel('GRF rate (kg/s)', fontsize=10)
ax.set_title('GRF Rate of Change (event onset detector)', fontsize=11)
ax.axvspan(2.0, 3.5, alpha=0.12, color='red')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)
ax.set_xlabel('Time from recording start (seconds)', fontsize=11)

# Find GRF event onset
# The event is a weight shift: left goes up, right goes down (or vice versa)
# Look for the time when the left GRF first starts rising significantly
left_onset_candidates = np.where(grf_left_diff > 5.0)[0]  # >5 kg/s
if len(left_onset_candidates) > 0:
    grf_event_onset_idx = left_onset_candidates[0]
    grf_event_onset_time = grf_time_5s[grf_event_onset_idx]
    ax.axvline(x=grf_event_onset_time, color='green', linewidth=2, linestyle='--',
               alpha=0.7, label=f'GRF onset: {grf_event_onset_time:.3f}s')
    ax.legend(loc='upper right', fontsize=8)

    # Also find the right foot unloading onset
    right_onset_candidates = np.where(grf_right_diff < -5.0)[0]
    if len(right_onset_candidates) > 0:
        grf_right_onset_time = grf_time_5s[right_onset_candidates[0]]

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eag_grf_detailed_5s.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ==================== PRECISE EVENT TIMING ====================
print(f"\n{'='*70}")
print("PRECISE EVENT TIMING ANALYSIS")
print(f"{'='*70}")

# --- GRF Event Characterization ---
print("\n[GRF WEIGHT-SHIFT EVENT]")
print(f"  At GRF recording start (t=0): L={grf_left[0]:.2f}kg, R={grf_right[0]:.2f}kg, Total={grf_total[0]:.2f}kg")
print(f"  Subject is already standing on forceplate (Total ~= BW of {body_weight:.1f}kg)")

# Find the onset of the weight shift event more precisely
# Define "onset" as when Left GRF exceeds its baseline mean + 2*std
baseline_mask = grf_time < 1.5  # use first 1.5s as baseline
left_baseline_mean = np.mean(grf_left[baseline_mask])
left_baseline_std = np.std(grf_left[baseline_mask])
right_baseline_mean = np.mean(grf_right[baseline_mask])
right_baseline_std = np.std(grf_right[baseline_mask])

print(f"  Baseline (0-1.5s): Left={left_baseline_mean:.2f}+/-{left_baseline_std:.2f}kg, "
      f"Right={right_baseline_mean:.2f}+/-{right_baseline_std:.2f}kg")

# Left rise onset: first time Left > baseline + 3*std
left_threshold = left_baseline_mean + 3 * left_baseline_std
onset_mask = (grf_time > 1.0) & (grf_left > left_threshold)
if np.any(onset_mask):
    grf_left_onset_time = grf_time[np.argmax(onset_mask)]
    print(f"  Left rise onset (L > {left_threshold:.1f}kg): t = {grf_left_onset_time:.4f}s")

# Right drop onset: first time Right < baseline - 3*std
right_threshold = right_baseline_mean - 3 * right_baseline_std
drop_mask = (grf_time > 1.0) & (grf_right < right_threshold)
if np.any(drop_mask):
    grf_right_drop_time = grf_time[np.argmax(drop_mask)]
    print(f"  Right drop onset (R < {right_threshold:.1f}kg): t = {grf_right_drop_time:.4f}s")

# Peak weight shift
event_mask = (grf_time >= 2.0) & (grf_time <= 4.0)
event_left = grf_left[event_mask]
event_right = grf_right[event_mask]
event_time = grf_time[event_mask]

peak_left_idx = np.argmax(event_left)
peak_left_time = event_time[peak_left_idx]
min_right_idx = np.argmin(event_right)
min_right_time = event_time[min_right_idx]

print(f"  Peak Left GRF: {event_left[peak_left_idx]:.2f}kg at t={peak_left_time:.4f}s")
print(f"  Min Right GRF: {event_right[min_right_idx]:.2f}kg at t={min_right_time:.4f}s")
print(f"  Peak imbalance ratio: {(event_left[peak_left_idx] - event_right[min_right_idx]) / body_weight:.2f}")

# Return to baseline after event
post_event_mask = (grf_time > 3.5) & (grf_time < 5.0)
post_event_left = grf_left[post_event_mask]
post_event_right = grf_right[post_event_mask]
# Find when it returns to within 2 SD of baseline
recovery_mask_left = (grf_time > 3.0) & (np.abs(grf_left - left_baseline_mean) < 2 * left_baseline_std)
if np.any(recovery_mask_left):
    recovery_time = grf_time[np.argmax(recovery_mask_left)]
    print(f"  Recovery time: t = {recovery_time:.4f}s")

# --- EAG Event Characterization ---
print(f"\n[EAG EVENT IN 2-3.5s WINDOW]")
# Look more carefully at what happens at t=2.37s
eag_window_mask = (eag_time >= 1.5) & (eag_time <= 4.0)
eag_window = eag_raw[eag_window_mask]
eag_window_time = eag_time[eag_window_mask]

for ch in range(EEG_CHANNELS):
    data_ch = eag_window[:, ch]
    grad = np.gradient(data_ch)

    # Find the largest transient
    max_grad_idx = np.argmax(np.abs(grad))
    max_grad_time = eag_window_time[max_grad_idx]
    max_grad_val = grad[max_grad_idx]

    # Find sustained signal change (mean shift)
    first_half = data_ch[:len(data_ch)//2]
    second_half = data_ch[len(data_ch)//2:]
    mean_shift = np.mean(second_half) - np.mean(first_half)

    print(f"  Ch{ch+1}: max_gradient={max_grad_val:+.1f} at t={max_grad_time:.4f}s, "
          f"mean_shift={mean_shift:+.1f}")

# ==================== TIMESTAMP-BASED OFFSET ====================
print(f"\n{'='*70}")
print("CURRENT TIMESTAMP-BASED ALIGNMENT")
print(f"{'='*70}")

import re
# EAG start
UTC_OFFSET = 9
eag_start_unix = eag_timestamps[1]  # skip sample 0
eag_start_sod = (eag_start_unix % 86400) + UTC_OFFSET * 3600
if eag_start_sod >= 86400:
    eag_start_sod -= 86400

# GRF start from filename
basename = os.path.basename(GRF_FILE)
pattern = r'(\d{2})\d{1,2}월\d{2}_(\d{1,2})_(\d{2})_(\d{2})'
matches = re.findall(pattern, basename)
_, h, m, s = matches[0]
grf_start_sod = int(h) * 3600 + int(m) * 60 + int(s)

ts_offset = grf_start_sod - eag_start_sod

print(f"  EAG recording starts at {eag_start_sod:.3f} SoD (13:42:04.959 KST)")
print(f"  GRF recording starts at {grf_start_sod} SoD (13:41:58 KST)")
print(f"  GRF started {-ts_offset:.3f}s BEFORE EAG")
print(f"  Current offset (GRF_start - EAG_start): {ts_offset:.3f}s")

# With this offset, what EAG time corresponds to the GRF event?
print(f"\n  With timestamp offset ({ts_offset:.3f}s):")
print(f"    GRF event onset at GRF t={grf_left_onset_time:.3f}s")
print(f"    = absolute SoD: {grf_start_sod + grf_left_onset_time:.3f}")
print(f"    = EAG time: {grf_left_onset_time + ts_offset:.3f}s from EAG start")
print(f"    (i.e., {grf_left_onset_time - (-ts_offset):.3f}s from EAG start)")

grf_event_in_eag_time = grf_left_onset_time - (-ts_offset)
print(f"\n  So GRF's weight-shift event at GRF t={grf_left_onset_time:.3f}s")
print(f"  maps to EAG t={grf_event_in_eag_time:.3f}s using timestamp alignment")

# ==================== EVENT-BASED OFFSET CALCULATION ====================
print(f"\n{'='*70}")
print("EVENT-BASED ALIGNMENT CALCULATION")
print(f"{'='*70}")

# The key insight: if the same physical event appears at different times
# in each signal, the true offset = EAG_event_time - GRF_event_time

# For EAG: find the event time in the 2-3.5s window
# The max jump was at t=2.372s across most channels
# Let's compute more precisely

eag_event_candidates = []
for ch in range(EEG_CHANNELS):
    eag_25_mask = (eag_time >= 1.5) & (eag_time <= 4.0)
    data_ch = eag_raw[eag_25_mask, ch]
    time_ch = eag_time[eag_25_mask]
    grad = np.abs(np.gradient(data_ch))
    # Smooth slightly
    grad_smooth = pd.Series(grad).rolling(5, center=True, min_periods=1).mean().values
    peak_idx = np.argmax(grad_smooth)
    eag_event_candidates.append(time_ch[peak_idx])

eag_event_time = np.median(eag_event_candidates)
print(f"\n  EAG event time (median of ch max gradients): {eag_event_time:.4f}s")
print(f"    Individual channel times: {[f'{t:.4f}' for t in eag_event_candidates]}")

print(f"\n  GRF event onset time: {grf_left_onset_time:.4f}s")
print(f"  GRF peak weight-shift time: {peak_left_time:.4f}s")

# If these are the same physical event:
event_offset = eag_event_time - grf_left_onset_time
print(f"\n  Event-based offset (EAG_event - GRF_onset): {event_offset:.4f}s")
print(f"  This means: GRF_time + ({event_offset:.4f}) = EAG_time")
print(f"  Or: EAG started {-event_offset:.4f}s {'after' if event_offset < 0 else 'before'} GRF")

event_offset_peak = eag_event_time - peak_left_time
print(f"\n  Alternative (using GRF peak instead of onset): {event_offset_peak:.4f}s")

# Compare
print(f"\n{'='*70}")
print("COMPARISON: Timestamp vs Event-based offsets")
print(f"{'='*70}")
print(f"  Timestamp offset (GRF-EAG): {ts_offset:.3f}s")
print(f"  Event-based offset (EAG_event - GRF_onset): {event_offset:.4f}s")
print(f"  Difference: {event_offset - ts_offset:.4f}s")
print(f"\n  If the GRF weight-shift event is the 'sync signal':  ")
print(f"    The GRF filename timestamp has {-ts_offset:.3f}s offset from EAG Unix ts")
print(f"    But the actual physical events differ by {event_offset:.4f}s")
print(f"    Correction needed: {event_offset - ts_offset:.4f}s")

# ==================== PLOT 4: Aligned comparison ====================
print(f"\nCreating alignment comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(24, 12))

# Top-left: EAG channels (individual) in 1-5s
ax = axes[0, 0]
for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch] - np.mean(eag_5s[:, ch])
    ax.plot(eag_time_5s, data_ch, linewidth=0.5, color=CHANNEL_COLORS[ch],
            label=CHANNEL_NAMES[ch], alpha=0.7)
ax.axvline(x=eag_event_time, color='red', linewidth=2, linestyle='--',
           label=f'EAG event: {eag_event_time:.3f}s')
ax.set_title('EAG Channels (from EAG t=0)', fontsize=12, fontweight='bold')
ax.set_ylabel('Amplitude (mean-subtracted)', fontsize=10)
ax.set_xlabel('EAG time (seconds)', fontsize=10)
ax.legend(fontsize=7, ncol=5, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 5)

# Top-right: GRF in 1-5s
ax = axes[0, 1]
ax.plot(grf_time_5s, grf_left_5s, color='#2196F3', linewidth=1.2, label='Left GRF')
ax.plot(grf_time_5s, grf_right_5s, color='#F44336', linewidth=1.2, label='Right GRF')
ax.axvline(x=grf_left_onset_time, color='green', linewidth=2, linestyle='--',
           label=f'GRF onset: {grf_left_onset_time:.3f}s')
ax.axvline(x=peak_left_time, color='darkgreen', linewidth=2, linestyle=':',
           label=f'GRF peak: {peak_left_time:.3f}s')
ax.set_title('GRF (from GRF t=0)', fontsize=12, fontweight='bold')
ax.set_ylabel('GRF (kg)', fontsize=10)
ax.set_xlabel('GRF time (seconds)', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 5)

# Bottom-left: TIMESTAMP-aligned overlay
ax = axes[1, 0]
# Using timestamp offset: GRF_time_in_EAG = grf_time + ts_offset
grf_time_ts_aligned = grf_time + ts_offset  # map GRF time to EAG time
grf_ts_5s_mask = (grf_time_ts_aligned >= 0) & (grf_time_ts_aligned <= 5.0)

for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch] - np.mean(eag_5s[:, ch])
    ax.plot(eag_time_5s, data_ch / np.std(data_ch) * 10, linewidth=0.3,
            color=CHANNEL_COLORS[ch], alpha=0.5)

ax2 = ax.twinx()
ax2.plot(grf_time_ts_aligned[grf_ts_5s_mask], grf_left[grf_ts_5s_mask],
         color='blue', linewidth=1.5, label='Left GRF')
ax2.plot(grf_time_ts_aligned[grf_ts_5s_mask], grf_right[grf_ts_5s_mask],
         color='red', linewidth=1.5, label='Right GRF')
ax2.set_ylabel('GRF (kg)', fontsize=10, color='blue')

ax.axvline(x=eag_event_time, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax.set_title(f'TIMESTAMP-ALIGNED (offset={ts_offset:.3f}s)', fontsize=12, fontweight='bold')
ax.set_ylabel('EAG (normalized)', fontsize=10)
ax.set_xlabel('EAG time (seconds)', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 5)
ax2.legend(loc='upper right', fontsize=8)

# Bottom-right: EVENT-aligned overlay
ax = axes[1, 1]
# Using event-based offset: GRF_time_in_EAG = grf_time + event_offset
grf_time_evt_aligned = grf_time + event_offset
grf_evt_5s_mask = (grf_time_evt_aligned >= 0) & (grf_time_evt_aligned <= 5.0)

for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch] - np.mean(eag_5s[:, ch])
    ax.plot(eag_time_5s, data_ch / np.std(data_ch) * 10, linewidth=0.3,
            color=CHANNEL_COLORS[ch], alpha=0.5)

ax2 = ax.twinx()
ax2.plot(grf_time_evt_aligned[grf_evt_5s_mask], grf_left[grf_evt_5s_mask],
         color='blue', linewidth=1.5, label='Left GRF')
ax2.plot(grf_time_evt_aligned[grf_evt_5s_mask], grf_right[grf_evt_5s_mask],
         color='red', linewidth=1.5, label='Right GRF')
ax2.set_ylabel('GRF (kg)', fontsize=10, color='blue')

ax.axvline(x=eag_event_time, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax.set_title(f'EVENT-ALIGNED (offset={event_offset:.3f}s)', fontsize=12, fontweight='bold')
ax.set_ylabel('EAG (normalized)', fontsize=10)
ax.set_xlabel('EAG time (seconds)', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 5)
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'alignment_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ==================== FINAL SUMMARY ====================
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
print(f"""
DATA OVERVIEW:
  EAG: 15058 samples @ 250Hz = 60.2s, starts at 13:42:04.959 KST
  GRF: 15000 samples @ 250Hz = 60.0s, starts at 13:41:58 KST
  GRF started {-ts_offset:.3f}s before EAG (by clock timestamps)

EVENTS IN 2-3.5s WINDOW:
  GRF: Major weight-shift event (left foot loading)
    - Onset: {grf_left_onset_time:.4f}s from GRF start
    - Peak: {peak_left_time:.4f}s (Left={event_left[peak_left_idx]:.1f}kg, Right={event_right[min_right_idx]:.1f}kg)
    - Pattern: Right foot unloads nearly to 0, all weight shifts to left

  EAG: Simultaneous transient across all 8 channels
    - Event time: {eag_event_time:.4f}s from EAG start
    - All channels show sharp gradient at ~2.37s
    - Ch6 & Ch7 show positive jump, others show negative jump

ALIGNMENT:
  Timestamp-based offset: {ts_offset:.3f}s (GRF_start - EAG_start)
  Event-based offset:     {event_offset:.4f}s (EAG_event - GRF_onset)
  Difference:             {event_offset - ts_offset:.4f}s

  The timestamp approach says GRF started {-ts_offset:.3f}s before EAG.
  Under timestamp alignment, GRF onset at {grf_left_onset_time:.3f}s maps to
  EAG time {grf_left_onset_time + ts_offset:.3f}s (={grf_left_onset_time:.3f}+({ts_offset:.3f}))

  But the actual EAG event is at {eag_event_time:.3f}s.
  So the timestamp alignment places the GRF event at EAG t={grf_left_onset_time + ts_offset:.3f}s,
  while the EAG transient is at {eag_event_time:.3f}s.

  Discrepancy: {eag_event_time - (grf_left_onset_time + ts_offset):.4f}s
""")

print(f"Plots saved to: {OUTPUT_DIR}")
print(f"{'='*70}")
