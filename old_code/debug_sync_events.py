"""
Debug script: Visualize EAG and GRF raw data in the first 10 seconds
to identify synchronization events at ~2-3.5 seconds from recording start.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, '/workspace/research/EAGstatic')
from grf_viewer import load_grf_data, extract_body_weight, find_raw_data_start
from eag_analyzer import SAMPLE_RATE, EEG_CHANNELS, CHANNEL_NAMES, CHANNEL_COLORS

# === File paths ===
SESSION_DIR = "/workspace/research/EAGstatic/data/(02.28_13)황진성_2/OpenBCISession_2026-02-28_13-41-42-s1"
EAG_FILE = os.path.join(SESSION_DIR, "BrainFlow-RAW_2026-02-28_13-41-42-s1_0.csv")
GRF_FILE = os.path.join(SESSION_DIR, "양발_자세_황_진성2__282월26_13_41_58_282월26_13_43_39.csv")
OUTPUT_DIR = "/workspace/research/EAGstatic/results/debug_sync"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LOAD RAW EAG ====================
print("=" * 60)
print("Loading EAG data...")
eag_data = pd.read_csv(EAG_FILE, header=None, sep='\t', quotechar='"')
if len(eag_data.columns) == 1:
    eag_data = eag_data.iloc[:, 0].str.split('\t', expand=True)
    eag_data = eag_data.astype(float)

eag_raw = eag_data.iloc[:, 1:9].values.astype(float)  # 8 EEG channels
eag_timestamps = eag_data.iloc[:, 22].values.astype(float)  # Unix timestamps

print(f"  Total samples: {len(eag_raw)}")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Duration: {len(eag_raw)/SAMPLE_RATE:.1f} seconds")
print(f"  Timestamp[0]: {eag_timestamps[0]:.6f}")
print(f"  Timestamp[1]: {eag_timestamps[1]:.6f}")
print(f"  Timestamp[-1]: {eag_timestamps[-1]:.6f}")

# Check sample 0 artifact
if eag_timestamps[0] == 0:
    print("  [!] Sample 0 has timestamp=0 (artifact) - will skip in analysis")
    eag_start_unix = eag_timestamps[1]
else:
    eag_start_unix = eag_timestamps[0]

# Convert to KST
UTC_OFFSET = 9
eag_start_sod = (eag_start_unix % 86400) + UTC_OFFSET * 3600
if eag_start_sod >= 86400:
    eag_start_sod -= 86400
eag_h = int(eag_start_sod // 3600)
eag_m = int((eag_start_sod % 3600) // 60)
eag_s = eag_start_sod % 60
print(f"  EAG start time (KST): {eag_h:02d}:{eag_m:02d}:{eag_s:06.3f}")

# Time axis from recording start (seconds)
eag_time = np.arange(len(eag_raw)) / SAMPLE_RATE

# ==================== LOAD RAW GRF ====================
print(f"\n{'='*60}")
print("Loading GRF data...")
grf_df = load_grf_data(GRF_FILE)
body_weight = extract_body_weight(GRF_FILE)

grf_time = grf_df['time'].values
grf_left = grf_df['left_grf'].values
grf_right = grf_df['right_grf'].values
grf_total = grf_df['total_grf'].values

print(f"  Total samples: {len(grf_time)}")
print(f"  Time range: {grf_time[0]:.3f} to {grf_time[-1]:.3f} seconds")
print(f"  Duration: {grf_time[-1] - grf_time[0]:.1f} seconds")
print(f"  Body weight: {body_weight} kg")

# Parse GRF filename for start time
import re
basename = os.path.basename(GRF_FILE)
pattern = r'(\d{2})\d{1,2}월\d{2}_(\d{1,2})_(\d{2})_(\d{2})'
matches = re.findall(pattern, basename)
if matches:
    _, h, m, s = matches[0]
    grf_start_sod = int(h) * 3600 + int(m) * 60 + int(s)
    print(f"  GRF start time (KST): {int(h):02d}:{int(m):02d}:{int(s):02d}")
    print(f"  GRF start time (SoD): {grf_start_sod}")

print(f"\n{'='*60}")
print("Time comparison:")
print(f"  EAG start (SoD): {eag_start_sod:.3f}")
print(f"  GRF start (SoD): {grf_start_sod}")
print(f"  Offset (GRF - EAG): {grf_start_sod - eag_start_sod:.3f} seconds")
print(f"  (EAG started {eag_start_sod - grf_start_sod:.3f}s BEFORE GRF" if eag_start_sod < grf_start_sod else f"  (GRF started {grf_start_sod - eag_start_sod:.3f}s BEFORE EAG")
print(f"{'='*60}")

# ==================== PLOT 1: EAG RAW (first 10 seconds) ====================
print("\nPlotting EAG raw data (first 10 seconds)...")

# Use first 10 seconds
eag_10s_mask = eag_time <= 10.0
eag_time_10s = eag_time[eag_10s_mask]
eag_raw_10s = eag_raw[eag_10s_mask]

fig, axes = plt.subplots(EEG_CHANNELS, 1, figsize=(18, 2.5 * EEG_CHANNELS), sharex=True)

for ch in range(EEG_CHANNELS):
    ax = axes[ch]
    data_ch = eag_raw_10s[:, ch]
    ax.plot(eag_time_10s, data_ch, linewidth=0.5, color=CHANNEL_COLORS[ch])
    ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(raw)', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Highlight 2-3.5 second window
    ax.axvspan(2.0, 3.5, alpha=0.2, color='red', label='2-3.5s window')

    # Mark any large amplitude changes
    # Simple derivative to find sudden changes
    diff = np.abs(np.diff(data_ch))
    threshold = np.mean(diff) + 3 * np.std(diff)
    spike_idx = np.where(diff > threshold)[0]
    if len(spike_idx) > 0:
        for idx in spike_idx:
            if eag_time_10s[idx] <= 10.0:
                ax.axvline(x=eag_time_10s[idx], color='red', alpha=0.3, linewidth=0.5)

    # Add statistics
    ax.text(0.02, 0.95, f'mean={np.mean(data_ch):.0f}, std={np.std(data_ch):.0f}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Time from recording start (seconds)', fontsize=10)
fig.suptitle(f'EAG RAW Data - First 10 Seconds\n'
             f'Session: s1 | Start: {eag_h:02d}:{eag_m:02d}:{eag_s:06.3f} KST | '
             f'{len(eag_raw)} total samples @ {SAMPLE_RATE}Hz',
             fontsize=13, fontweight='bold')
plt.tight_layout()
eag_plot_path = os.path.join(OUTPUT_DIR, 'eag_raw_first10s.png')
plt.savefig(eag_plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {eag_plot_path}")

# ==================== PLOT 2: GRF (first 10 seconds) ====================
print("\nPlotting GRF data (first 10 seconds)...")

# GRF time relative to its own start
grf_rel_time = grf_time - grf_time[0]  # start from 0
grf_10s_mask = grf_rel_time <= 10.0
grf_time_10s = grf_rel_time[grf_10s_mask]
grf_left_10s = grf_left[grf_10s_mask]
grf_right_10s = grf_right[grf_10s_mask]
grf_total_10s = grf_total[grf_10s_mask]

fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

# Left GRF
axes[0].plot(grf_time_10s, grf_left_10s, color='#2196F3', linewidth=0.8, label='Left GRF')
axes[0].set_ylabel('Left GRF (kg)', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axvspan(2.0, 3.5, alpha=0.2, color='red', label='2-3.5s window')
axes[0].axhline(y=body_weight/2, color='gray', linestyle=':', alpha=0.5, label=f'BW/2 = {body_weight/2:.1f}kg')
axes[0].legend(loc='upper right', fontsize=8)

# Right GRF
axes[1].plot(grf_time_10s, grf_right_10s, color='#F44336', linewidth=0.8, label='Right GRF')
axes[1].set_ylabel('Right GRF (kg)', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].axvspan(2.0, 3.5, alpha=0.2, color='red', label='2-3.5s window')
axes[1].axhline(y=body_weight/2, color='gray', linestyle=':', alpha=0.5)
axes[1].legend(loc='upper right', fontsize=8)

# Total + Imbalance
ax2 = axes[2]
ax2.plot(grf_time_10s, grf_total_10s, color='green', linewidth=0.8, label='Total GRF')
ax2.set_ylabel('Total GRF (kg)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axvspan(2.0, 3.5, alpha=0.2, color='red', label='2-3.5s window')
ax2.axhline(y=body_weight, color='gray', linestyle=':', alpha=0.5, label=f'BW = {body_weight:.1f}kg')

# Secondary axis for imbalance
ax2_twin = ax2.twinx()
safe_total = np.maximum(grf_total_10s, 0.1)
imbalance_10s = (grf_left_10s - grf_right_10s) / safe_total
ax2_twin.plot(grf_time_10s, imbalance_10s, color='purple', linewidth=0.5, alpha=0.7, label='L-R imbalance')
ax2_twin.set_ylabel('Imbalance (L-R)/Total', fontsize=10, color='purple')
ax2_twin.axhline(y=0, color='purple', linestyle=':', alpha=0.3)
ax2_twin.legend(loc='lower right', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Time from recording start (seconds)', fontsize=10)
fig.suptitle(f'GRF Data - First 10 Seconds\n'
             f'Session: s1 | Start: {int(h):02d}:{int(m):02d}:{int(s):02d} KST | '
             f'Body Weight: {body_weight:.1f}kg',
             fontsize=13, fontweight='bold')
plt.tight_layout()
grf_plot_path = os.path.join(OUTPUT_DIR, 'grf_first10s.png')
plt.savefig(grf_plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {grf_plot_path}")

# ==================== DETAILED ANALYSIS: 0-5 second window ====================
print(f"\n{'='*60}")
print("DETAILED ANALYSIS: First 5 seconds")
print(f"{'='*60}")

# EAG: Look for significant events in first 5s
eag_5s_mask = eag_time <= 5.0
eag_5s = eag_raw[eag_5s_mask]
eag_time_5s = eag_time[eag_5s_mask]

print("\n--- EAG Channel Statistics (0-5s) ---")
for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch]
    diff = np.diff(data_ch)

    # Find large jumps
    threshold = np.mean(np.abs(diff)) + 3 * np.std(np.abs(diff))
    jumps = np.where(np.abs(diff) > threshold)[0]

    print(f"\n  Ch{ch+1}:")
    print(f"    Range: {np.min(data_ch):.1f} to {np.max(data_ch):.1f}")
    print(f"    Mean: {np.mean(data_ch):.1f}, Std: {np.std(data_ch):.1f}")

    if len(jumps) > 0:
        print(f"    Large jumps (>{threshold:.1f}) at times:")
        for j in jumps[:10]:  # show first 10
            t = eag_time_5s[j]
            print(f"      t={t:.4f}s: delta={diff[j]:.1f}")

# GRF: Look for significant events in first 5s
grf_5s_mask = grf_rel_time <= 5.0
grf_5s_left = grf_left[grf_5s_mask]
grf_5s_right = grf_right[grf_5s_mask]
grf_5s_total = grf_total[grf_5s_mask]
grf_5s_time = grf_rel_time[grf_5s_mask]

print(f"\n--- GRF Statistics (0-5s) ---")
print(f"  Left GRF range: {np.min(grf_5s_left):.2f} to {np.max(grf_5s_left):.2f} kg")
print(f"  Right GRF range: {np.min(grf_5s_right):.2f} to {np.max(grf_5s_right):.2f} kg")
print(f"  Total GRF range: {np.min(grf_5s_total):.2f} to {np.max(grf_5s_total):.2f} kg")

# Find when total GRF first exceeds a threshold (stepping on)
bw_threshold = body_weight * 0.1  # 10% of body weight
above_thresh = grf_5s_total > bw_threshold
if np.any(above_thresh):
    first_above = np.argmax(above_thresh)
    print(f"\n  Total GRF first exceeds {bw_threshold:.1f}kg (10% BW) at t={grf_5s_time[first_above]:.4f}s")

bw_threshold_50 = body_weight * 0.5  # 50% of body weight
above_50 = grf_5s_total > bw_threshold_50
if np.any(above_50):
    first_50 = np.argmax(above_50)
    print(f"  Total GRF first exceeds {bw_threshold_50:.1f}kg (50% BW) at t={grf_5s_time[first_50]:.4f}s")

bw_threshold_90 = body_weight * 0.9  # 90% of body weight
above_90 = grf_5s_total > bw_threshold_90
if np.any(above_90):
    first_90 = np.argmax(above_90)
    print(f"  Total GRF first exceeds {bw_threshold_90:.1f}kg (90% BW) at t={grf_5s_time[first_90]:.4f}s")

# Check GRF values at very start
print(f"\n  GRF values at recording start:")
for i in range(min(10, len(grf_5s_time))):
    print(f"    t={grf_5s_time[i]:.4f}s: L={grf_5s_left[i]:.2f}, R={grf_5s_right[i]:.2f}, Total={grf_5s_total[i]:.2f}")

# ==================== PLOT 3: Zoomed 0-5s both signals overlaid ====================
print(f"\n{'='*60}")
print("Creating zoomed comparison plot (0-5s)...")

fig, axes = plt.subplots(3, 1, figsize=(18, 12))

# Top: EAG all channels superimposed (normalized)
ax = axes[0]
for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch]
    # Normalize to zero-mean
    data_norm = data_ch - np.mean(data_ch)
    ax.plot(eag_time_5s, data_norm, linewidth=0.5, color=CHANNEL_COLORS[ch],
            label=CHANNEL_NAMES[ch], alpha=0.7)
ax.set_ylabel('EAG (zero-mean raw)', fontsize=10)
ax.set_title('EAG All Channels (0-5 seconds from recording start)', fontsize=12)
ax.axvspan(2.0, 3.5, alpha=0.15, color='red', label='2-3.5s target window')
ax.legend(loc='upper right', fontsize=7, ncol=5)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# Middle: EAG channel-by-channel derivative (to detect transients)
ax = axes[1]
for ch in range(EEG_CHANNELS):
    data_ch = eag_5s[:, ch]
    diff = np.diff(data_ch)
    ax.plot(eag_time_5s[1:], diff, linewidth=0.3, color=CHANNEL_COLORS[ch],
            label=CHANNEL_NAMES[ch], alpha=0.6)
ax.set_ylabel('EAG derivative', fontsize=10)
ax.set_title('EAG Derivative (sudden changes)', fontsize=12)
ax.axvspan(2.0, 3.5, alpha=0.15, color='red')
ax.legend(loc='upper right', fontsize=7, ncol=5)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# Bottom: GRF total + left + right
ax = axes[2]
ax.plot(grf_5s_time, grf_5s_left, color='#2196F3', linewidth=1.0, label='Left GRF')
ax.plot(grf_5s_time, grf_5s_right, color='#F44336', linewidth=1.0, label='Right GRF')
ax.plot(grf_5s_time, grf_5s_total, color='green', linewidth=1.5, label='Total GRF')
ax.axvspan(2.0, 3.5, alpha=0.15, color='red', label='2-3.5s target window')
ax.axhline(y=body_weight, color='gray', linestyle=':', alpha=0.5, label=f'BW={body_weight:.0f}kg')
ax.axhline(y=body_weight/2, color='gray', linestyle='--', alpha=0.3)
ax.set_ylabel('GRF (kg)', fontsize=10)
ax.set_title('GRF Left/Right/Total (0-5 seconds from recording start)', fontsize=12)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlabel('Time from recording start (seconds)', fontsize=10)
ax.set_xlim(0, 5)

plt.tight_layout()
zoom_plot_path = os.path.join(OUTPUT_DIR, 'eag_grf_first5s_comparison.png')
plt.savefig(zoom_plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {zoom_plot_path}")

# ==================== EVENT DETECTION in 2-3.5s window ====================
print(f"\n{'='*60}")
print("EVENT DETECTION: 2.0-3.5 second window")
print(f"{'='*60}")

# EAG: 2-3.5s window
eag_window_mask = (eag_time >= 2.0) & (eag_time <= 3.5)
eag_window = eag_raw[eag_window_mask]
eag_window_time = eag_time[eag_window_mask]

print(f"\n--- EAG in 2.0-3.5s window ---")
print(f"  Samples: {len(eag_window)} ({SAMPLE_RATE} Hz)")
for ch in range(EEG_CHANNELS):
    data_ch = eag_window[:, ch]
    diff = np.diff(data_ch)
    max_jump_idx = np.argmax(np.abs(diff))
    max_jump_time = eag_window_time[max_jump_idx]
    max_jump_val = diff[max_jump_idx]
    print(f"  Ch{ch+1}: range={np.max(data_ch)-np.min(data_ch):.1f}, "
          f"max_jump={max_jump_val:.1f} at t={max_jump_time:.4f}s")

# GRF: 2-3.5s window
grf_window_mask = (grf_rel_time >= 2.0) & (grf_rel_time <= 3.5)
grf_window_left = grf_left[grf_window_mask]
grf_window_right = grf_right[grf_window_mask]
grf_window_total = grf_total[grf_window_mask]
grf_window_time = grf_rel_time[grf_window_mask]

print(f"\n--- GRF in 2.0-3.5s window ---")
print(f"  Samples: {len(grf_window_left)}")
print(f"  Left:  {np.min(grf_window_left):.2f} to {np.max(grf_window_left):.2f} kg")
print(f"  Right: {np.min(grf_window_right):.2f} to {np.max(grf_window_right):.2f} kg")
print(f"  Total: {np.min(grf_window_total):.2f} to {np.max(grf_window_total):.2f} kg")

# Find the biggest change in GRF in this window
grf_total_diff = np.diff(grf_window_total)
max_change_idx = np.argmax(np.abs(grf_total_diff))
print(f"  Biggest total GRF change: {grf_total_diff[max_change_idx]:.2f} kg at t={grf_window_time[max_change_idx]:.4f}s")

# ==================== CURRENT vs EVENT-BASED OFFSET ====================
print(f"\n{'='*60}")
print("OFFSET COMPARISON")
print(f"{'='*60}")

# Current method: Unix timestamp based
current_offset = grf_start_sod - eag_start_sod
print(f"\n  Current method (Unix timestamp):")
print(f"    EAG start SoD: {eag_start_sod:.3f}")
print(f"    GRF start SoD: {grf_start_sod}")
print(f"    Offset (GRF - EAG): {current_offset:.3f}s")
print(f"    This means GRF sample at t=0 corresponds to EAG at t={current_offset:.3f}s")

# For event-based alignment we need to identify specific events
# The question is: what event at ~2-3.5s in each signal should be aligned?
print(f"\n  For event-based alignment:")
print(f"    We need to identify a common physical event visible in both signals")
print(f"    at approximately 2-3.5 seconds from each signal's recording start.")
print(f"    Typical sync event: subject stepping onto forceplate")

print(f"\n{'='*60}")
print("Done! Check plots in:", OUTPUT_DIR)
print(f"{'='*60}")
