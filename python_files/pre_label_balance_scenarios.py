import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-training.csv")

# --- PARAMETERS ---
MIN_TARGET = 750   # Minimum rows per scenario file (tune as needed)
MAX_TARGET = 1500   # Maximum rows per scenario file (tune as needed)
RANDOM_SEED = 42
PERTURB_STD_FRAC = 0.35  # Aggressive upscaling

print(f"\n=== BALANCED DATASET SCRIPT (Scenario-only, Prelabeling Data) ===")
print(f"Input file: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV, low_memory=False)
if 'scenario_file' not in df.columns:
    raise ValueError("Combined CSV must have a 'scenario_file' column!")

print(f"Total rows before cleaning: {len(df)}")

# --- FEATURE CLEANING BEFORE BALANCING ---
KEY_FEATURES = [
    "Distance", "Speed", "Interferers", "PacketSize", "TrafficRate",
    "Throughput(Mbps)", "PacketLoss(%)", "AvgDelay(ms)", "RxPackets", "TxPackets", "lastSnr"
]
KEY_FEATURES = [c for c in KEY_FEATURES if c in df.columns]
df = df.dropna(subset=KEY_FEATURES)
print(f"Removed rows with NaNs in key features. Remaining: {len(df)}")

# Filter by SNR (remove low SNR rows)
if "lastSnr" in df.columns:
    snr_before = len(df)
    df = df[df["lastSnr"] >= 3]
    print(f"Filtered {snr_before - len(df)} rows with lastSnr < 3. Remaining: {len(df)}")

# Fix out-of-range values
if "PacketLoss(%)" in df.columns:
    df["PacketLoss(%)"] = df["PacketLoss(%)"].clip(lower=0, upper=100)
for col in ["Interferers", "RxPackets", "TxPackets", "PacketSize"]:
    if col in df.columns:
        df[col] = df[col].clip(lower=0).round().astype(int)
for col in ["Distance", "Speed", "TrafficRate", "Throughput(Mbps)", "AvgDelay(ms)"]:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

print(f"Total rows after feature cleaning: {len(df)}")

# --- Remove empty/corrupt scenarios ---
scenario_counts = df['scenario_file'].value_counts()
valid_scenarios = scenario_counts[scenario_counts > 0].index.tolist()
df = df[df['scenario_file'].isin(valid_scenarios)]

print(f"Found {len(valid_scenarios)} valid scenario files.")

balanced_dfs = []

SAFE_FEATURES = [
    "Distance", "Speed", "Interferers", "PacketSize", "TrafficRate",
    "Throughput(Mbps)", "PacketLoss(%)", "AvgDelay(ms)", "RxPackets", "TxPackets"
]
SAFE_FEATURES = [c for c in SAFE_FEATURES if c in df.columns]

def safe_perturb(val, std, global_min, global_max, col):
    delta = np.random.uniform(-PERTURB_STD_FRAC, PERTURB_STD_FRAC) * std
    new_val = val + delta
    new_val = max(global_min, min(global_max, new_val))
    if col == 'PacketLoss(%)':
        new_val = max(0, min(100, new_val))
    if col in ['Interferers', 'RxPackets', 'TxPackets', 'PacketSize']:
        new_val = int(round(max(0, new_val)))
    if col in ['Distance', 'Speed', 'TrafficRate', 'Throughput(Mbps)', 'AvgDelay(ms)']:
        new_val = max(0, new_val)
    return new_val

global_mins = df[SAFE_FEATURES].min()
global_maxs = df[SAFE_FEATURES].max()

for fname in valid_scenarios:
    df_sub = df[df['scenario_file'] == fname]
    n = len(df_sub)
    print(f"\n--- {fname}: {n} rows ---")
    subframes = []
    # No label balancing, just scenario balancing
    if n < MIN_TARGET:
        # SMART UPSCALING FOR WHOLE SCENARIO
        n_needed = MIN_TARGET - n
        means = df_sub[SAFE_FEATURES].mean()
        stds = df_sub[SAFE_FEATURES].std().replace({0: 1e-6})
        synthetic_rows = []
        for idx in range(n_needed):
            base = df_sub.sample(1, random_state=RANDOM_SEED + idx).iloc[0].copy()
            new_row = base.copy()
            for col in SAFE_FEATURES:
                val = base[col]
                new_row[col] = safe_perturb(val, stds[col], global_mins[col], global_maxs[col], col)
            synthetic_rows.append(new_row)
        oversampled = pd.concat([df_sub, pd.DataFrame(synthetic_rows)], ignore_index=True)
        balanced_dfs.append(oversampled)
        print(f"  Scenario smart oversampled: {n} -> {MIN_TARGET}")
    elif n > MAX_TARGET:
        downsampled = df_sub.sample(MAX_TARGET, replace=False, random_state=RANDOM_SEED)
        balanced_dfs.append(downsampled)
        print(f"  Scenario downsampled: {n} -> {MAX_TARGET}")
    else:
        balanced_dfs.append(df_sub)
        print(f"  Scenario kept: {n} rows")

final_balanced_df = pd.concat(balanced_dfs, ignore_index=True)
print(f"\nTotal rows after balancing: {len(final_balanced_df)}")
print(f"Saving balanced CSV: {OUTPUT_CSV}")
final_balanced_df.to_csv(OUTPUT_CSV, index=False)

print("\n=== FINAL SCENARIO DISTRIBUTION ===")
scenario_summary = final_balanced_df['scenario_file'].value_counts()
print("Rows per scenario (top 10):")
print(scenario_summary.head(10))
print("\n✓ BALANCING COMPLETE. Use this file for ML label construction.")