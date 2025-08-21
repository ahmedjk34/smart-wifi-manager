import pandas as pd
import numpy as np
import os
from collections import Counter

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smartv4-ml-ready.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smartv4-ml-balanced.csv")


# --- PARAMETERS ---
MIN_TARGET = 35000   # Minimum rows per scenario file (tune as needed)
MAX_TARGET = 50000   # Maximum rows per scenario file (tune as needed)
MIN_LABEL_COUNTS = 15000  # Minimum rows per class label within each scenario
RANDOM_SEED = 42

print(f"\n=== BALANCED DATASET SCRIPT ===")
print(f"Input file: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV, low_memory=False)
if 'scenario_file' not in df.columns:
    raise ValueError("Combined CSV must have a 'scenario_file' column!")

print(f"Total rows before balancing: {len(df)}")

# --- Remove empty/corrupt scenarios ---
scenario_counts = df['scenario_file'].value_counts()
valid_scenarios = scenario_counts[scenario_counts > 0].index.tolist()
df = df[df['scenario_file'].isin(valid_scenarios)]

print(f"Found {len(valid_scenarios)} valid scenario files.")

balanced_dfs = []
label_col = None

# --- Choose balancing label ---
for candidate in ['oracle_best_rateIdx', 'v3_rateIdx', 'packetSuccessLabel']:
    if candidate in df.columns:
        label_col = candidate
        break
if label_col is None:
    raise ValueError("No label column found for balancing! (oracle_best_rateIdx, v3_rateIdx, packetSuccessLabel)")

print(f"Balancing by scenario and label: '{label_col}'")

for fname in valid_scenarios:
    df_sub = df[df['scenario_file'] == fname]
    n = len(df_sub)
    print(f"\n--- {fname}: {n} rows ---")
    labels = df_sub[label_col].dropna().unique()
    label_counts = dict(Counter(df_sub[label_col].dropna()))
    print("Label counts:", label_counts)
    subframes = []

    for label in labels:
        label_df = df_sub[df_sub[label_col] == label]
        count = len(label_df)
        if count < MIN_LABEL_COUNTS:
            # Oversample rare label
            oversampled = label_df.sample(MIN_LABEL_COUNTS, replace=True, random_state=RANDOM_SEED)
            subframes.append(oversampled)
            print(f"  Oversampled label {label}: {count} -> {MIN_LABEL_COUNTS}")
        elif count > MAX_TARGET // len(labels):
            # Downsample dominant label
            downsampled = label_df.sample(MAX_TARGET // len(labels), replace=False, random_state=RANDOM_SEED)
            subframes.append(downsampled)
            print(f"  Downsampled label {label}: {count} -> {MAX_TARGET // len(labels)}")
        else:
            subframes.append(label_df)
            print(f"  Kept label {label}: {count} rows")

    balanced_scenario_df = pd.concat(subframes, ignore_index=True)
    # Now overall up/downsample scenario if needed
    total_rows = len(balanced_scenario_df)
    if total_rows < MIN_TARGET:
        oversampled = balanced_scenario_df.sample(MIN_TARGET, replace=True, random_state=RANDOM_SEED)
        balanced_dfs.append(oversampled)
        print(f"  Scenario oversampled: {total_rows} -> {MIN_TARGET}")
    elif total_rows > MAX_TARGET:
        downsampled = balanced_scenario_df.sample(MAX_TARGET, replace=False, random_state=RANDOM_SEED)
        balanced_dfs.append(downsampled)
        print(f"  Scenario downsampled: {total_rows} -> {MAX_TARGET}")
    else:
        balanced_dfs.append(balanced_scenario_df)
        print(f"  Scenario kept: {total_rows} rows")

final_balanced_df = pd.concat(balanced_dfs, ignore_index=True)
print(f"\nTotal rows after balancing: {len(final_balanced_df)}")
print(f"Saving balanced CSV: {OUTPUT_CSV}")
final_balanced_df.to_csv(OUTPUT_CSV, index=False)

# --- Deep label balancing summary ---
print("\n=== FINAL LABEL AND SCENARIO DISTRIBUTION ===")
scenario_summary = final_balanced_df['scenario_file'].value_counts()
print("Rows per scenario (top 10):")
print(scenario_summary.head(10))
if label_col in final_balanced_df.columns:
    label_summary = final_balanced_df[label_col].value_counts()
    print(f"\nRows per label '{label_col}':")
    print(label_summary)
else:
    print("Label column missing in final output.")

# --- Optionally: Show cross-distribution ---
cross_dist = final_balanced_df.groupby(['scenario_file', label_col]).size().unstack(fill_value=0)
print("\nCross-distribution scenario vs label (top 5 scenarios):")
print(cross_dist.head(5))
print("\n✓ BALANCING COMPLETE. Use this file for ML label construction.")