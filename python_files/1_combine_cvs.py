"""
FIXED: Robust CSV Combiner (NO BALANCING)
Handles malformed CSV rows safely

PHASE 1A UPDATE (2025-10-03):
- Now expects 25 columns (was 19)
- Added 6 new features: retryRate, frameErrorRate, channelBusyRatio,
  recentRateAvg, rateStability, sinceLastChange

PHASE 1B UPDATE (2025-10-03):
- Now expects 29 columns (was 25)
- Added 4 new features: rssiVariance, interferenceLevel,
  distanceMetric, avgPacketSize

Author: ahmedjk34
Date: 2025-10-03 08:45:00 UTC (PHASE 1B UPDATE)
Version: 3.1 (NO BALANCING)
"""

import os
import sys
import pandas as pd
import gc
import warnings
from typing import List, Tuple

# Suppress ALL warnings before anything else
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
pd.set_option('mode.copy_on_write', True)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = os.path.join(PARENT_DIR, "balanced-results")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")

EXPECTED_COLUMNS = [
    # Metadata (4)
    'time', 'stationId', 'rateIdx', 'phyRate',

    # SNR features (7)
    'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
    'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',

    # Previous window success (2)
    'shortSuccRatio', 'medSuccRatio',

    # Previous window loss (1)
    'packetLossRate',

    # Network state (2)
    'channelWidth', 'mobilityMetric',

    # Assessment (2)
    'severity', 'confidence',

    # PHASE 1A features (6)
    'retryRate', 'frameErrorRate', 'channelBusyRatio',
    'recentRateAvg', 'rateStability', 'sinceLastChange',

    # PHASE 1B features (4)
    'rssiVariance', 'interferenceLevel', 'distanceMetric', 'avgPacketSize',

    # Scenario identifier (1)
    'scenario_file'
]

# TOTAL: 29 columns


def validate_and_fix_csv(filepath: str) -> Tuple[bool, str]:
    """Validate and fix malformed CSVs."""
    try:
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
            header_cols = header_line.split(',')

        if len(header_cols) != 29:
            print(f"  ⚠️ {os.path.basename(filepath)}: {len(header_cols)} columns (expected 29) - FIXING...")

            with open(filepath, 'r') as f:
                lines = f.readlines()

            fixed_header = ','.join(EXPECTED_COLUMNS) + '\n'
            fixed_lines = [fixed_header]

            for i, line in enumerate(lines[1:], start=2):
                if not line.strip():
                    continue

                parts = line.strip().split(',')
                if len(parts) < 29:
                    parts += [''] * (29 - len(parts))
                elif len(parts) > 29:
                    parts = parts[:28] + [parts[-1]]

                fixed_lines.append(','.join(parts) + '\n')

            temp_file = filepath + '.fixed'
            with open(temp_file, 'w') as f:
                f.writelines(fixed_lines)

            os.replace(temp_file, filepath)
            print(f"  ✅ Fixed {os.path.basename(filepath)}")

        return True, ""

    except Exception as e:
        return False, str(e)


def get_file_info(log_dir: str) -> List[tuple]:
    """Get file information without loading full files."""
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    file_info = []

    print(f"📂 Found {len(csv_files)} CSV files\n")

    for fname in csv_files:
        fpath = os.path.join(log_dir, fname)

        try:
            success, error = validate_and_fix_csv(fpath)
            if not success:
                print(f"❌ {fname}: Validation failed - {error}")
                continue

            sample_df = pd.read_csv(fpath, nrows=5, low_memory=False, on_bad_lines='skip')

            with open(fpath, 'r') as f:
                row_count = sum(1 for line in f) - 1

            file_info.append((fname, row_count, len(sample_df.columns), sample_df.columns.tolist()))
            print(f"✓ {fname}: {row_count:,} rows, {len(sample_df.columns)} cols")

        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")

    return file_info


def clean_dataframe(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    """Clean and standardize dataframe."""
    if df.empty:
        return df

    if len(df.columns) > 29:
        df = df.iloc[:, :29]

    df['scenario_file'] = fname.replace('_detailed.csv', '')

    cols_to_check = EXPECTED_COLUMNS[:-1]
    existing_cols = [col for col in cols_to_check if col in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols, how='all')

    numeric_cols = [
        'time', 'stationId', 'rateIdx', 'phyRate',
        'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
        'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',
        'shortSuccRatio', 'medSuccRatio', 'packetLossRate',
        'channelWidth', 'mobilityMetric', 'severity', 'confidence',
        'retryRate', 'frameErrorRate', 'channelBusyRatio',
        'recentRateAvg', 'rateStability', 'sinceLastChange',
        'rssiVariance', 'interferenceLevel', 'distanceMetric', 'avgPacketSize'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None)

    return df


def process_files_in_chunks(log_dir: str, output_csv: str, chunk_size: int = 10000):
    """Process files in chunks (no balancing)."""
    file_info = get_file_info(log_dir)
    if not file_info:
        print("❌ No valid files to process!")
        return

    total_rows = sum(info[1] for info in file_info)
    print(f"\n📊 Total estimated rows: {total_rows:,}")

    file_info.sort(key=lambda x: x[1])

    header_written = False
    processed_rows = 0

    print("\n🔄 Processing files...\n")

    for fname, row_count, col_count, file_cols in file_info:
        fpath = os.path.join(log_dir, fname)
        print(f"Processing {fname} ({row_count:,} rows)...")

        try:
            if row_count > chunk_size:
                chunk_iter = pd.read_csv(fpath, chunksize=chunk_size,
                                         low_memory=False, on_bad_lines='skip')
                for i, chunk in enumerate(chunk_iter):
                    chunk = clean_dataframe(chunk, fname)
                    if chunk.empty:
                        continue

                    write_mode = 'w' if not header_written else 'a'
                    chunk.to_csv(output_csv, mode=write_mode,
                                 index=False, header=not header_written)
                    header_written = True
                    processed_rows += len(chunk)

                    print(f"  Chunk {i+1}: {len(chunk):,} rows (Total: {processed_rows:,})")

                    del chunk
                    gc.collect()
            else:
                df = pd.read_csv(fpath, low_memory=False, on_bad_lines='skip')
                df = clean_dataframe(df, fname)
                if not df.empty:
                    write_mode = 'w' if not header_written else 'a'
                    df.to_csv(output_csv, mode=write_mode,
                              index=False, header=not header_written)
                    header_written = True
                    processed_rows += len(df)
                    print(f"  Added {len(df):,} rows (Total: {processed_rows:,})")
                del df
                gc.collect()

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            continue

    print(f"\n✅ Combined {processed_rows:,} rows into {output_csv}")


def process_all_files(log_dir: str, output_csv: str):
    """Load all files, combine directly (no balancing)."""
    file_info = get_file_info(log_dir)
    if not file_info:
        print("❌ No valid files to process!")
        return

    dfs = []
    total_original = 0

    for fname, row_count, col_count, file_cols in file_info:
        fpath = os.path.join(log_dir, fname)
        total_original += row_count
        try:
            df = pd.read_csv(fpath, low_memory=False, on_bad_lines='skip')
            df = clean_dataframe(df, fname)
            if not df.empty:
                dfs.append(df)
                print(f"Loaded {len(df):,} rows from {fname}")
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

    if dfs:
        print(f"\n🔄 Combining {len(dfs)} dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        print(f"\n📊 Combined dataset: {len(combined_df):,} rows")

        combined_df.to_csv(output_csv, index=False)
        print(f"\n✅ Output written to: {os.path.abspath(output_csv)}")
        print(f"📊 Original: {total_original:,} → Final: {len(combined_df):,} rows")
    else:
        print("❌ No data to save!")


if __name__ == "__main__":
    print("="*60)
    print("FIXED CSV Combiner (NO BALANCING)")
    print("Author: ahmedjk34")
    print("Date: 2025-10-03 08:45:00 UTC")
    print("Version: 3.1 (NO BALANCING)")
    print("="*60 + "\n")

    if not os.path.exists(LOG_DIR):
        print(f"❌ Directory '{LOG_DIR}' does not exist.")
        sys.exit(1)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"🗑️  Removed existing output file\n")

    print("Choose processing method:")
    print("1. Chunk processing (memory efficient)")
    print("2. Load all & combine (RECOMMENDED)")
    print("3. File info only (validation check)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        process_files_in_chunks(LOG_DIR, OUTPUT_CSV, chunk_size=5000)
    elif choice == "2":
        process_all_files(LOG_DIR, OUTPUT_CSV)
    elif choice == "3":
        get_file_info(LOG_DIR)
    else:
        print("❌ Invalid choice")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ PROCESS COMPLETE")
    print("="*60)
