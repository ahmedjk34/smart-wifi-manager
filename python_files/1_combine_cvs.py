"""
FIXED: Robust CSV Combiner with Rate-Balanced Sampling
Handles malformed CSV rows AND balances samples across rate classes

CRITICAL FIX: Implements stratified sampling per rate to prevent high-rate domination
- Equal-time simulation generates more high-rate samples
- This combiner ensures balanced training data (15K samples per rate)

PHASE 1A UPDATE (2025-10-03):
- Now expects 25 columns (was 19)
- Added 6 new features: retryRate, frameErrorRate, channelBusyRatio,
  recentRateAvg, rateStability, sinceLastChange

Author: ahmedjk34
Date: 2025-10-03 08:45:00 UTC (PHASE 1A UPDATE)
Version: 3.1 (PHASE 1A COMPATIBLE)
"""

import os
import sys
import pandas as pd
import gc
import warnings
from typing import List, Iterator, Tuple

# CRITICAL: Suppress ALL warnings before anything else
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
pd.set_option('mode.copy_on_write', True)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = os.path.join(PARENT_DIR, "balanced-results")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")
RANDOM_SEED = 42


# PHASE 1B UPDATE (2025-10-03): Now 29 columns (was 25)
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
    
    # PHASE 1A: New features (6)
    'retryRate', 'frameErrorRate', 'channelBusyRatio',
    'recentRateAvg', 'rateStability', 'sinceLastChange',
    
    # PHASE 1B: NEW FEATURES (4)
    'rssiVariance', 'interferenceLevel', 'distanceMetric', 'avgPacketSize',
    
    # Scenario identifier (1)
    'scenario_file'
]  # TOTAL: 29 columns (4 metadata + 24 features + 1 scenario)
def validate_and_fix_csv(filepath: str) -> Tuple[bool, str]:
    """
    Validate CSV has exactly 19 columns, fix if needed
    Returns: (success, error_message)
    """
    try:
        # Read header line
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
            header_cols = header_line.split(',')
        
        # Check column count
        if len(header_cols) != 29:
            print(f"  ⚠️ {os.path.basename(filepath)}: {len(header_cols)} columns (expected 29) - FIXING...")
            
            # Read all lines
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Fix header
            fixed_header = ','.join(EXPECTED_COLUMNS) + '\n'
            
            # Fix data lines
            fixed_lines = [fixed_header]
            for i, line in enumerate(lines[1:], start=2):
                if not line.strip():
                    continue
                
                parts = line.strip().split(',')
                
            if len(parts) < 29:
                parts += [''] * (29 - len(parts))
            elif len(parts) > 29:
                parts = parts[:28] + [parts[-1]]  # Keep scenario_file as last column

                            
                fixed_lines.append(','.join(parts) + '\n')
            
            # Write fixed file
            temp_file = filepath + '.fixed'
            with open(temp_file, 'w') as f:
                f.writelines(fixed_lines)
            
            os.replace(temp_file, filepath)
            print(f"  ✅ Fixed {os.path.basename(filepath)}")
        
        return True, ""
        
    except Exception as e:
        return False, str(e)

def get_file_info(log_dir: str) -> List[tuple]:
    """Get file information without loading full files"""
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
    """Clean and standardize dataframe"""
    if df.empty:
        return df
    
    # Ensure exactly 19 columns
    if len(df.columns) != 25:
        if len(df.columns) > 25:
            df = df.iloc[:, :25]
        # Set scenario_file
        df['scenario_file'] = fname.replace('_detailed.csv', '')
    
    # Remove blank rows
    cols_to_check = EXPECTED_COLUMNS[:-1]
    existing_cols = [col for col in cols_to_check if col in df.columns]
    
    if existing_cols:
        df = df.dropna(subset=existing_cols, how='all')
    
    # Convert numeric columns
    numeric_cols = [
        'time', 'stationId', 'rateIdx', 'phyRate',
        'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
        'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',
        'shortSuccRatio', 'medSuccRatio', 'packetLossRate',
        'channelWidth', 'mobilityMetric', 'severity', 'confidence',
        # PHASE 1A features (6)
        'retryRate', 'frameErrorRate', 'channelBusyRatio',
        'recentRateAvg', 'rateStability', 'sinceLastChange',
        # PHASE 1B features (4)  ← ADD THIS
        'rssiVariance', 'interferenceLevel', 'distanceMetric', 'avgPacketSize'
    ]
        
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None)
    
    return df

def balance_samples_per_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Balance samples across rate classes
    
    Even with equal simulation time, high rates generate more packets.
    This ensures balanced training data across all 8 rate classes.
    
    Strategy:
    - Identify minimum samples per rate
    - Target: 3x minimum or 15K per rate (whichever is lower)
    - Undersample over-represented classes
    - Keep all samples from under-represented classes
    """
    print("\n🔧 Balancing samples across rate classes...")
    print("="*60)
    
    if 'rateIdx' not in df.columns:
        print("⚠️ rateIdx column not found - skipping balancing")
        return df
    
    # Get current distribution
    rate_counts = df['rateIdx'].value_counts().sort_index()
    print("📊 Original distribution:")
    for rate, count in rate_counts.items():
        pct = (count / len(df)) * 100
        print(f"   Rate {rate}: {count:,} samples ({pct:.1f}%)")
    
    # Determine target samples per rate
    min_samples = rate_counts.min()
    max_samples = rate_counts.max()
    
    # Target: 3x minimum, capped at 15K
    target_per_rate = min(min_samples * 3, 15000)
    
    # If minimum is too low, use 10K as absolute minimum target
    if target_per_rate < 5000:
        target_per_rate = min(10000, max_samples)
    
    print(f"\n🎯 Target samples per rate: {target_per_rate:,}")
    print(f"   Min in dataset: {min_samples:,}")
    print(f"   Max in dataset: {max_samples:,}")
    print(f"   Imbalance ratio: {max_samples/min_samples:.1f}x")
    
    # Stratified sampling per rate
    balanced_dfs = []
    
    for rate in range(8):
        rate_df = df[df['rateIdx'] == rate].copy()
        original_count = len(rate_df)
        
        if original_count == 0:
            print(f"⚠️ Rate {rate}: NO SAMPLES - skipping")
            continue
        
        if original_count > target_per_rate:
            # Undersample high-rate classes
            rate_df = rate_df.sample(n=target_per_rate, random_state=RANDOM_SEED)
            reduction_pct = ((original_count - target_per_rate) / original_count) * 100
            print(f"   Rate {rate}: {original_count:,} → {target_per_rate:,} "
                 f"({reduction_pct:.0f}% reduced)")
        else:
            # Keep all samples from low-rate classes
            print(f"   Rate {rate}: {original_count:,} (kept all)")
        
        balanced_dfs.append(rate_df)
    
    # Combine balanced data
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle to prevent sequential bias
    balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Report final distribution
    final_counts = balanced_df['rateIdx'].value_counts().sort_index()
    final_imbalance = final_counts.max() / final_counts.min()
    
    print(f"\n✅ Balanced dataset created:")
    print(f"   Total samples: {len(df):,} → {len(balanced_df):,} "
         f"({len(balanced_df)/len(df)*100:.1f}% retained)")
    print(f"   Imbalance ratio: {max_samples/min_samples:.1f}x → {final_imbalance:.1f}x")
    print(f"\n📊 Final distribution:")
    
    for rate, count in final_counts.items():
        pct = (count / len(balanced_df)) * 100
        print(f"   Rate {rate}: {count:,} samples ({pct:.1f}%)")
    
    print("="*60)
    
    return balanced_df

def process_files_in_chunks(log_dir: str, output_csv: str, chunk_size: int = 10000):
    """Process files in chunks (for very large datasets)"""
    
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
    
    print(f"\n✅ Combined {processed_rows:,} rows")
    
    # CRITICAL: Now balance the combined file!
    print("\n🔄 Loading combined file for balancing...")
    combined_df = pd.read_csv(output_csv, low_memory=False)
    
    balanced_df = balance_samples_per_rate(combined_df)
    
    balanced_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved balanced data: {len(balanced_df):,} rows → {output_csv}")

def process_with_smart_balancing(log_dir: str, output_csv: str):
    """
    RECOMMENDED: Load all files, then apply smart rate balancing
    """
    
    file_info = get_file_info(log_dir)
    
    if not file_info:
        print("❌ No valid files to process!")
        return
    
    total_rows = sum(info[1] for info in file_info)
    estimated_memory_mb = (total_rows * 19 * 8) / (1024 * 1024) * 3  # 3x overhead
    
    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    print(f"\n📊 Memory estimation:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Estimated RAM needed: {estimated_memory_mb:.0f} MB")
    print(f"   Available RAM: {available_memory_mb:.0f} MB")
    
    # Safety check
    if estimated_memory_mb > available_memory_mb * 0.6:
        print(f"\n⚠️ WARNING: Estimated memory usage too high!")
        print(f"   Switching to chunk processing (Option 1)...")
        process_files_in_chunks(log_dir, output_csv, chunk_size=5000)
        return
    
    print(f"\n📊 SMART BALANCING MODE (equal samples per rate, FULL DATASET)\n")
    
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
        
        # CRITICAL: Apply smart balancing
        balanced_df = balance_samples_per_rate(combined_df)
        
        balanced_df.to_csv(output_csv, index=False)
        print(f"\n✅ Output written to: {os.path.abspath(output_csv)}")
        print(f"📊 Original: {total_original:,} → Final: {len(balanced_df):,} "
              f"({len(balanced_df)/total_original*100:.1f}%)")
    else:
        print("❌ No data to save!")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("FIXED CSV Combiner - Smart Rate Balancing")
    print(f"Author: ahmedjk34")
    print(f"Date: 2025-10-01 18:24:49 UTC")
    print(f"Version: 3.0 (SMART BALANCING)")
    print("="*60 + "\n")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ Directory '{LOG_DIR}' does not exist.")
        sys.exit(1)
    
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"🗑️  Removed existing output file\n")
    
    print("Choose processing method:")
    print("1. Chunk processing + smart balancing (memory efficient)")
    print("2. Smart balancing (RECOMMENDED - load all → balance by rate)")
    print("3. File info only (validation check)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        process_files_in_chunks(LOG_DIR, OUTPUT_CSV, chunk_size=5000)
    elif choice == "2":
        try:
            import psutil
            process_with_smart_balancing(LOG_DIR, OUTPUT_CSV)
        except ImportError:
            print("⚠️ psutil not installed, using regular load all...")
            process_with_smart_balancing(LOG_DIR, OUTPUT_CSV)
    elif choice == "3":
        get_file_info(LOG_DIR)
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ PROCESS COMPLETE")
    print("="*60)