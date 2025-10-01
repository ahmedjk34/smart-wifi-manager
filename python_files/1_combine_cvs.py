"""
FIXED: Robust CSV Combiner with Strict Column Enforcement
Handles malformed CSV rows by enforcing exactly 19 columns
Prevents terminal spam and invalid CSV generation

Author: ahmedjk34
Date: 2025-10-01 16:06:18 UTC
Version: 2.0 (EMERGENCY FIX)
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

# FIXED: Define EXACT expected columns (19 fields)
EXPECTED_COLUMNS = [
    'time', 'stationId', 'rateIdx', 'phyRate',
    'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
    'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',
    'shortSuccRatio', 'medSuccRatio', 'packetLossRate',
    'channelWidth', 'mobilityMetric',
    'severity', 'confidence', 'scenario_file'
]

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
        if len(header_cols) != 19:
            print(f"  ⚠️ {os.path.basename(filepath)}: {len(header_cols)} columns (expected 19) - FIXING...")
            
            # Read all lines
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Fix header
            fixed_header = ','.join(EXPECTED_COLUMNS) + '\n'
            
            # Fix data lines - keep only first 19 comma-separated values
            fixed_lines = [fixed_header]
            for i, line in enumerate(lines[1:], start=2):
                if not line.strip():
                    continue
                
                parts = line.strip().split(',')
                
                # Enforce exactly 19 columns
                if len(parts) < 19:
                    # Pad with empty strings
                    parts += [''] * (19 - len(parts))
                elif len(parts) > 19:
                    # Truncate to 19 (keep first 18 + last for scenario_file)
                    parts = parts[:18] + [parts[-1]]
                
                fixed_lines.append(','.join(parts) + '\n')
            
            # Write fixed file
            temp_file = filepath + '.fixed'
            with open(temp_file, 'w') as f:
                f.writelines(fixed_lines)
            
            # Replace original
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
            # Validate and fix CSV structure first
            success, error = validate_and_fix_csv(fpath)
            if not success:
                print(f"❌ {fname}: Validation failed - {error}")
                continue
            
            # Now safely read the file
            sample_df = pd.read_csv(fpath, nrows=5, low_memory=False, on_bad_lines='skip')
            
            # Count total rows
            with open(fpath, 'r') as f:
                row_count = sum(1 for line in f) - 1  # -1 for header
            
            file_info.append((fname, row_count, len(sample_df.columns), sample_df.columns.tolist()))
            print(f"✓ {fname}: {row_count:,} rows, {len(sample_df.columns)} cols")
            
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")
            
    return file_info

def clean_dataframe(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    """Clean and standardize dataframe with strict validation"""
    if df.empty:
        return df
    
    # FIXED: Ensure exactly 19 columns
    if len(df.columns) != 19:
        print(f"  ⚠️ DataFrame has {len(df.columns)} columns, enforcing 19...")
        
        # If too many columns, keep first 19
        if len(df.columns) > 19:
            df = df.iloc[:, :19]
        
        # Force correct column names
        df.columns = EXPECTED_COLUMNS
    
    # Set scenario_file column (last column)
    df['scenario_file'] = fname.replace('_detailed.csv', '')
    
    # Remove completely blank rows (excluding scenario_file)
    cols_to_check = EXPECTED_COLUMNS[:-1]  # All except scenario_file
    existing_cols = [col for col in cols_to_check if col in df.columns]
    
    if existing_cols:
        df = df.dropna(subset=existing_cols, how='all')
    
    # FIXED: Convert numeric columns with error suppression
    numeric_cols = [
        'time', 'stationId', 'rateIdx', 'phyRate',
        'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
        'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',
        'shortSuccRatio', 'medSuccRatio', 'packetLossRate',
        'channelWidth', 'mobilityMetric', 'severity', 'confidence'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None)
    
    return df

def process_files_in_chunks(log_dir: str, output_csv: str, chunk_size: int = 10000):
    """Process files in chunks to avoid memory issues"""
    
    # Get and validate file information
    file_info = get_file_info(log_dir)
    
    if not file_info:
        print("❌ No valid files to process!")
        return
    
    total_rows = sum(info[1] for info in file_info)
    print(f"\n📊 Total estimated rows: {total_rows:,}")
    
    # Sort files by size (process smaller ones first)
    file_info.sort(key=lambda x: x[1])
    
    # Initialize output
    header_written = False
    processed_rows = 0
    
    print("\n🔄 Processing files...\n")
    
    for fname, row_count, col_count, file_cols in file_info:
        fpath = os.path.join(log_dir, fname)
        print(f"Processing {fname} ({row_count:,} rows)...")
        
        try:
            # Process large files in chunks
            if row_count > chunk_size:
                chunk_iter = pd.read_csv(fpath, chunksize=chunk_size, low_memory=False, on_bad_lines='skip')
                
                for i, chunk in enumerate(chunk_iter):
                    chunk = clean_dataframe(chunk, fname)
                    
                    if chunk.empty:
                        continue
                    
                    # Write to output
                    write_mode = 'w' if not header_written else 'a'
                    chunk.to_csv(output_csv, mode=write_mode, 
                                index=False, header=not header_written)
                    header_written = True
                    processed_rows += len(chunk)
                    
                    print(f"  Chunk {i+1}: {len(chunk):,} rows (Total: {processed_rows:,})")
                    
                    del chunk
                    gc.collect()
            else:
                # Process smaller files normally
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
    
    print(f"\n✅ Successfully combined {processed_rows:,} rows into {output_csv}")

def process_with_sampling(log_dir: str, output_csv: str, sample_ratio: float = 0.1, max_samples_per_file: int = 10000):
    """Process with random sampling to reduce data size"""
    
    file_info = get_file_info(log_dir)
    
    if not file_info:
        print("❌ No valid files to process!")
        return
    
    print(f"\n📊 SAMPLING MODE (10% of data, max {max_samples_per_file:,} rows per file)\n")
    
    dfs = []
    total_original = 0
    total_sampled = 0
    
    for fname, row_count, col_count, file_cols in file_info:
        fpath = os.path.join(log_dir, fname)
        total_original += row_count
        
        try:
            if row_count > 5000:
                # Sample 10% of large files
                sample_size = min(int(row_count * sample_ratio), max_samples_per_file)
                df = pd.read_csv(fpath, low_memory=False, on_bad_lines='skip')
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                print(f"Sampled {len(df):,} rows from {fname} (10% of {row_count:,})")
            else:
                # Load all for small files
                df = pd.read_csv(fpath, low_memory=False, on_bad_lines='skip')
                print(f"Loaded all {len(df):,} rows from {fname}")
            
            df = clean_dataframe(df, fname)
            
            if not df.empty:
                dfs.append(df)
                total_sampled += len(df)
            
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
    
    if dfs:
        print(f"\n🔄 Combining {len(dfs)} dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        
        print(f"✅ Saved sampled data: {len(combined_df):,} rows")
        print(f"📊 Original: {total_original:,} → Sampled: {total_sampled:,} ({total_sampled/total_original*100:.1f}%)")
        
        combined_df.to_csv(output_csv, index=False)
        print(f"✅ Output written to: {os.path.abspath(output_csv)}")
    else:
        print("❌ No data to save!")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("FIXED CSV Combiner - Column Validation & Error Handling")
    print(f"Author: ahmedjk34")
    print(f"Date: 2025-10-01 16:06:18 UTC")
    print("="*60 + "\n")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ Directory '{LOG_DIR}' does not exist.")
        sys.exit(1)
    
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"🗑️  Removed existing output file\n")
    
    print("Choose processing method:")
    print("1. Chunk processing (memory efficient)")
    print("2. Sampling (10% of data, max 10K rows per file)")
    print("3. File info only (validation check)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        process_files_in_chunks(LOG_DIR, OUTPUT_CSV, chunk_size=5000)
    elif choice == "2":
        process_with_sampling(LOG_DIR, OUTPUT_CSV, sample_ratio=0.1, max_samples_per_file=10000)
    elif choice == "3":
        get_file_info(LOG_DIR)
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ PROCESS COMPLETE")
    print("="*60)