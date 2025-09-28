import os
import pandas as pd
import gc
from typing import List, Iterator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = os.path.join(PARENT_DIR, "balanced-results")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")

def get_file_info(log_dir: str) -> List[tuple]:
    """Get file information without loading the full files"""
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    file_info = []
    
    for fname in csv_files:
        fpath = os.path.join(log_dir, fname)
        try:
            # Read just the first few rows to get column info and estimate size
            sample_df = pd.read_csv(fpath, nrows=5, low_memory=False)
            
            # Count total rows (more memory efficient than loading all)
            with open(fpath, 'r') as f:
                row_count = sum(1 for line in f) - 1  # -1 for header
            
            file_info.append((fname, row_count, len(sample_df.columns), sample_df.columns.tolist()))
            print(f"{fname}: {row_count:,} rows, {len(sample_df.columns)} cols")
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            
    return file_info

def process_files_in_chunks(log_dir: str, output_csv: str, chunk_size: int = 10000):
    """Process files in chunks to avoid memory issues"""
    
    # Get file information first
    file_info = get_file_info(log_dir)
    total_rows = sum(info[1] for info in file_info)
    print(f"\nTotal estimated rows: {total_rows:,}")
    
    # Get all unique columns
    all_columns = set()
    for _, _, _, cols in file_info:
        all_columns.update(cols)
    all_columns = sorted(all_columns)
    print(f"Total unique columns: {len(all_columns)}")
    
    # Sort files by size (process smaller ones first)
    file_info.sort(key=lambda x: x[1])
    
    # Initialize output file with headers
    header_written = False
    processed_rows = 0
    
    for fname, row_count, col_count, file_cols in file_info:
        fpath = os.path.join(log_dir, fname)
        print(f"\nProcessing {fname} ({row_count:,} rows)...")
        
        try:
            # Process large files in chunks
            if row_count > chunk_size:
                chunk_iter = pd.read_csv(fpath, chunksize=chunk_size, low_memory=False)
                
                for i, chunk in enumerate(chunk_iter):
                    # Clean the chunk
                    chunk = clean_dataframe(chunk, fname, all_columns)
                    
                    if chunk.empty:
                        continue
                        
                    # Write to output
                    write_mode = 'w' if not header_written else 'a'
                    chunk.to_csv(output_csv, mode=write_mode, 
                                index=False, header=not header_written)
                    header_written = True
                    processed_rows += len(chunk)
                    
                    print(f"  Chunk {i+1}: {len(chunk):,} rows (Total: {processed_rows:,})")
                    
                    # Force garbage collection
                    del chunk
                    gc.collect()
            else:
                # Process smaller files normally
                df = pd.read_csv(fpath, low_memory=False)
                df = clean_dataframe(df, fname, all_columns)
                
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
            print(f"Error processing {fname}: {e}")
            continue
    
    print(f"\n✓ Successfully combined {processed_rows:,} rows into {output_csv}")

def clean_dataframe(df: pd.DataFrame, fname: str, all_columns: List[str]) -> pd.DataFrame:
    """Clean and standardize dataframe"""
    if df.empty:
        return df
    
    # Add scenario_file column
    df['scenario_file'] = fname
    
    # Remove blank rows (excluding scenario_file)
    cols_to_check = [col for col in df.columns if col != 'scenario_file']
    if cols_to_check:
        df = df.dropna(subset=cols_to_check, how='all')
        df = df.loc[
            ~(df[cols_to_check].apply(lambda row: all(
                (pd.isna(x) or (isinstance(x, str) and x.strip() == "")) for x in row
            ), axis=1))
        ]
    
    # Align columns
    for col in all_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[all_columns]
    
    return df

def process_with_sampling(log_dir: str, output_csv: str, sample_ratio: float = 0.1):
    """Process with random sampling to reduce data size"""
    file_info = get_file_info(log_dir)
    
    print(f"\n=== SAMPLING MODE (taking {sample_ratio*100}% of data) ===")
    
    dfs = []
    for fname, row_count, col_count, file_cols in file_info:
        fpath = os.path.join(log_dir, fname)
        
        try:
            if row_count > 5000:  # Only sample large files
                sample_size = max(int(row_count * sample_ratio), 100)  # At least 100 rows
                df = pd.read_csv(fpath, low_memory=False).sample(n=min(sample_size, row_count))
                print(f"Sampled {len(df):,} rows from {fname}")
            else:
                df = pd.read_csv(fpath, low_memory=False)
                print(f"Loaded all {len(df):,} rows from {fname}")
            
            df['scenario_file'] = fname
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    
    if dfs:
        # Get all columns
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        
        # Align columns
        for df in dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None
        
        # Filter out empty dataframes before concatenating
        non_empty_dfs = [df for df in dfs if not df.empty]
        if non_empty_dfs:
            combined_df = pd.concat(non_empty_dfs, ignore_index=True, sort=False)
        else:
            combined_df = pd.DataFrame()
        combined_df.to_csv(output_csv.replace('.csv', '_sampled.csv'), index=False)
        print(f"✓ Saved sampled data: {len(combined_df):,} rows")

# Main execution
if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        print(f"Error: Directory '{LOG_DIR}' does not exist.")
        exit(1)
    
    print("Choose processing method:")
    print("1. Chunk processing (memory efficient)")
    print("2. Sampling (10% of data)")
    print("3. File info only")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        process_files_in_chunks(LOG_DIR, OUTPUT_CSV, chunk_size=5000)
    elif choice == "2":
        process_with_sampling(LOG_DIR, OUTPUT_CSV, sample_ratio=0.1)
    elif choice == "3":
        get_file_info(LOG_DIR)
    else:
        print("Invalid choice")