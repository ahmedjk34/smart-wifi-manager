import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = os.path.join(PARENT_DIR, "balanced-results")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")

# Check if the directory exists before proceeding
if not os.path.exists(LOG_DIR):
    print(f"Error: Directory '{LOG_DIR}' does not exist.")
    print("Please check if:")
    print("1. The 'logged-results' folder exists in the same directory as this script")
    print("2. You're running the script from the correct location")
    print("3. The folder name is spelled correctly")
    exit(1)

# Check if directory is empty
csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.csv')]

if not csv_files:
    print(f"No CSV files found in '{LOG_DIR}'")
    exit(1)

print(f"Found {len(csv_files)} CSV files in '{LOG_DIR}'")

dfs = []
bad_files = []
for fname in csv_files:
    fpath = os.path.join(LOG_DIR, fname)
    try:
        # Try multiple parsing strategies for corrupted CSVs
        df = None
        
        # Strategy 1: Standard parsing with error handling
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except pd.errors.ParserError:
            # Strategy 2: Skip bad lines
            try:
                df = pd.read_csv(fpath, on_bad_lines='skip', low_memory=False)
                print(f"Warning: {fname} had bad lines that were skipped")
            except:
                # Strategy 3: Use python engine (slower but more robust)
                try:
                    df = pd.read_csv(fpath, engine='python', on_bad_lines='skip', low_memory=False)
                    print(f"Warning: {fname} required python engine with line skipping")
                except:
                    raise Exception("All parsing strategies failed")
        
        if df is None or df.empty:
            print(f"Warning: {fname} is empty")
            bad_files.append(fname)
            continue

        # --- PATCH: Drop blank rows (all columns except scenario_file are NaN/empty) ---
        cols_to_check = [col for col in df.columns if col != 'scenario_file']
        df_clean = df.dropna(subset=cols_to_check, how='all')
        # Also drop rows that are only empty strings except scenario_file
        df_clean = df_clean.loc[
            ~(df_clean[cols_to_check].apply(lambda row: all(
                (pd.isna(x) or (isinstance(x, str) and x.strip() == "")) for x in row
            ), axis=1))
        ]
        if df_clean.empty:
            print(f"Warning: {fname} only contained blank rows, skipping.")
            bad_files.append(fname)
            continue

        df_clean['scenario_file'] = fname
        dfs.append(df_clean)
        print(f"Successfully loaded: {fname} ({df_clean.shape[0]} rows, {df_clean.shape[1]} cols)")
        
    except Exception as e:
        print(f"Skipping {fname}: {str(e)[:100]}...")
        bad_files.append(fname)

if dfs:
    print(f"\n--- COMBINING DATA ---")
    
    # Check if all dataframes have compatible columns before combining
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    
    print(f"Total unique columns found: {len(all_columns)}")
    
    # Align all dataframes to have the same columns
    for i, df in enumerate(dfs):
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df[col] = None  # Fill missing columns with None
    
    combined_df = pd.concat(dfs, ignore_index=True, sort=False)

    # --- PATCH: Drop blank rows after combining (extra safety) ---
    cols_to_check = [col for col in combined_df.columns if col != 'scenario_file']
    combined_df = combined_df.dropna(subset=cols_to_check, how='all')
    combined_df = combined_df.loc[
        ~(combined_df[cols_to_check].apply(lambda row: all(
            (pd.isna(x) or (isinstance(x, str) and x.strip() == "")) for x in row
        ), axis=1))
    ]

    # Try to save with error handling
    try:
        combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✓ Combined {len(dfs)} files into {OUTPUT_CSV}")
    except PermissionError:
        # Try with a different filename if the original is locked
        import time
        timestamp = int(time.time())
        backup_name = os.path.join(PARENT_DIR, f"smart-v3-logged-ALL-{timestamp}.csv")
        try:
            combined_df.to_csv(backup_name, index=False)
            print(f"✓ Original file was locked, saved as: {backup_name}")
        except Exception as e:
            print(f"✗ Failed to save file: {e}")
            print("Make sure no programs have the output file open and you have write permissions")
            exit(1)
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        exit(1)
    print(f"✓ Total rows: {combined_df.shape[0]}")
    print(f"✓ Total columns: {combined_df.shape[1]}")
    
    # Show summary statistics
    print(f"\n--- SUMMARY BY FILE TYPE ---")
    file_summary = combined_df['scenario_file'].value_counts()
    print(f"Files with most data:")
    for fname, count in file_summary.head(5).items():
        print(f"  {fname}: {count} rows")
        
else:
    print("No valid CSV files found to combine.")

if bad_files:
    print(f"\n⚠ Skipped {len(bad_files)} bad/empty/corrupt files:")
    for f in bad_files:
        print("  ", f)