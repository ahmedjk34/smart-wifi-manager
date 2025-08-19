import pandas as pd
import numpy as np
import os
import logging
import sys
from typing import Union

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOGGING SETUP ---
def setup_logging():
    """Setup logging with fallback if file creation fails"""
    handlers = []
    
    # Try to create file handler in script directory
    try:
        log_file = os.path.join(BASE_DIR, 'ml_data_prep.log')
        handlers.append(logging.FileHandler(log_file))
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not create log file ({e}), logging to console only")
    
    # Always add console handler
    handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()
INPUT_CSV = os.path.join(BASE_DIR, "..", "smart-v3-logged-ALL.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-ready.csv")

# List of available rates in bps (update as per your PHY config)
ALL_PHY_RATES = [
    1000000, 2000000, 5500000, 6000000, 9000000, 11000000,
    12000000, 18000000, 24000000, 36000000, 48000000, 54000000
]

# SNR thresholds for each rate for success estimation (tune for your environment)
SNR_THRESHOLDS = {
    1000000: 3, 2000000: 5, 5500000: 7, 6000000: 8, 9000000: 10,
    11000000: 12, 12000000: 13, 18000000: 15, 24000000: 18,
    36000000: 20, 48000000: 22, 54000000: 24
}

# --- FUNCTIONS ---

def estimate_p_success(row, rate: int) -> float:
    """Estimate probability of success for a given rate"""
    try:
        # For current rate, use shortSuccRatio; for others, use SNR threshold model
        if int(row['phyRate']) == rate:
            return float(row['shortSuccRatio'])
        
        if rate not in SNR_THRESHOLDS:
            logger.warning(f"Rate {rate} not found in SNR_THRESHOLDS, using default threshold")
            return 0.0
            
        return float(row['lastSnr'] >= SNR_THRESHOLDS[rate])
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error in estimate_p_success for rate {rate}: {e}")
        return 0.0

def find_oracle_best_rate(row) -> int:
    """Returns rate index that maximizes expected goodput (p_success x rate)"""
    try:
        best_rate = ALL_PHY_RATES[0]
        best_goodput = 0.0
        
        for rate in ALL_PHY_RATES:
            p_succ = estimate_p_success(row, rate)
            goodput = p_succ * rate
            if goodput > best_goodput:
                best_goodput = goodput
                best_rate = rate
        
        return ALL_PHY_RATES.index(best_rate)
    except (ValueError, IndexError) as e:
        logger.error(f"Error in find_oracle_best_rate: {e}")
        return 0  # Default to first rate index

def find_oracle_success(row) -> int:
    """Returns 1 if expected p_success at best rate > 0.5, else 0"""
    try:
        best_rate_idx = find_oracle_best_rate(row)
        rate = ALL_PHY_RATES[best_rate_idx]
        p_succ = estimate_p_success(row, rate)
        return int(p_succ > 0.5)
    except Exception as e:
        logger.error(f"Error in find_oracle_success: {e}")
        return 0

def filter_row(row) -> bool:
    """Filter function for removing noisy/ambiguous rows"""
    try:
        return float(row['lastSnr']) >= 3
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Error in filter_row, excluding row: {e}")
        return False

def validate_input_file(filepath: str) -> bool:
    """Validate input CSV file exists and is readable"""
    if not os.path.exists(filepath):
        logger.error(f"Input file does not exist: {filepath}")
        return False
    
    if not os.access(filepath, os.R_OK):
        logger.error(f"Input file is not readable: {filepath}")
        return False
    
    logger.info(f"Input file validated: {filepath}")
    return True

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate dataframe has required columns"""
    required_cols = [
        'lastSnr', 'shortSuccRatio', 'phyRate', 'rateIdx', 'packetSuccess'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    logger.info(f"DataFrame validation passed. Shape: {df.shape}")
    return True

def main():
    """Main execution function with comprehensive error handling"""
    try:
        logger.info("Starting ML data preparation script")
        logger.info(f"Input file: {INPUT_CSV}")
        logger.info(f"Output file: {OUTPUT_CSV}")
        
        # Validate input file
        if not validate_input_file(INPUT_CSV):
            sys.exit(1)
        
        # --- LOAD DATA ---
        logger.info("Loading input CSV...")
        try:
            df = pd.read_csv(INPUT_CSV)
            logger.info(f"Successfully loaded {len(df)} rows from input CSV")
        except pd.errors.EmptyDataError:
            logger.error("Input CSV is empty")
            sys.exit(1)
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error loading CSV: {e}")
            sys.exit(1)
        
        # Validate dataframe structure
        if not validate_dataframe(df):
            sys.exit(1)
        
        # --- FILTER NOISY/AMBIGUOUS ROWS ---
        logger.info("Filtering noisy/ambiguous rows...")
        initial_count = len(df)
        df = df[df.apply(filter_row, axis=1)]
        filtered_count = len(df)
        logger.info(f"Filtered {initial_count - filtered_count} rows, {filtered_count} remaining")
        
        if filtered_count == 0:
            logger.error("No rows remaining after filtering")
            sys.exit(1)
        
        # --- LABEL CONSTRUCTION ---
        logger.info("Constructing labels...")
        
        try:
            # 1. Oracle best rate (index)
            logger.info("Computing oracle_best_rateIdx...")
            df['oracle_best_rateIdx'] = df.apply(find_oracle_best_rate, axis=1)
            
            # 2. V3 imitation label (actual index chosen by algorithm)
            logger.info("Computing v3_rateIdx...")
            # Handle NaN values in rateIdx
            df['v3_rateIdx'] = pd.to_numeric(df['rateIdx'], errors='coerce').fillna(0).astype(int)
            nan_rateidx_count = df['rateIdx'].isna().sum()
            if nan_rateidx_count > 0:
                logger.warning(f"Found {nan_rateidx_count} NaN values in rateIdx, filled with 0")
            
            # 3. Binary success label (true outcome for current rate)
            logger.info("Computing packetSuccessLabel...")
            # Handle NaN and non-finite values in packetSuccess
            df['packetSuccessLabel'] = pd.to_numeric(df['packetSuccess'], errors='coerce').fillna(0).astype(int)
            nan_success_count = df['packetSuccess'].isna().sum()
            if nan_success_count > 0:
                logger.warning(f"Found {nan_success_count} NaN values in packetSuccess, filled with 0")
            
            # 4. Oracle success for best rate (optional, for binary oracle classification)
            logger.info("Computing oracle_best_success...")
            df['oracle_best_success'] = df.apply(find_oracle_success, axis=1)
            
            logger.info("Label construction completed successfully")
            
        except Exception as e:
            logger.error(f"Error during label construction: {e}")
            sys.exit(1)
        
        # --- FEATURE SELECTION ---
        feature_cols = [
            'lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
            'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
            'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance'
        ]
        
        # Check if all feature columns exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing feature columns (will be skipped): {missing_features}")
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"Using {len(feature_cols)} feature columns")
        
        # --- LABEL SELECTION ---
        label_cols = [
            'oracle_best_rateIdx',   # recommended for ML training (multi-class)
            'v3_rateIdx',            # imitation learning (multi-class)
            'packetSuccessLabel',    # actual result (binary classification)
            'oracle_best_success'    # oracle result at best rate (binary classification)
        ]
        
        # --- METADATA (Optional) ---
        meta_cols = ['time','stationId','rateIdx','phyRate','decisionReason','scenario_file']
        available_meta_cols = [col for col in meta_cols if col in df.columns]
        if len(available_meta_cols) != len(meta_cols):
            missing_meta = [col for col in meta_cols if col not in df.columns]
            logger.warning(f"Missing metadata columns: {missing_meta}")
        
        # --- EXPORT DATASET ---
        logger.info("Preparing final dataset...")
        all_cols = available_meta_cols + feature_cols + label_cols
        
        try:
            ml_df = df[all_cols].copy()
            logger.info(f"Final dataset shape: {ml_df.shape}")
        except KeyError as e:
            logger.error(f"Error selecting columns: {e}")
            sys.exit(1)
        
        # Check for any NaN values and log warnings
        nan_cols = ml_df.columns[ml_df.isnull().any()].tolist()
        if nan_cols:
            logger.warning(f"Columns with NaN values: {nan_cols}")
            for col in nan_cols:
                nan_count = ml_df[col].isnull().sum()
                logger.warning(f"  {col}: {nan_count} NaN values ({nan_count/len(ml_df)*100:.2f}%)")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(OUTPUT_CSV)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        logger.info(f"Exporting to {OUTPUT_CSV}...")
        try:
            ml_df.to_csv(OUTPUT_CSV, index=False)
            logger.info("Export completed successfully")
        except PermissionError:
            logger.error(f"Permission denied writing to {OUTPUT_CSV}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error writing output CSV: {e}")
            sys.exit(1)
        
        # --- SUMMARY REPORT ---
        logger.info("=== PROCESSING SUMMARY ===")
        logger.info(f"Input rows: {initial_count}")
        logger.info(f"Output rows: {len(ml_df)}")
        logger.info(f"Filtered rows: {initial_count - len(ml_df)}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Labels: {len(label_cols)}")
        logger.info(f"Metadata cols: {len(available_meta_cols)}")
        
        print(f"\nML-ready CSV exported: {OUTPUT_CSV}")
        print(f"Shape: {ml_df.shape}")
        print("Columns:", ml_df.columns.tolist())
        print("\nLabel distribution summary:")
        
        for label in label_cols:
            try:
                print(f"- {label}:")
                print(ml_df[label].value_counts())
                print()
            except Exception as e:
                logger.error(f"Error displaying distribution for {label}: {e}")
        
        logger.info("Script completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        sys.exit(1)

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    main()