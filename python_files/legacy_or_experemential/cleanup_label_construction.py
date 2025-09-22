import os
import pandas as pd
import logging
import sys

# --- LOGGING SETUP ---
def setup_logging():
    """Setup logging with fallback if file creation fails"""
    handlers = []
    
    # Try to create file handler in script directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(script_dir, 'cleanup_data_prep.log')
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

def main():
    """Main cleanup function with comprehensive error handling"""
    try:
        logger.info("Starting ML dataset cleanup script")
        
        # --- Configuration ---
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        INPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-ready.csv")
        
        ALL_PHY_RATES = [
            1000000, 2000000, 5500000, 6000000, 9000000, 11000000,
            12000000, 18000000, 24000000, 36000000, 48000000, 54000000
        ]
        
        # --- Load your ML-ready CSV ---
        logger.info(f"Loading ML-ready CSV from: {INPUT_CSV}")
        
        if not os.path.exists(INPUT_CSV):
            logger.error(f"Input file does not exist: {INPUT_CSV}")
            sys.exit(1)
        
        try:
            # Suppress the dtype warning by specifying dtype for problematic columns
            ml_df = pd.read_csv(INPUT_CSV, dtype={'time': str}, low_memory=False)
            logger.info(f"Successfully loaded {len(ml_df)} rows from input CSV")
            logger.info(f"Original shape: {ml_df.shape}")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            sys.exit(1)
        
        # --- 1. Remove or fix rare/invalid v3_rateIdx values ---
        logger.info("Cleaning v3_rateIdx values...")
        valid_idxs = set(range(len(ALL_PHY_RATES)))
        
        if 'v3_rateIdx' not in ml_df.columns:
            logger.error("Column 'v3_rateIdx' not found in dataset")
            sys.exit(1)
        
        # Check for invalid rate indices
        invalid_mask = ~ml_df['v3_rateIdx'].isin(valid_idxs)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} rows with invalid v3_rateIdx values")
            invalid_values = ml_df.loc[invalid_mask, 'v3_rateIdx'].unique()
            logger.warning(f"Invalid v3_rateIdx values: {invalid_values}")
            
            ml_df = ml_df[~invalid_mask]
            logger.info(f"Removed {invalid_count} rows with invalid v3_rateIdx")
        else:
            logger.info("All v3_rateIdx values are valid")
        
        # --- 2. Replace -1 in packetSuccessLabel with 0 (fail) ---
        logger.info("Cleaning packetSuccessLabel values...")
        
        if 'packetSuccessLabel' not in ml_df.columns:
            logger.error("Column 'packetSuccessLabel' not found in dataset")
            sys.exit(1)
        
        negative_mask = ml_df['packetSuccessLabel'] == -1
        negative_count = negative_mask.sum()
        
        if negative_count > 0:
            logger.info(f"Replacing {negative_count} instances of -1 with 0 in packetSuccessLabel")
            ml_df.loc[negative_mask, 'packetSuccessLabel'] = 0
        else:
            logger.info("No -1 values found in packetSuccessLabel")
        
        # --- 3. Handle NaN values ---
        logger.info("Checking for NaN values...")
        
        # Count NaN values per column
        nan_counts = ml_df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        if len(nan_cols) > 0:
            logger.warning(f"Found NaN values in {len(nan_cols)} columns:")
            for col, count in nan_cols.items():
                percentage = (count / len(ml_df)) * 100
                logger.warning(f"  {col}: {count} NaN values ({percentage:.2f}%)")
            
            logger.info("Filling NaN values with 0...")
            ml_df = ml_df.fillna(0)
            logger.info("NaN values filled successfully")
        else:
            logger.info("No NaN values found")
        
        # --- 4. Validate data ranges ---
        logger.info("Validating data ranges...")
        
        # Check packetSuccessLabel is binary
        unique_success_vals = ml_df['packetSuccessLabel'].unique()
        if not set(unique_success_vals).issubset({0, 1}):
            logger.warning(f"packetSuccessLabel has non-binary values: {unique_success_vals}")
        
        # Check oracle labels are in valid range
        if 'oracle_best_rateIdx' in ml_df.columns:
            oracle_vals = ml_df['oracle_best_rateIdx'].unique()
            invalid_oracle = [v for v in oracle_vals if v not in valid_idxs]
            if invalid_oracle:
                logger.warning(f"oracle_best_rateIdx has invalid values: {invalid_oracle}")
        
        # --- 5. Save cleaned version ---
        SAVE_DIR = os.path.expanduser("~/ns-allinone-3.41/ns-3.41")
        SAVE_PATH = os.path.join(SAVE_DIR, "smartv4-ml-ready-cleaned.csv")
        
        # Ensure save directory exists
        if not os.path.exists(SAVE_DIR):
            logger.info(f"Creating save directory: {SAVE_DIR}")
            try:
                os.makedirs(SAVE_DIR, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create save directory: {e}")
                sys.exit(1)
        
        logger.info(f"Saving cleaned dataset to: {SAVE_PATH}")
        
        try:
            ml_df.to_csv(SAVE_PATH, index=False)
            logger.info("Cleaned dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving cleaned dataset: {e}")
            sys.exit(1)
        
        # --- Final Summary ---
        logger.info("=== CLEANUP SUMMARY ===")
        logger.info(f"Final shape: {ml_df.shape}")
        logger.info(f"Rows removed: {invalid_count}")
        logger.info(f"NaN values filled: {nan_cols.sum() if len(nan_cols) > 0 else 0}")
        
        print(f"\nCleaned ML dataset saved as {SAVE_PATH}")
        print(f"Shape: {ml_df.shape}")
        
        # Display label distributions
        print("\nLabel distributions:")
        
        if 'v3_rateIdx' in ml_df.columns:
            print("v3_rateIdx value counts:")
            print(ml_df['v3_rateIdx'].value_counts().sort_index())
            print()
        
        if 'packetSuccessLabel' in ml_df.columns:
            print("packetSuccessLabel value counts:")
            print(ml_df['packetSuccessLabel'].value_counts().sort_index())
            print()
        
        if 'oracle_best_rateIdx' in ml_df.columns:
            print("oracle_best_rateIdx value counts:")
            print(ml_df['oracle_best_rateIdx'].value_counts().sort_index())
            print()
        
        logger.info("Cleanup script completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in cleanup: {e}")
        sys.exit(1)

# --- Alternative cleanup option (commented out) ---
def cleanup_with_dropna():
    """Alternative cleanup that drops NaN rows instead of filling"""
    logger.info("Alternative: Dropping rows with NaN values instead of filling")
    # ml_df = ml_df.dropna()
    pass

if __name__ == "__main__":
    main()