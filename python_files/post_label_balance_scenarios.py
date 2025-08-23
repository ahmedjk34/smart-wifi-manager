import pandas as pd
import numpy as np
import os
import logging
import sys

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-pre-balancing.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-balanced.csv")

# --- LOGGING SETUP ---
def setup_logging():
    handlers = []
    try:
        log_file = os.path.join(BASE_DIR, 'ml_post_balancing_hybrid.log')
        handlers.append(logging.FileHandler(log_file))
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not create log file ({e}), will log to console only.")
    handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- BALANCING PARAMETERS ---
LABEL_COL = 'oracle_best_rateIdx'
MINORITY_CLASSES = [1, 2, 3, 4, 5, 6]
MAX_PER_CLASS = 15000  # Target for minority classes
MAJORITY_CLASS = 7
MAJORITY_CAP = 45000   # Hybrid: cap majority to 3x minority count
PERTURB_STD_FRAC = 0.35
RANDOM_SEED = 42

SAFE_FEATURES = [
    "lastSnr","snrFast","snrSlow","shortSuccRatio","medSuccRatio",
    "consecSuccess","consecFailure","severity","confidence","T1","T2","T3",
    "offeredLoad","queueLen","retryCount","channelWidth","mobilityMetric","snrVariance"
]

def safe_perturb(val, std, global_min, global_max, col):
    try:
        delta = np.random.uniform(-PERTURB_STD_FRAC, PERTURB_STD_FRAC) * std
        new_val = val + delta
        new_val = max(global_min, min(global_max, new_val))
        if col in ['queueLen', 'retryCount', 'channelWidth', 'stationId']:
            new_val = int(round(max(0, new_val)))
        if col in ['shortSuccRatio', 'medSuccRatio', 'confidence']:
            new_val = min(max(0.0, new_val), 1.0)
        if col in ['severity', 'mobilityMetric', 'snrVariance']:
            new_val = max(0.0, new_val)
        return new_val
    except Exception as e:
        logger.warning(f"Perturbation failed for {col}: {e}")
        return val

def main():
    try:
        logger.info("=== Hybrid Post-label Class Balancing Script Started ===")
        logger.info(f"Input file: {INPUT_CSV}")

        if not os.path.exists(INPUT_CSV):
            logger.error(f"Input file does not exist: {INPUT_CSV}")
            sys.exit(1)
        if not os.access(INPUT_CSV, os.R_OK):
            logger.error(f"Input file is not readable: {INPUT_CSV}")
            sys.exit(1)

        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df)} rows from ML-ready CSV.")

        # Filter features to those present
        safe_features_actual = [c for c in SAFE_FEATURES if c in df.columns]
        logger.info(f"Safe features for perturbation: {safe_features_actual}")

        # Precompute global min/max/std for each feature
        global_mins = df[safe_features_actual].min()
        global_maxs = df[safe_features_actual].max()
        global_stds = df[safe_features_actual].std().replace({0: 1e-6})

        # Initial label distribution
        logger.info("Initial label distribution (oracle_best_rateIdx):")
        logger.info(f"\n{df[LABEL_COL].value_counts().sort_index()}")

        balanced_dfs = []
        dropped_rows = 0
        synthetic_rows_total = 0

        # Balance minority classes
        for label_val in MINORITY_CLASSES:
            class_df = df[df[LABEL_COL] == label_val]
            count = len(class_df)
            logger.info(f"Class {label_val}: {count} rows")
            if count == 0:
                logger.warning(f"No rows found for class {label_val}. Skipping...")
                continue
            if count < MAX_PER_CLASS:
                logger.info(f"Robust upsampling class {label_val}: {count} -> {MAX_PER_CLASS}")
                upsampled_rows = []
                needed = MAX_PER_CLASS - count
                upsampled_rows.append(class_df)
                for idx in range(needed):
                    base = class_df.sample(1, random_state=RANDOM_SEED + idx).iloc[0].copy()
                    new_row = base.copy()
                    for col in safe_features_actual:
                        val = base[col]
                        std = global_stds[col]
                        new_row[col] = safe_perturb(val, std, global_mins[col], global_maxs[col], col)
                    upsampled_rows.append(pd.DataFrame([new_row]))
                upsampled_df = pd.concat(upsampled_rows, ignore_index=True)
                balanced_dfs.append(upsampled_df)
                synthetic_rows_total += needed
                logger.info(f"Generated {needed} synthetic rows for class {label_val}.")
            else:
                logger.info(f"Downsampling class {label_val}: {count} -> {MAX_PER_CLASS}")
                downsampled = class_df.sample(MAX_PER_CLASS, replace=False, random_state=RANDOM_SEED)
                balanced_dfs.append(downsampled)
                dropped = count - MAX_PER_CLASS
                dropped_rows += dropped
                logger.info(f"Dropped {dropped} rows from class {label_val}.")

        # Hybrid handling for majority class (7)
        class7_df = df[df[LABEL_COL] == MAJORITY_CLASS]
        count7 = len(class7_df)
        logger.info(f"Class {MAJORITY_CLASS}: {count7} rows")
        if count7 > MAJORITY_CAP:
            logger.info(f"Hybrid downsampling class {MAJORITY_CLASS}: {count7} -> {MAJORITY_CAP}")
            downsampled7 = class7_df.sample(MAJORITY_CAP, replace=False, random_state=RANDOM_SEED)
            balanced_dfs.append(downsampled7)
            dropped = count7 - MAJORITY_CAP
            dropped_rows += dropped
            logger.info(f"Dropped {dropped} rows from class {MAJORITY_CLASS}.")
        else:
            logger.info(f"Kept class {MAJORITY_CLASS}: {count7} rows (below hybrid cap).")
            balanced_dfs.append(class7_df)

        # Edge class (0): keep at natural size
        class0_df = df[df[LABEL_COL] == 0]
        logger.info(f"Kept class 0: {len(class0_df)} rows (no balancing applied).")
        balanced_dfs.append(class0_df)

        # Concatenate all balanced data
        final_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Final row count after balancing: {len(final_df)} (dropped {dropped_rows} rows, {synthetic_rows_total} synthetic rows added).")

        # Final label distribution
        logger.info("Final label distribution (oracle_best_rateIdx):")
        logger.info(f"\n{final_df[LABEL_COL].value_counts().sort_index()}")

        # Save output
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Balanced ML-ready CSV exported: {OUTPUT_CSV}")
        logger.info(f"Shape: {final_df.shape}")

        # Diagnostics for other labels
        logger.info("Other label distributions (for reference):")
        for col in ['v3_rateIdx', 'packetSuccessLabel', 'oracle_best_success']:
            if col in final_df.columns:
                logger.info(f"\n{col}:\n{final_df[col].value_counts().sort_index()}")

        logger.info("=== Hybrid Robust Balancing complete ===")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()