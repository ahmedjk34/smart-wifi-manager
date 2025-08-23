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
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-pre-balancing.csv")

# --- PHY RATES CONFIG ---
# Restrict strictly to 802.11g (8 valid rates, indices 0–7)
G_RATES_BPS = [1000000, 2000000, 5500000, 6000000,
               9000000, 11000000, 12000000, 18000000]

# SNR thresholds for each valid rate (tuned for env.)
SNR_THRESHOLDS = {
    1000000: 6,
    2000000: 8,
    5500000: 11,
    6000000: 12,
    9000000: 14,
    11000000: 16,
    12000000: 17,
    18000000: 20,
}

# --- FUNCTIONS ---

def estimate_p_success(row, rate: int) -> float:
    """Estimate probability of success for a given rate"""
    try:
        if int(row['phyRate']) == rate:
            return float(row['shortSuccRatio'])
        return float(row['lastSnr'] >= SNR_THRESHOLDS.get(rate, 99))
    except Exception as e:
        logger.error(f"Error in estimate_p_success for rate {rate}: {e}")
        return 0.0

def find_oracle_best_rate(row) -> int:
    """Returns best rate index (0–7) maximizing goodput"""
    try:
        best_rate_idx, best_goodput = 0, 0.0
        for i, rate_bps in enumerate(G_RATES_BPS):
            p_succ = estimate_p_success(row, rate_bps)
            goodput = p_succ * rate_bps
            if goodput > best_goodput:
                best_goodput = goodput
                best_rate_idx = i
        return best_rate_idx
    except Exception as e:
        logger.error(f"Error in find_oracle_best_rate: {e}")
        return 3  # safe fallback

def find_oracle_success(row) -> int:
    """Binary oracle success at best rate"""
    try:
        best_rate_idx = find_oracle_best_rate(row)
        rate = G_RATES_BPS[best_rate_idx]
        return int(estimate_p_success(row, rate) > 0.5)
    except Exception as e:
        logger.error(f"Error in find_oracle_success: {e}")
        return 0

def filter_row(row) -> bool:
    """Remove rows with too low SNR"""
    try:
        return float(row['lastSnr']) >= 3
    except Exception:
        return False

def clean_rateidx(x: Union[int, str, float]) -> Union[int, np.nan]:
    """Ensure v3_rateIdx is a valid 0–7 index"""
    try:
        val = int(x)
        if 0 <= val <= 7:
            return val
        return np.nan
    except Exception:
        return np.nan

# --- VALIDATION ---

def validate_input_file(filepath: str) -> bool:
    if not os.path.exists(filepath):
        logger.error(f"Input file does not exist: {filepath}")
        return False
    if not os.access(filepath, os.R_OK):
        logger.error(f"Input file is not readable: {filepath}")
        return False
    logger.info(f"Input file validated: {filepath}")
    return True

def validate_dataframe(df: pd.DataFrame) -> bool:
    required_cols = ['lastSnr','shortSuccRatio','phyRate','rateIdx','packetSuccess']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    return True

# --- MAIN ---

def main():
    try:
        logger.info("=== ML Data Prep Started ===")
        if not validate_input_file(INPUT_CSV):
            sys.exit(1)

        # Load input CSV
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df)} rows")

        if not validate_dataframe(df):
            sys.exit(1)

        # Filter rows
        initial_count = len(df)
        df = df[df.apply(filter_row, axis=1)]
        logger.info(f"Filtered {initial_count - len(df)} rows (low SNR), {len(df)} remain")

        # --- LABELS ---
        logger.info("Building labels...")
        df['oracle_best_rateIdx'] = df.apply(find_oracle_best_rate, axis=1)

        # Clean v3_rateIdx
        df['v3_rateIdx'] = df['rateIdx'].apply(clean_rateidx)
        invalid_v3 = df['v3_rateIdx'].isna().sum()
        if invalid_v3 > 0:
            logger.warning(f"Dropping {invalid_v3} invalid v3_rateIdx rows")
            df = df.dropna(subset=['v3_rateIdx'])
        df['v3_rateIdx'] = df['v3_rateIdx'].astype(int)

        # Clean packetSuccessLabel
        df['packetSuccessLabel'] = pd.to_numeric(df['packetSuccess'], errors='coerce').fillna(0).astype(int)
        df = df[df['packetSuccessLabel'] >= 0]

        # Oracle binary success
        df['oracle_best_success'] = df.apply(find_oracle_success, axis=1)

        # --- FEATURES ---
        feature_cols = [
            'lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
            'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
            'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance'
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

        # --- METADATA ---
        meta_cols = ['time','stationId','rateIdx','phyRate','decisionReason','scenario_file']
        meta_cols = [c for c in meta_cols if c in df.columns]

        # --- FINAL EXPORT ---
        label_cols = ['oracle_best_rateIdx','v3_rateIdx','packetSuccessLabel','oracle_best_success']
        ml_df = df[meta_cols + feature_cols + label_cols].copy()

        # Handle NaNs
        nan_cols = ml_df.columns[ml_df.isnull().any()].tolist()
        if nan_cols:
            for col in nan_cols:
                logger.warning(f"Column {col} has {ml_df[col].isna().sum()} NaN values")

        # Save
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        ml_df.to_csv(OUTPUT_CSV, index=False)

        # --- SUMMARY ---
        logger.info("=== SUMMARY ===")
        logger.info(f"Input rows: {initial_count}")
        logger.info(f"Output rows: {len(ml_df)}")
        logger.info(f"Features used: {len(feature_cols)}")
        logger.info(f"Labels used: {len(label_cols)}")

        print(f"\nML-ready CSV exported: {OUTPUT_CSV}")
        print(f"Shape: {ml_df.shape}")
        print("Labels distribution:")
        for col in label_cols:
            print(f"- {col}:\n{ml_df[col].value_counts()}\n")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

# --- RUN ---
if __name__ == "__main__":
    main()
