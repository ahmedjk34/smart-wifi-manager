"""
ML Data Preparation - FIXED: Tightened oracle variance to ~3 labels/SNR
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import json

# ================== CONFIGURATION ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-ml-cleaned.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-ml-enriched.csv")
LOG_FILE = os.path.join(BASE_DIR, "ml_data_prep.log")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

G_RATES_BPS = [6000000, 9000000, 12000000, 18000000, 24000000, 36000000, 48000000, 54000000]
G_RATE_INDICES = list(range(8))

RATE_MAPPING = {
    0: "6 Mbps (BPSK 1/2)", 1: "9 Mbps (BPSK 3/4)", 2: "12 Mbps (QPSK 1/2)",
    3: "18 Mbps (QPSK 3/4)", 4: "24 Mbps (16-QAM 1/2)", 5: "36 Mbps (16-QAM 3/4)",
    6: "48 Mbps (64-QAM 2/3)", 7: "54 Mbps (64-QAM 3/4)"
}

SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "mobilityMetric", "retryRate", "frameErrorRate",
    "rssiVariance", "interferenceLevel", "distanceMetric", "avgPacketSize"
]

TEMPORAL_LEAKAGE_FEATURES = [
    "consecSuccess", "consecFailure", "retrySuccessRatio",
    "timeSinceLastRateChange", "rateStabilityScore", "recentRateChanges", "packetSuccess"
]

KNOWN_LEAKY_FEATURES = [
    "phyRate", "optimalRateDistance", "recentThroughputTrend",
    "conservativeFactor", "aggressiveFactor", "recommendedSafeRate"
]

ESSENTIAL_COLS = ["rateIdx", "lastSnr"]

CONTEXT_THRESHOLDS = {
    'snr_critical': 8, 'snr_poor': 13, 'snr_marginal': 19,
    'snr_good': 22, 'snr_excellent': 25,
    'variance_high': 5.0, 'variance_moderate': 3.0,
    'mobility_high': 10.0, 'mobility_moderate': 5.0
}

SYNTHETIC_EDGE_CASES = 1000

# ================== LOGGING ==================
def setup_logging():
    logger = logging.getLogger("MLDataPrep")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    try:
        fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"File logging disabled: {e}")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("="*80)
    logger.info("ML DATA PREP - FIXED: Tightened oracle variance to ~3 labels/SNR")
    logger.info("="*80)
    logger.info("CHANGES:")
    logger.info("  - Conservative: 75/20/5 (was 60/30/8/2)")
    logger.info("  - Balanced: 70/20/10 (was 50/25/15/7/3)")
    logger.info("  - Aggressive: 75/20/5 (was 50/25/15/7/3)")
    logger.info("  - Expected: 2-3 unique labels per SNR bin (was 4-6)")
    logger.info("="*80)
    return logger

logger = setup_logging()

# ================== UTILITIES ==================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def is_valid_rateidx(x):
    try:
        v = safe_int(x)
        return 0 <= v <= 7
    except Exception:
        return False

def clamp_rateidx(x):
    try:
        x = safe_int(x)
        return max(0, min(7, x))
    except Exception:
        return 0

# ================== CLASS WEIGHTS ==================
def compute_and_save_class_weights(df: pd.DataFrame, label_cols: List[str], output_dir: str) -> Dict[str, Dict]:
    logger.info("Computing class weights...")
    class_weights_dict = {}
    
    for label_col in label_cols:
        if label_col not in df.columns:
            continue
        valid_labels = df[label_col].dropna()
        if len(valid_labels) == 0:
            continue
        
        unique_classes = np.array(sorted(valid_labels.unique()))
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=valid_labels)
        class_weights = np.minimum(class_weights, 50.0)
        
        weight_dict = {}
        for class_val, weight in zip(unique_classes, class_weights):
            python_key = int(class_val) if isinstance(class_val, (np.integer, np.int64)) else float(class_val)
            weight_dict[python_key] = float(weight)
        
        class_weights_dict[label_col] = weight_dict
        
        class_counts = Counter(valid_labels)
        logger.info(f"\n{label_col} - Class Distribution:")
        for class_val in unique_classes:
            count = class_counts[class_val]
            weight = weight_dict[int(class_val) if isinstance(class_val, (np.integer, np.int64)) else class_val]
            pct = (count / len(valid_labels)) * 100
            logger.info(f"  Class {class_val}: {count:,} ({pct:.1f}%) -> weight: {weight:.3f}")
    
    weights_file = os.path.join(output_dir, "class_weights.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(weights_file, 'w') as f:
        json.dump(class_weights_dict, f, indent=2)
    
    logger.info(f"Class weights saved to: {weights_file}")
    return class_weights_dict

# ================== CLEANING ==================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Initial rows: {len(df)}")
    before = len(df)
    df_clean = df.dropna(subset=ESSENTIAL_COLS, how="any")
    logger.info(f"Dropped {before - len(df_clean)} rows missing essentials")
    
    cols_to_check = [col for col in df_clean.columns if col != 'scenario_file']
    def all_blank(row):
        return all((pd.isna(x) or (isinstance(x, str) and x.strip() == "")) for x in row)
    df_clean = df_clean.loc[~(df_clean[cols_to_check].apply(all_blank, axis=1))]
    
    return df_clean

def filter_sane_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    conditions = [
        df['rateIdx'].apply(lambda x: is_valid_rateidx(x)),
        df['lastSnr'].apply(lambda x: -10 < safe_float(x) < 60)
    ]
    if 'phyRate' in df.columns:
        conditions.append(df['phyRate'].apply(lambda x: 1000000 <= safe_int(x) <= 54000000))
    
    combined = conditions[0]
    for cond in conditions[1:]:
        combined &= cond
    
    df_filtered = df[combined]
    logger.info(f"Kept {len(df_filtered)}/{before} rows ({len(df_filtered)/before*100:.1f}%)")
    return df_filtered

def remove_leaky_and_temporal_features(df):
    ALL_TO_REMOVE = list(set(TEMPORAL_LEAKAGE_FEATURES + KNOWN_LEAKY_FEATURES))
    removed = [f for f in ALL_TO_REMOVE if f in df.columns]
    df_clean = df.drop(columns=removed)
    logger.info(f"Removed {len(removed)} leaky/temporal features")
    return df_clean

# ================== CONTEXT CLASSIFICATION ==================
def classify_network_context(row) -> str:
    snr = safe_float(row.get('lastSnr', 20))
    snr_variance = safe_float(row.get('snrVariance', 0))
    rssi_variance = safe_float(row.get('rssiVariance', 0))
    interference = safe_float(row.get('interferenceLevel', 0))
    mobility = safe_float(row.get('mobilityMetric', 0))
    
    if snr < CONTEXT_THRESHOLDS['snr_critical']:
        base = 'emergency_recovery'
    elif snr < CONTEXT_THRESHOLDS['snr_poor']:
        base = 'poor'
    elif snr < CONTEXT_THRESHOLDS['snr_marginal']:
        base = 'marginal_conditions'
    elif snr < CONTEXT_THRESHOLDS['snr_good']:
        base = 'good'
    elif snr >= CONTEXT_THRESHOLDS['snr_excellent']:
        base = 'excellent'
    else:
        base = 'good'
    
    combined_variance = max(snr_variance, rssi_variance)
    if combined_variance > CONTEXT_THRESHOLDS['variance_high']:
        stability = 'unstable'
    elif combined_variance > CONTEXT_THRESHOLDS['variance_moderate']:
        stability = 'somewhat_unstable'
    else:
        stability = 'stable'
    
    if interference > 0.7:
        if base in ['excellent', 'good']:
            base = 'marginal_conditions'
        elif base == 'marginal_conditions':
            base = 'poor'
    
    if mobility > CONTEXT_THRESHOLDS['mobility_high']:
        if base == 'excellent':
            base = 'good'
        elif base == 'good':
            base = 'marginal_conditions'
    
    if base == 'emergency_recovery':
        return 'emergency_recovery'
    elif stability == 'unstable' and base in ['good', 'excellent']:
        return f'{base}_unstable'
    elif stability == 'unstable':
        return 'poor_unstable'
    else:
        return f'{base}_stable'

# ================== ORACLE LABELS - TIGHTENED ==================
def create_snr_based_oracle_labels(row: pd.Series, context: str, current_rate: int) -> Dict[str, int]:
    """
    FIXED: Tightened distributions to produce ~3 unique labels per SNR bin
    
    Strategy:
    - Conservative: 75/20/5 → Mostly base, sometimes -1, rarely -2
    - Balanced: 70/20/10 → Mostly base, sometimes ±1
    - Aggressive: 75/20/5 → Mostly base, sometimes +1, rarely +2
    
    Expected: Each SNR → 2-3 labels (not 4-6)
    """
    snr = safe_float(row.get('lastSnr', 20))
    snr_variance = safe_float(row.get('snrVariance', 0))
    rssi_variance = safe_float(row.get('rssiVariance', 0))
    interference = safe_float(row.get('interferenceLevel', 0))
    mobility = safe_float(row.get('mobilityMetric', 0))
    
    # Base rate from SNR
    if snr < 8: base = 0
    elif snr < 10: base = 1
    elif snr < 13: base = 2
    elif snr < 16: base = 3
    elif snr < 19: base = 4
    elif snr < 22: base = 5
    elif snr < 25: base = 6
    else: base = 7
    
    # Apply penalties
    penalty = 0
    combined_variance = max(snr_variance, rssi_variance)
    if combined_variance > CONTEXT_THRESHOLDS['variance_high']:
        penalty += 1
    elif combined_variance > CONTEXT_THRESHOLDS['variance_moderate']:
        penalty += 0.5
    
    if mobility > CONTEXT_THRESHOLDS['mobility_high']:
        penalty += 1
    elif mobility > CONTEXT_THRESHOLDS['mobility_moderate']:
        penalty += 0.5
    
    if interference > 0.7:
        penalty += 1
    elif interference > 0.4:
        penalty += 0.5
    
    base = max(0, int(base - penalty))
    
    # Context adjustments
    if context == 'emergency_recovery':
        base = max(0, base - 1)
    elif context in ['poor_unstable', 'poor_stable']:
        base = max(0, base - 1)
    elif context in ['excellent_stable', 'excellent_unstable']:
        base = min(7, base + 1)
    
    # TIGHTENED: Conservative (75/20/5)
    rand_cons = np.random.rand()
    if rand_cons < 0.75:
        cons = base
    elif rand_cons < 0.95:
        cons = max(0, base - 1)
    else:
        cons = max(0, base - 2)
    
    # TIGHTENED: Balanced (70/20/10)
    rand_bal = np.random.rand()
    if rand_bal < 0.70:
        bal = base
    elif rand_bal < 0.90:
        bal = max(0, base - 1) if np.random.rand() < 0.5 else min(7, base + 1)
    else:
        bal = max(0, base - 2) if np.random.rand() < 0.5 else min(7, base + 2)
    
    # TIGHTENED: Aggressive (75/20/5)
    rand_agg = np.random.rand()
    if rand_agg < 0.75:
        agg = base
    elif rand_agg < 0.95:
        agg = min(7, base + 1)
    else:
        agg = min(7, base + 2)
    
    return {
        "oracle_conservative": cons,
        "oracle_balanced": bal,
        "oracle_aggressive": agg,
    }

# ================== SYNTHETIC EDGE CASES ==================
def generate_critical_edge_cases(target_samples: int = SYNTHETIC_EDGE_CASES) -> pd.DataFrame:
    edge_cases = []
    scenarios = [
        create_high_snr_high_rate, create_low_snr_low_rate,
        create_mid_snr_mid_rate, create_high_variance_scenario,
        create_high_mobility_scenario
    ]
    samples_per = target_samples // len(scenarios)
    for fn in scenarios:
        for _ in range(samples_per):
            edge_cases.append(fn())
    logger.info(f"Generated {len(edge_cases)} synthetic edge cases")
    return pd.DataFrame(edge_cases)

def create_high_snr_high_rate() -> Dict[str, Any]:
    snr = np.random.uniform(25, 35)
    variance = np.random.uniform(0.1, 1.0)
    rssi_variance = np.random.uniform(0.1, 1.0)
    interference = np.random.uniform(0.0, 0.2)
    mobility = np.random.uniform(0, 3)
    
    row = pd.Series({
        'lastSnr': snr, 'snrVariance': variance,
        'rssiVariance': rssi_variance, 'interferenceLevel': interference,
        'mobilityMetric': mobility
    })
    oracle_labels = create_snr_based_oracle_labels(row, 'excellent_stable', 7)
    
    return {
        'lastSnr': snr, 'snrVariance': variance, 'rssiVariance': rssi_variance,
        'interferenceLevel': interference, 'mobilityMetric': mobility,
        'channelWidth': 20, 'rateIdx': 7,
        **oracle_labels, 'network_context': 'excellent_stable'
    }

def create_mid_snr_mid_rate() -> Dict[str, Any]:
    snr = np.random.uniform(12, 20)
    variance = np.random.uniform(0.5, 2.5)
    rssi_variance = np.random.uniform(0.5, 2.5)
    interference = np.random.uniform(0.2, 0.5)
    mobility = np.random.uniform(0, 8)
    
    row = pd.Series({
        'lastSnr': snr, 'snrVariance': variance,
        'rssiVariance': rssi_variance, 'interferenceLevel': interference,
        'mobilityMetric': mobility
    })
    oracle_labels = create_snr_based_oracle_labels(row, 'good_stable', 4)
    
    return {
        'lastSnr': snr, 'snrVariance': variance, 'rssiVariance': rssi_variance,
        'interferenceLevel': interference, 'mobilityMetric': mobility,
        'channelWidth': 20, 'rateIdx': np.random.choice([3, 4, 5]),
        **oracle_labels, 'network_context': 'good_stable'
    }

def create_low_snr_low_rate() -> Dict[str, Any]:
    snr = np.random.uniform(3, 10)
    variance = np.random.uniform(0.5, 3.0)
    rssi_variance = np.random.uniform(0.5, 3.0)
    interference = np.random.uniform(0.6, 1.0)
    mobility = np.random.uniform(0, 5)
    
    row = pd.Series({
        'lastSnr': snr, 'snrVariance': variance,
        'rssiVariance': rssi_variance, 'interferenceLevel': interference,
        'mobilityMetric': mobility
    })
    oracle_labels = create_snr_based_oracle_labels(row, 'emergency_recovery', 0)
    
    return {
        'lastSnr': snr, 'snrVariance': variance, 'rssiVariance': rssi_variance,
        'interferenceLevel': interference, 'mobilityMetric': mobility,
        'channelWidth': 20, 'rateIdx': np.random.choice([0, 1, 2]),
        **oracle_labels, 'network_context': 'emergency_recovery'
    }

def create_high_variance_scenario() -> Dict[str, Any]:
    snr = np.random.uniform(15, 25)
    variance = np.random.uniform(5, 10)
    rssi_variance = np.random.uniform(5, 10)
    interference = np.random.uniform(0.3, 0.6)
    mobility = np.random.uniform(0, 5)
    
    row = pd.Series({
        'lastSnr': snr, 'snrVariance': variance,
        'rssiVariance': rssi_variance, 'interferenceLevel': interference,
        'mobilityMetric': mobility
    })
    oracle_labels = create_snr_based_oracle_labels(row, 'good_unstable', 4)
    
    return {
        'lastSnr': snr, 'snrVariance': variance, 'rssiVariance': rssi_variance,
        'interferenceLevel': interference, 'mobilityMetric': mobility,
        'channelWidth': 20, 'rateIdx': np.random.choice([3, 4, 5, 6]),
        **oracle_labels, 'network_context': 'good_unstable'
    }

def create_high_mobility_scenario() -> Dict[str, Any]:
    snr = np.random.uniform(15, 25)
    variance = np.random.uniform(2, 5)
    rssi_variance = np.random.uniform(2, 5)
    interference = np.random.uniform(0.3, 0.6)
    mobility = np.random.uniform(10, 50)
    
    row = pd.Series({
        'lastSnr': snr, 'snrVariance': variance,
        'rssiVariance': rssi_variance, 'interferenceLevel': interference,
        'mobilityMetric': mobility
    })
    oracle_labels = create_snr_based_oracle_labels(row, 'good_stable', 4)
    
    return {
        'lastSnr': snr, 'snrVariance': variance, 'rssiVariance': rssi_variance,
        'interferenceLevel': interference, 'mobilityMetric': mobility,
        'channelWidth': 20, 'rateIdx': np.random.choice([3, 4, 5, 6]),
        **oracle_labels, 'network_context': 'good_stable'
    }

# ================== VALIDATION ==================
def validate_oracle_randomness(df: pd.DataFrame, logger):
    logger.info("\n" + "="*80)
    logger.info("VALIDATING ORACLE RANDOMNESS")
    logger.info("="*80)
    
    df_temp = df.copy()
    df_temp['snr_bin'] = pd.cut(df_temp['lastSnr'], bins=20)
    
    all_passed = True
    for oracle_col in ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive']:
        if oracle_col not in df_temp.columns:
            continue
        
        labels_per_bin = df_temp.groupby('snr_bin')[oracle_col].nunique()
        avg_labels = labels_per_bin.mean()
        min_labels = labels_per_bin.min()
        
        logger.info(f"\n{oracle_col}:")
        logger.info(f"   Avg labels/SNR bin: {avg_labels:.2f}")
        logger.info(f"   Min labels/SNR bin: {min_labels}")
        
        if avg_labels < 1.5:
            logger.error(f"   FAILED: Deterministic! (avg {avg_labels:.2f} < 1.5)")
            all_passed = False
        elif avg_labels > 3.5:
            logger.warning(f"   WARNING: Too much variance (avg {avg_labels:.2f} > 3.5)")
            logger.warning(f"   This may cause excessive rate changes")
        else:
            logger.info(f"   PASSED: Good variance ({avg_labels:.2f} labels/bin)")
        
        logger.info(f"\n   Example SNR->Label mappings:")
        for idx, (bin_range, group) in enumerate(df_temp.groupby('snr_bin')):
            if idx >= 5:
                break
            labels = sorted(group[oracle_col].unique())
            logger.info(f"      {bin_range}: {labels}")
    
    if all_passed:
        logger.info("\nVALIDATION PASSED")
    else:
        logger.error("\nVALIDATION FAILED")
    
    return all_passed

# ================== MAIN ==================
def main():
    logger.info("=== ML Data Prep Started ===")
    if not os.path.exists(INPUT_CSV):
        logger.error(f"Input not found: {INPUT_CSV}")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} rows")
    
    df = clean_dataframe(df)
    df = filter_sane_rows(df)
    
    logger.info("Classifying context...")
    df['network_context'] = df.apply(classify_network_context, axis=1)
    
    logger.info("Generating oracle labels...")
    oracle_labels = []
    for idx, row in df.iterrows():
        current_rate = clamp_rateidx(row.get('rateIdx', 0))
        context = row['network_context']
        labels = create_snr_based_oracle_labels(row, context, current_rate)
        oracle_labels.append(labels)
        if idx % 100000 == 0 and idx > 0:
            logger.info(f"Processed {idx} rows...")
    
    oracle_df = pd.DataFrame(oracle_labels)
    df = pd.concat([df.reset_index(drop=True), oracle_df.reset_index(drop=True)], axis=1)
    
    logger.info("Generating synthetic samples...")
    synthetic_df = generate_critical_edge_cases()
    
    logger.info("Combining datasets...")
    final_df = pd.concat([df, synthetic_df], ignore_index=True, sort=False)
    logger.info(f"Final shape: {final_df.shape}")
    
    randomness_passed = validate_oracle_randomness(final_df, logger)
    if not randomness_passed:
        logger.error("CRITICAL: Validation failed!")
        sys.exit(1)
    
    weights_dir = os.path.join(BASE_DIR, "model_artifacts")
    label_cols = ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive', 'rateIdx']
    compute_and_save_class_weights(final_df, label_cols, weights_dir)
    
    logger.info("Removing leaky features...")
    final_df = remove_leaky_and_temporal_features(final_df)
    
    try:
        final_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved: {OUTPUT_CSV} ({final_df.shape[0]} rows, {final_df.shape[1]} cols)")
        print(f"\nSaved: {OUTPUT_CSV}")
        print(f"  Rows: {final_df.shape[0]:,}")
        print(f"  Cols: {final_df.shape[1]}")
        print(f"  Oracle variance: ~2-3 labels/SNR (tightened)")
    except Exception as e:
        logger.error(f"Save failed: {e}")
        sys.exit(1)
    
    print("\n--- LABEL DISTRIBUTION ---")
    for lbl in ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive']:
        if lbl in final_df.columns:
            print(f"{lbl}:\n{final_df[lbl].value_counts().sort_index()}\n")
    
    logger.info("=== Finished ===")

if __name__ == "__main__":
    main()