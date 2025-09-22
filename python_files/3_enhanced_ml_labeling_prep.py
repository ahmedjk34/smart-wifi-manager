import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import json

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-ml-cleaned.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-ml-enriched.csv")
LOG_FILE = os.path.join(BASE_DIR, "ml_data_prep.log")

G_RATES_BPS = [1000000, 2000000, 5500000, 6000000, 9000000, 11000000, 12000000, 18000000]
G_RATE_INDICES = list(range(8))
SNR_THRESHOLDS = {
    0: 3, 1: 6, 2: 9, 3: 12, 4: 15, 5: 18, 6: 21, 7: 24
}

ESSENTIAL_COLS = ["rateIdx", "lastSnr", "shortSuccRatio", "consecFailure", "phyRate"]

# ----------------------------------------
# LOGGING SETUP
# ----------------------------------------
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
    return logger

logger = setup_logging()

# ----------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------
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

# ----------------------------------------
# CLASS WEIGHTS COMPUTATION (NEW)
# ----------------------------------------
def compute_and_save_class_weights(df: pd.DataFrame, label_cols: List[str], output_dir: str) -> Dict[str, Dict]:
    """Compute class weights for imbalanced labels and save them."""
    logger.info("🔢 Computing class weights for imbalanced target labels...")
    
    class_weights_dict = {}
    
    for label_col in label_cols:
        if label_col not in df.columns:
            logger.warning(f"Label column {label_col} not found, skipping...")
            continue
            
        # Remove NaN values for weight computation
        valid_labels = df[label_col].dropna()
        
        if len(valid_labels) == 0:
            logger.warning(f"No valid labels found for {label_col}, skipping...")
            continue
            
        # Get unique classes
        unique_classes = sorted(valid_labels.unique())
        
        # Compute balanced class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=valid_labels
        )
        
        # Create dictionary mapping class -> weight
        weight_dict = dict(zip(unique_classes, class_weights))
        class_weights_dict[label_col] = weight_dict
        
        # Log distribution and weights
        class_counts = Counter(valid_labels)
        logger.info(f"\n📊 {label_col} - Class Distribution & Weights:")
        for class_val in unique_classes:
            count = class_counts[class_val]
            weight = weight_dict[class_val]
            pct = (count / len(valid_labels)) * 100
            logger.info(f"  Class {class_val}: {count:,} samples ({pct:.1f}%) -> weight: {weight:.3f}")
        
        print(f"\n{label_col} Class Weights:")
        for class_val, weight in weight_dict.items():
            print(f"  {class_val}: {weight:.3f}")
    
    # Save weights to JSON file
    weights_file = os.path.join(output_dir, "class_weights.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(weights_file, 'w') as f:
        json.dump(class_weights_dict, f, indent=2)
    
    logger.info(f"💾 Class weights saved to: {weights_file}")
    print(f"💾 Class weights saved to: {weights_file}")
    
    return class_weights_dict

# ----------------------------------------
# CLEANING FUNCTIONS
# ----------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Initial row count: {len(df)}")
    # Drop rows missing ALL essential columns
    before = len(df)
    df_clean = df.dropna(subset=ESSENTIAL_COLS, how="any")
    logger.info(f"Dropped {before - len(df_clean)} rows missing ANY essential columns")
    
    # Drop rows where most essential columns are missing (thresh=3)
    before2 = len(df_clean)
    df_clean = df_clean.dropna(subset=ESSENTIAL_COLS, thresh=3)
    logger.info(f"Dropped {before2 - len(df_clean)} rows with <3 essential columns present")

    # Drop rows where all values except scenario_file are blank/empty string
    before3 = len(df_clean)
    cols_to_check = [col for col in df_clean.columns if col != 'scenario_file']
    def all_blank(row):
        return all((pd.isna(x) or (isinstance(x, str) and x.strip() == "")) for x in row)
    df_clean = df_clean.loc[~(df_clean[cols_to_check].apply(all_blank, axis=1))]
    logger.info(f"Dropped {before3 - len(df_clean)} rows with all blank except scenario_file")

    # Stats for missing values
    missing_stats = df_clean.isnull().sum()
    logger.info("Missing value counts per column (after cleaning):")
    for col, cnt in missing_stats.items():
        if cnt > 0:
            logger.info(f"  {col}: {cnt}")
    print("\n--- CLEANING SUMMARY ---")
    print(f"Rows after cleaning: {len(df_clean)}")
    for col, cnt in missing_stats.items():
        if cnt > 0:
            print(f"  {col}: {cnt} missing")
    return df_clean

# ----------------------------------------
# OUTLIER/SANITY FILTERING
# ----------------------------------------
def filter_sane_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Filter based on reasonable ranges for essential features
    df_filtered = df[
        df['rateIdx'].apply(lambda x: is_valid_rateidx(x)) &
        df['phyRate'].apply(lambda x: safe_int(x) in G_RATES_BPS) &
        df['lastSnr'].apply(lambda x: 0 < safe_float(x) < 200) &
        df['shortSuccRatio'].apply(lambda x: 0 <= safe_float(x) <= 1.1) &
        df['medSuccRatio'].apply(lambda x: 0 <= safe_float(x) <= 1.1 if not pd.isna(x) else True) &
        df['consecFailure'].apply(lambda x: 0 <= safe_int(x) < 1000) &
        df['consecSuccess'].apply(lambda x: 0 <= safe_int(x) < 100000) &
        df['snrVariance'].apply(lambda x: 0 <= safe_float(x) < 10000 if not pd.isna(x) else True) &
        df['severity'].apply(lambda x: 0 <= safe_float(x) < 1e7 if not pd.isna(x) else True) &
        df['confidence'].apply(lambda x: 0 <= safe_float(x) <= 1.1 if not pd.isna(x) else True) &
        df['queueLen'].apply(lambda x: 0 <= safe_float(x) < 1e6 if not pd.isna(x) else True) &
        df['retryCount'].apply(lambda x: 0 <= safe_float(x) < 1e6 if not pd.isna(x) else True) &
        df['channelWidth'].apply(lambda x: safe_int(x) in [20, 40] if not pd.isna(x) else True) &
        df['mobilityMetric'].apply(lambda x: 0 <= safe_float(x) < 1e6 if not pd.isna(x) else True)
    ]
    logger.info(f"Dropped {before - len(df_filtered)} rows failing sanity-range checks")
    print(f"Dropped {before - len(df_filtered)} rows failing sanity-range checks")
    # Optionally, print a few examples of dropped rows (for debugging)
    dropped_rows = before - len(df_filtered)
    if dropped_rows > 0:
        print("Example dropped rows:")
        print(df[~df.index.isin(df_filtered.index)].head(5))
    return df_filtered

# ----------------------------------------
# CONTEXT CLASSIFICATION
# ----------------------------------------
def classify_network_context(row) -> str:
    snr = safe_float(row.get('lastSnr', 0))
    snr_variance = safe_float(row.get('snrVariance', 0))
    success_ratio = safe_float(row.get('shortSuccRatio', 1))
    consec_failures = safe_int(row.get('consecFailure', 0))
    if snr < 10 or success_ratio < 0.5 or consec_failures >= 3:
        return 'emergency_recovery'
    elif snr < 15 or snr_variance > 5:
        return 'poor_unstable'
    elif snr < 20 or success_ratio < 0.8:
        return 'marginal_conditions'
    elif snr_variance > 3:
        return 'good_unstable'
    elif snr > 25 and success_ratio > 0.9:
        return 'excellent_stable'
    else:
        return 'good_stable'

# ----------------------------------------
# ORACLE LABEL CREATION
# ----------------------------------------
def create_context_specific_labels(row: pd.Series, context: str, current_rate: int) -> Dict[str, int]:
    snr = safe_float(row.get('lastSnr', 0))
    success_ratio = safe_float(row.get('shortSuccRatio', 1))
    consec_failures = safe_int(row.get('consecFailure', 0))
    consec_success = safe_int(row.get('consecSuccess', 0))
    labels = {}
    if context == 'emergency_recovery':
        labels['oracle_conservative'] = get_emergency_safe_rate(snr, success_ratio)
        labels['oracle_balanced'] = labels['oracle_conservative']
        labels['oracle_aggressive'] = min(labels['oracle_conservative'] + 1, 7)
    elif context == 'poor_unstable':
        safe_rate = get_snr_based_safe_rate(snr)
        labels['oracle_conservative'] = safe_rate
        labels['oracle_balanced'] = min(safe_rate + 1, 7) if success_ratio > 0.7 else safe_rate
        labels['oracle_aggressive'] = get_throughput_optimal_rate(row)
    elif context == 'marginal_conditions':
        conservative = get_snr_based_safe_rate(snr)
        aggressive = get_throughput_optimal_rate(row)
        labels['oracle_conservative'] = conservative
        labels['oracle_balanced'] = (conservative + aggressive) // 2
        labels['oracle_aggressive'] = aggressive
    elif context in ['good_stable', 'good_unstable']:
        conservative = max(current_rate - 1, 0) if success_ratio < 0.8 else current_rate
        aggressive = get_throughput_optimal_rate(row)
        labels['oracle_conservative'] = conservative
        labels['oracle_balanced'] = min(aggressive, 7)
        labels['oracle_aggressive'] = min(aggressive + 1, 7) if consec_success > 5 else aggressive
    elif context == 'excellent_stable':
        labels['oracle_conservative'] = get_throughput_optimal_rate(row)
        labels['oracle_balanced'] = min(get_throughput_optimal_rate(row) + 1, 7)
        labels['oracle_aggressive'] = 7
    for key in labels:
        labels[key] = clamp_rateidx(labels[key])
    return labels

def get_emergency_safe_rate(snr: float, success_ratio: float) -> int:
    if success_ratio < 0.3 or snr < 8:
        return 0
    elif snr < 12:
        return 1
    elif snr < 16:
        return 2
    else:
        return 3

def get_snr_based_safe_rate(snr: float) -> int:
    for rate_idx in range(7, -1, -1):
        required_snr = SNR_THRESHOLDS[rate_idx] + 2
        if snr >= required_snr:
            return rate_idx
    return 0

def get_throughput_optimal_rate(row: pd.Series) -> int:
    snr = safe_float(row.get('lastSnr', 0))
    success_ratio = safe_float(row.get('shortSuccRatio', 1))
    best_rate, best_goodput = 0, 0.0
    for rate_idx in G_RATE_INDICES:
        p_success = estimate_success_probability(row, rate_idx)
        rate_bps = G_RATES_BPS[rate_idx]
        goodput = p_success * rate_bps
        if goodput > best_goodput:
            best_goodput = goodput
            best_rate = rate_idx
    return best_rate

def estimate_success_probability(row: pd.Series, target_rate_idx: int) -> float:
    snr = safe_float(row.get('lastSnr', 0))
    current_rate = clamp_rateidx(row.get('rateIdx', 0))
    current_success = safe_float(row.get('shortSuccRatio', 1))
    required_snr = SNR_THRESHOLDS[target_rate_idx]
    if target_rate_idx == current_rate:
        return current_success
    elif snr >= required_snr + 3:
        return 0.95
    elif snr >= required_snr:
        return 0.80
    elif snr >= required_snr - 2:
        return 0.60
    else:
        return 0.30

# ----------------------------------------
# SYNTHETIC EDGE CASES
# ----------------------------------------
def generate_critical_edge_cases(target_samples: int = 12000) -> pd.DataFrame:
    edge_cases: List[Dict[str, Any]] = []
    scenarios = [
        create_high_rate_failure_scenario,
        create_low_rate_recovery_scenario,
        create_snr_volatility_scenario,
        create_queue_saturation_scenario,
        create_mobility_spike_scenario,
        create_persistent_failure_scenario,
        create_sudden_improvement_scenario,
        create_consecutive_success_scenario,
        create_channel_width_change_scenario,
        create_rate_flip_flop_scenario,
        create_low_snr_success_scenario,
        create_high_snr_failure_scenario
    ]
    samples_per_scenario = target_samples // len(scenarios)
    for scenario_fn in scenarios:
        for _ in range(samples_per_scenario):
            edge_case = scenario_fn()
            edge_cases.append(edge_case)
    logger.info(f"Generated {len(edge_cases)} synthetic edge/corner cases across {len(scenarios)} scenarios.")
    return pd.DataFrame(edge_cases)

def create_high_rate_failure_scenario() -> Dict[str, Any]:
    rate = np.random.choice([5, 6, 7])
    snr = np.random.uniform(8, 15)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.2, 0.5),
        'consecFailure': np.random.randint(3, 8),
        'consecSuccess': 0,
        'severity': np.random.uniform(0.8, 1.0),
        'snrVariance': np.random.uniform(3, 8),
        'oracle_conservative': 0,
        'oracle_balanced': np.random.choice([0, 1, 2]),
        'oracle_aggressive': np.random.choice([1, 2, 3]),
        'network_context': 'emergency_recovery'
    }

def create_low_rate_recovery_scenario() -> Dict[str, Any]:
    rate = np.random.choice([0, 1, 2])
    snr = np.random.uniform(16, 28)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.8, 1.0),
        'consecFailure': 0,
        'consecSuccess': np.random.randint(5, 12),
        'severity': np.random.uniform(0.0, 0.2),
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': rate,
        'oracle_balanced': rate + 1 if rate < 6 else 7,
        'oracle_aggressive': rate + 2 if rate < 5 else 7,
        'network_context': 'good_stable'
    }

def create_snr_volatility_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(10, 28)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.5, 0.9),
        'consecFailure': np.random.randint(0, 5),
        'consecSuccess': np.random.randint(0, 5),
        'severity': np.random.uniform(0.2, 0.6),
        'snrVariance': np.random.uniform(5, 15),
        'oracle_conservative': get_snr_based_safe_rate(snr),
        'oracle_balanced': rate,
        'oracle_aggressive': min(rate + 1, 7),
        'network_context': 'poor_unstable'
    }

def create_queue_saturation_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(14, 25)
    queue_len = np.random.randint(50, 200)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.3, 0.7),
        'consecFailure': np.random.randint(0, 4),
        'consecSuccess': np.random.randint(0, 4),
        'severity': np.random.uniform(0.3, 0.9),
        'queueLen': queue_len,
        'snrVariance': np.random.uniform(0.5, 4.0),
        'oracle_conservative': clamp_rateidx(rate - 2),
        'oracle_balanced': clamp_rateidx(rate - 1),
        'oracle_aggressive': rate,
        'network_context': 'marginal_conditions'
    }

def create_mobility_spike_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(10, 24)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.4, 0.9),
        'consecFailure': np.random.randint(0, 6),
        'consecSuccess': np.random.randint(0, 6),
        'severity': np.random.uniform(0.4, 0.8),
        'mobilityMetric': np.random.uniform(15, 100),
        'snrVariance': np.random.uniform(2, 10),
        'oracle_conservative': get_snr_based_safe_rate(snr),
        'oracle_balanced': rate,
        'oracle_aggressive': min(rate + 1, 7),
        'network_context': 'poor_unstable'
    }

def create_persistent_failure_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(8, 16)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.1, 0.6),
        'consecFailure': np.random.randint(5, 12),
        'consecSuccess': 0,
        'severity': np.random.uniform(0.8, 1.0),
        'snrVariance': np.random.uniform(3, 9),
        'oracle_conservative': 0,
        'oracle_balanced': clamp_rateidx(rate - 2),
        'oracle_aggressive': clamp_rateidx(rate - 1),
        'network_context': 'emergency_recovery'
    }

def create_sudden_improvement_scenario() -> Dict[str, Any]:
    rate = np.random.choice([0, 1, 2])
    snr = np.random.uniform(22, 30)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.95, 1.00),
        'consecFailure': 0,
        'consecSuccess': np.random.randint(6, 15),
        'severity': 0.0,
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': rate,
        'oracle_balanced': min(rate + 2, 7),
        'oracle_aggressive': min(rate + 3, 7),
        'network_context': 'excellent_stable'
    }

def create_consecutive_success_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(18, 28)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.98, 1.00),
        'consecFailure': 0,
        'consecSuccess': np.random.randint(10, 20),
        'severity': 0.0,
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': rate,
        'oracle_balanced': min(rate + 1, 7),
        'oracle_aggressive': min(rate + 2, 7),
        'network_context': 'good_stable'
    }

def create_channel_width_change_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(12, 22)
    channel_width = np.random.choice([20, 40])
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.7, 0.98),
        'consecFailure': np.random.randint(0, 3),
        'consecSuccess': np.random.randint(0, 6),
        'severity': np.random.uniform(0.2, 0.7),
        'channelWidth': channel_width,
        'snrVariance': np.random.uniform(2, 7),
        'oracle_conservative': clamp_rateidx(rate - 2),
        'oracle_balanced': clamp_rateidx(rate - 1),
        'oracle_aggressive': rate,
        'network_context': 'good_unstable'
    }

def create_rate_flip_flop_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(12, 26)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.6, 0.9),
        'consecFailure': np.random.randint(1, 4),
        'consecSuccess': np.random.randint(1, 8),
        'severity': np.random.uniform(0.3, 0.8),
        'snrVariance': np.random.uniform(5, 15),
        'oracle_conservative': clamp_rateidx(rate - 1),
        'oracle_balanced': rate,
        'oracle_aggressive': min(rate + 1, 7),
        'network_context': 'poor_unstable'
    }

def create_low_snr_success_scenario() -> Dict[str, Any]:
    rate = 0
    snr = np.random.uniform(3, 8)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.8, 1.0),
        'consecFailure': 0,
        'consecSuccess': np.random.randint(5, 12),
        'severity': 0.0,
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': 0,
        'oracle_balanced': 1,
        'oracle_aggressive': 2,
        'network_context': 'emergency_recovery'
    }

def create_high_snr_failure_scenario() -> Dict[str, Any]:
    rate = np.random.choice([6, 7])
    snr = np.random.uniform(25, 32)
    return {
        'lastSnr': snr, 'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.0, 0.4),
        'consecFailure': np.random.randint(4, 9),
        'consecSuccess': 0,
        'severity': np.random.uniform(0.8, 1.0),
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': clamp_rateidx(rate - 3),
        'oracle_balanced': clamp_rateidx(rate - 2),
        'oracle_aggressive': clamp_rateidx(rate - 1),
        'network_context': 'marginal_conditions'
    }

# ----------------------------------------
# MAIN PIPELINE
# ----------------------------------------
def main():
    logger.info("=== ML Data Prep Script Started (WITH CLASS WEIGHTS) ===")
    if not os.path.exists(INPUT_CSV):
        logger.error(f"Input CSV does not exist: {INPUT_CSV}")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # --- Clean and validate ---
    df = clean_dataframe(df)
    df = filter_sane_rows(df)

    # --- Context classification and label creation ---
    logger.info("Classifying context and generating oracle labels...")
    df['network_context'] = df.apply(classify_network_context, axis=1)
    oracle_labels = []
    for idx, row in df.iterrows():
        current_rate = clamp_rateidx(row.get('rateIdx', 0))
        context = row['network_context']
        labels = create_context_specific_labels(row, context, current_rate)
        oracle_labels.append(labels)
        if idx % 100000 == 0 and idx > 0:
            logger.info(f"Processed {idx} rows for label creation...")
    oracle_df = pd.DataFrame(oracle_labels)
    df = pd.concat([df.reset_index(drop=True), oracle_df.reset_index(drop=True)], axis=1)
    logger.info(f"Added oracle labels and network context to dataframe.")

    # --- Synthetic Edge Case Generation ---
    logger.info("Generating synthetic edge/corner cases ...")
    synthetic_df = generate_critical_edge_cases(target_samples=12000)
    logger.info(f"Synthetic edge/corner cases shape: {synthetic_df.shape}")

    # --- Combine and output ---
    logger.info("Combining real and synthetic data ...")
    final_df = pd.concat([df, synthetic_df], ignore_index=True, sort=False)
    logger.info(f"Final dataframe shape: {final_df.shape}")

    # --- COMPUTE AND SAVE CLASS WEIGHTS (NEW ADDITION) ---
    weights_output_dir = os.path.join(BASE_DIR, "model_artifacts")
    oracle_labels = ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive']
    # Also compute weights for rateIdx (the main target)
    all_label_cols = oracle_labels + ['rateIdx']
    class_weights = compute_and_save_class_weights(final_df, all_label_cols, weights_output_dir)

    # --- Save ---
    try:
        final_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"ML-enriched CSV exported: {OUTPUT_CSV} (rows: {final_df.shape[0]})")
        print(f"\nML-enriched CSV exported: {OUTPUT_CSV} (rows: {final_df.shape[0]}, cols: {final_df.shape[1]})")
    except Exception as e:
        logger.error(f"Failed to save output CSV: {e}")
        sys.exit(1)

    # --- Stats and Summary ---
    print("\n--- LABEL DISTRIBUTION ---")
    for lbl in ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive']:
        if lbl in final_df.columns:
            vc = final_df[lbl].value_counts()
            print(f"{lbl}:\n{vc}\n")
            logger.info(f"{lbl} value counts:\n{vc}")

    print("\n--- NETWORK CONTEXT DISTRIBUTION ---")
    nc_vc = final_df['network_context'].value_counts()
    print(nc_vc)
    logger.info(f"Network context distribution:\n{nc_vc}")

    # Feature stats
    print("\n--- FEATURE STATISTICS ---")
    feature_cols = [c for c in final_df.columns if c not in ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive', 'network_context', 'scenario_file']]
    stats_df = final_df[feature_cols].describe(include='all').transpose()
    print(stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].fillna('N/A'))
    logger.info(f"Feature statistics:\n{stats_df}")

    logger.info("=== ML Data Prep Script Finished ===")

if __name__ == "__main__":
    main()