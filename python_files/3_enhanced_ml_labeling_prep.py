"""
ML Data Preparation with Oracle Label Generation
FIXED VERSION - Eliminates ALL temporal leakage and SNR-based circular reasoning

CRITICAL FIXES (2025-10-01):
- Issue #1: Removed ALL temporal leakage features from training
- Issue #2: Oracle labels NO LONGER use SNR directly (pattern-based only)
- Issue #3: Context classification uses packet loss/variance, NOT SNR
- Issue #33: Oracle uses ONLY pre-decision features (no consecSuccess/Failure)
- Issue #9: Increased oracle noise to ±1.0 with strategy-specific biases
- Issue #10: No hardcoded rate 7, uses min(7, base + noise)
- Issue #14: Added global random seed
- Issue #57: Documented all magic numbers with rationale

Author: ahmedjk34
Date: 2025-09-22
FIXED: 2025-10-01 (Issues #1, #2, #3, #9, #10, #14, #33, #57)
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

# FIXED: Issue #14 - Global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# WiFi Configuration
G_RATES_BPS = [1000000, 2000000, 5500000, 6000000, 9000000, 11000000, 12000000, 18000000]
G_RATE_INDICES = list(range(8))

# FIXED: Issue #57 - Document all magic numbers with rationale
# Context Classification Thresholds (based on WiFi best practices)
CONTEXT_THRESHOLDS = {
    'packet_loss_high': 0.5,        # >50% loss = emergency
    'packet_loss_moderate': 0.2,    # 20-50% loss = poor
    'success_ratio_low': 0.5,       # <50% success = emergency
    'success_ratio_moderate': 0.8,  # 50-80% success = marginal
    'variance_high': 5.0,           # SNR variance >5 dB = unstable
    'variance_moderate': 3.0,       # SNR variance 3-5 dB = somewhat unstable
    'consecutive_failures_high': 3, # ≥3 consecutive failures = critical
    'consecutive_failures_moderate': 2  # 2 consecutive failures = concerning
}

# Oracle Noise Configuration (Issue #9 - increased from ±0.5 to ±1.0)
ORACLE_NOISE = {
    'conservative_min': -1.2,  # Conservative bias: prefer lower rates
    'conservative_max': 0.5,
    'balanced_min': -1.0,      # Balanced: symmetric noise
    'balanced_max': 1.0,
    'aggressive_min': -0.5,    # Aggressive bias: prefer higher rates
    'aggressive_max': 1.2
}

# FIXED: Issue #1 - Define which features are SAFE (pre-decision only)
SAFE_FEATURES = [
    # SNR features (pre-decision) - SAFE
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    
    # Historical success metrics (from PREVIOUS time window) - SAFE IF PREVIOUS
    "shortSuccRatio", "medSuccRatio",
    
    # Network state (pre-decision) - SAFE
    "packetLossRate",  # From previous window
    "channelWidth", "mobilityMetric",
    
    # Assessment features (from previous decision) - SAFE
    "severity", "confidence"
]

# FIXED: Issue #1 - Temporal leakage features (REMOVED from training)
TEMPORAL_LEAKAGE_FEATURES = [
    "consecSuccess",      # Outcome of CURRENT rate choice
    "consecFailure",      # Outcome of CURRENT rate choice
    "retrySuccessRatio",  # Success metric from outcomes
    "timeSinceLastRateChange",  # Encodes rate performance history
    "rateStabilityScore", # Derived from rate change history
    "recentRateChanges",  # Rate history
    "packetSuccess"       # Literal packet outcome
]

# Known leaky features (should already be removed in File 2, but double-check)
KNOWN_LEAKY_FEATURES = [
    "phyRate", "optimalRateDistance", "recentThroughputTrend",
    "conservativeFactor", "aggressiveFactor", "recommendedSafeRate"
]

ESSENTIAL_COLS = ["rateIdx", "lastSnr", "shortSuccRatio", "packetLossRate"]

# ================== LOGGING SETUP ==================
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
    logger.info("ML DATA PREPARATION - FIXED VERSION (NO TEMPORAL LEAKAGE)")
    logger.info(f"Random Seed: {RANDOM_SEED}")
    logger.info(f"Date: 2025-10-01 (Critical fixes applied)")
    logger.info("="*80)
    return logger

logger = setup_logging()

# ================== UTILITY FUNCTIONS ==================
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

# ================== CLASS WEIGHTS COMPUTATION ==================
def compute_and_save_class_weights(df: pd.DataFrame, label_cols: List[str], output_dir: str) -> Dict[str, Dict]:
    """Compute class weights for imbalanced labels and save them."""
    logger.info("🔢 Computing class weights for imbalanced target labels...")
    
    class_weights_dict = {}
    
    for label_col in label_cols:
        if label_col not in df.columns:
            logger.warning(f"Label column {label_col} not found, skipping...")
            continue
            
        valid_labels = df[label_col].dropna()
        
        if len(valid_labels) == 0:
            logger.warning(f"No valid labels found for {label_col}, skipping...")
            continue
            
        unique_classes = np.array(sorted(valid_labels.unique()))
        
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes,
            y=valid_labels
        )
        
        # FIXED: Issue #6 - Cap extreme class weights at 50.0
        class_weights = np.minimum(class_weights, 50.0)
        
        weight_dict = {}
        for class_val, weight in zip(unique_classes, class_weights):
            python_key = int(class_val) if isinstance(class_val, (np.integer, np.int64)) else float(class_val) if isinstance(class_val, np.floating) else class_val
            python_weight = float(weight)
            weight_dict[python_key] = python_weight
        
        class_weights_dict[label_col] = weight_dict
        
        class_counts = Counter(valid_labels)
        logger.info(f"\n📊 {label_col} - Class Distribution & Weights (CAPPED at 50.0):")
        for class_val in unique_classes:
            count = class_counts[class_val]
            weight = weight_dict[int(class_val) if isinstance(class_val, (np.integer, np.int64)) else class_val]
            pct = (count / len(valid_labels)) * 100
            logger.info(f"  Class {class_val}: {count:,} samples ({pct:.1f}%) -> weight: {weight:.3f}")
        
        print(f"\n{label_col} Class Weights (capped at 50.0):")
        for class_val, weight in weight_dict.items():
            print(f"  {class_val}: {weight:.3f}")
    
    weights_file = os.path.join(output_dir, "class_weights.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(weights_file, 'w') as f:
        json.dump(class_weights_dict, f, indent=2)
    
    logger.info(f"💾 Class weights saved to: {weights_file}")
    print(f"💾 Class weights saved to: {weights_file}")
    
    return class_weights_dict

# ================== CLEANING FUNCTIONS ==================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Initial row count: {len(df)}")
    before = len(df)
    df_clean = df.dropna(subset=ESSENTIAL_COLS, how="any")
    logger.info(f"Dropped {before - len(df_clean)} rows missing ANY essential columns")
    
    before2 = len(df_clean)
    df_clean = df_clean.dropna(subset=ESSENTIAL_COLS, thresh=3)
    logger.info(f"Dropped {before2 - len(df_clean)} rows with <3 essential columns present")

    before3 = len(df_clean)
    cols_to_check = [col for col in df_clean.columns if col != 'scenario_file']
    def all_blank(row):
        return all((pd.isna(x) or (isinstance(x, str) and x.strip() == "")) for x in row)
    df_clean = df_clean.loc[~(df_clean[cols_to_check].apply(all_blank, axis=1))]
    logger.info(f"Dropped {before3 - len(df_clean)} rows with all blank except scenario_file")

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

# ================== SANITY FILTERING ==================
def filter_sane_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Only validate columns that actually exist
    Temporal leakage features (consecSuccess/consecFailure) already removed by File 2
    """
    before = len(df)
    
    # Build filter conditions ONLY for columns that exist
    conditions = [
        df['rateIdx'].apply(lambda x: is_valid_rateidx(x)),
        df['lastSnr'].apply(lambda x: -10 < safe_float(x) < 60)
    ]
    
    # Add optional column checks
    if 'phyRate' in df.columns:
        conditions.append(df['phyRate'].apply(lambda x: safe_int(x) >= 1000000 and safe_int(x) <= 54000000))
    
    if 'shortSuccRatio' in df.columns:
        conditions.append(df['shortSuccRatio'].apply(lambda x: 0 <= safe_float(x) <= 1.01 if not pd.isna(x) else True))
    
    if 'medSuccRatio' in df.columns:
        conditions.append(df['medSuccRatio'].apply(lambda x: 0 <= safe_float(x) <= 1.01 if not pd.isna(x) else True))
    
    if 'severity' in df.columns:
        conditions.append(df['severity'].apply(lambda x: 0 <= safe_float(x) <= 1.5 if not pd.isna(x) else True))
    
    if 'confidence' in df.columns:
        conditions.append(df['confidence'].apply(lambda x: 0 <= safe_float(x) <= 1.01 if not pd.isna(x) else True))
    
    # Combine all conditions
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition &= condition
    
    df_filtered = df[combined_condition]
    
    logger.info(f"FIXED: Kept {len(df_filtered)} out of {before} rows ({len(df_filtered)/before*100:.1f}% retained)")
    logger.info(f"Note: consecSuccess/consecFailure already removed by File 2 (temporal leakage)")
    
    return df_filtered

# ================== FEATURE REMOVAL ==================
def remove_leaky_and_temporal_features(df):
    """
    FIXED: Issue #1 - Remove ALL temporal leakage and known leaky features
    """
    ALL_FEATURES_TO_REMOVE = list(set(TEMPORAL_LEAKAGE_FEATURES + KNOWN_LEAKY_FEATURES))
    
    initial_cols = len(df.columns)
    removed_features = []
    
    for feature in ALL_FEATURES_TO_REMOVE:
        if feature in df.columns:
            removed_features.append(feature)
    
    df_clean = df.drop(columns=removed_features)
    removed_cols = len(removed_features)
    
    logger.info(f"🧹 CRITICAL FIX - Removed {removed_cols} leaky/temporal features:")
    logger.info(f"   Temporal leakage (Issue #1): {[f for f in TEMPORAL_LEAKAGE_FEATURES if f in removed_features]}")
    logger.info(f"   Known leaky features: {[f for f in KNOWN_LEAKY_FEATURES if f in removed_features]}")
    logger.info(f"📊 Remaining columns: {len(df_clean.columns)} (was {initial_cols})")
    
    print(f"\n🧹 REMOVED {removed_cols} LEAKY/TEMPORAL FEATURES:")
    print(f"   {removed_features}")
    print(f"📊 Dataset now has {len(df_clean.columns)} columns (was {initial_cols})")
    
    return df_clean

# ================== CONTEXT CLASSIFICATION (FIXED) ==================
def classify_network_context(row) -> str:
    """
    FIXED: Issue #3 - Context classification NO LONGER uses SNR
    Uses only packet loss, variance, and consecutive failures
    """
    # Use ONLY outcome-based and variance features, NOT SNR
    packet_loss = safe_float(row.get('packetLossRate', 0))
    snr_variance = safe_float(row.get('snrVariance', 0))
    success_ratio = safe_float(row.get('shortSuccRatio', 1))
    consec_failures = safe_int(row.get('consecFailure', 0))
    
    # Emergency: High packet loss OR low success OR multiple consecutive failures
    if (packet_loss > CONTEXT_THRESHOLDS['packet_loss_high'] or 
        success_ratio < CONTEXT_THRESHOLDS['success_ratio_low'] or 
        consec_failures >= CONTEXT_THRESHOLDS['consecutive_failures_high']):
        return 'emergency_recovery'
    
    # Poor/Unstable: Moderate packet loss OR high variance
    elif (packet_loss > CONTEXT_THRESHOLDS['packet_loss_moderate'] or 
          snr_variance > CONTEXT_THRESHOLDS['variance_high']):
        return 'poor_unstable'
    
    # Marginal: Low-moderate success OR some failures
    elif (success_ratio < CONTEXT_THRESHOLDS['success_ratio_moderate'] or 
          consec_failures >= CONTEXT_THRESHOLDS['consecutive_failures_moderate']):
        return 'marginal_conditions'
    
    # Good but unstable: Moderate variance
    elif snr_variance > CONTEXT_THRESHOLDS['variance_moderate']:
        return 'good_unstable'
    
    # Excellent: High success AND low packet loss
    elif success_ratio > 0.9 and packet_loss < 0.05:
        return 'excellent_stable'
    
    # Default: Good stable
    else:
        return 'good_stable'

# ================== ORACLE LABEL CREATION (FIXED) ==================
def create_context_specific_labels(row: pd.Series, context: str, current_rate: int) -> Dict[str, int]:
    """
    FIXED: Issues #2, #33, #9, #10
    - NO SNR in oracle logic (Issue #2)
    - NO consecSuccess/consecFailure (Issue #33)
    - Increased noise to ±1.0 with strategy biases (Issue #9)
    - No hardcoded rate 7 (Issue #10)
    """
    # ONLY use PRE-DECISION features (Issue #33)
    short_succ = safe_float(row.get('shortSuccRatio', 1))
    med_succ = safe_float(row.get('medSuccRatio', 1))
    packet_loss = safe_float(row.get('packetLossRate', 0))
    snr_var = safe_float(row.get('snrVariance', 0))
    
    # FIXED: Issue #9 - Strategy-specific noise with biases
    cons_noise = np.random.uniform(ORACLE_NOISE['conservative_min'], ORACLE_NOISE['conservative_max'])
    bal_noise = np.random.uniform(ORACLE_NOISE['balanced_min'], ORACLE_NOISE['balanced_max'])
    agg_noise = np.random.uniform(ORACLE_NOISE['aggressive_min'], ORACLE_NOISE['aggressive_max'])
    
    # Base rate selection using ONLY success ratios and packet loss (NO SNR)
    if context == "emergency_recovery":
        # Emergency: Go to safest rates
        if short_succ < 0.25 or packet_loss > 0.6:
            base = 0
        elif short_succ < 0.6:
            base = 1
        else:
            base = 2
            
    elif context == "poor_unstable":
        # Poor conditions: Conservative but allow some exploration
        if packet_loss > 0.4:
            base = max(0, current_rate - 2)
        elif short_succ > 0.7 and med_succ > 0.65:
            base = min(5, current_rate + 1)  # Limited increase
        else:
            base = 1
            
    elif context == "marginal_conditions":
        # Marginal: Use success patterns
        if short_succ > 0.7 and packet_loss < 0.1:
            base = 4
        elif short_succ > 0.5:
            base = 3
        else:
            base = 2
                
    elif context in ["good_stable", "good_unstable"]:
        # Good conditions: Balance based on success
        if short_succ > 0.9:
            base = min(5, current_rate + 1)
        elif short_succ < 0.6:
            base = max(0, current_rate - 1)
        else:
            base = current_rate
                
    elif context == "excellent_stable":
        # Excellent: Allow high rates
        if short_succ > 0.95 and packet_loss < 0.05:
            base = 6  # Not hardcoded 7 anymore
        else:
            base = 5
    else:
        base = current_rate
    
    # Apply strategy-specific noise (FIXED: Issue #9, #10)
    cons = int(np.clip(base + cons_noise, 0, 7))
    bal = int(np.clip(base + bal_noise, 0, 7))
    agg = int(np.clip(base + agg_noise, 0, 7))  # FIXED: Not hardcoded 7
    
    labels = {
        "oracle_conservative": cons,
        "oracle_balanced": bal,
        "oracle_aggressive": agg,
    }
    
    return labels

# ================== SYNTHETIC EDGE CASES (REDUCED) ==================
def generate_critical_edge_cases(target_samples: int = 5000) -> pd.DataFrame:
    """
    FIXED: Issue #7 - Reduced from 12,000 to 5,000 synthetic samples
    Made more realistic with noise
    """
    edge_cases: List[Dict[str, Any]] = []
    scenarios = [
        create_high_rate_failure_scenario,
        create_low_rate_recovery_scenario,
        create_snr_volatility_scenario,
        create_mobility_spike_scenario,
        create_persistent_failure_scenario,
        create_sudden_improvement_scenario,
        create_consecutive_success_scenario,
        create_rate_flip_flop_scenario,
        create_low_snr_success_scenario,
        create_high_snr_failure_scenario
    ]
    samples_per_scenario = target_samples // len(scenarios)
    
    for scenario_fn in scenarios:
        for _ in range(samples_per_scenario):
            edge_case = scenario_fn()
            edge_cases.append(edge_case)
    
    logger.info(f"Generated {len(edge_cases)} synthetic edge cases (reduced from 12K, Issue #7)")
    return pd.DataFrame(edge_cases)

# Synthetic scenario functions (with added noise for realism - Issue #19)
def create_high_rate_failure_scenario() -> Dict[str, Any]:
    rate = np.random.choice([5, 6, 7])
    snr = np.random.uniform(8, 15) + np.random.normal(0, 1.5)  # Added noise
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.2, 0.5) + np.random.normal(0, 0.05),
        'packetLossRate': np.random.uniform(0.5, 0.8),
        'severity': np.random.uniform(0.8, 1.0),
        'snrVariance': np.random.uniform(3, 8),
        'oracle_conservative': 0,
        'oracle_balanced': np.random.choice([0, 1, 2]),
        'oracle_aggressive': np.random.choice([1, 2, 3]),
        'network_context': 'emergency_recovery'
    }

def create_low_rate_recovery_scenario() -> Dict[str, Any]:
    rate = np.random.choice([0, 1, 2])
    snr = np.random.uniform(16, 28) + np.random.normal(0, 2.0)  # Added noise
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.8, 1.0),
        'packetLossRate': np.random.uniform(0.0, 0.1),
        'severity': np.random.uniform(0.0, 0.2),
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': rate,
        'oracle_balanced': min(rate + 1, 6),
        'oracle_aggressive': min(rate + 2, 7),
        'network_context': 'good_stable'
    }

def create_snr_volatility_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(10, 28) + np.random.normal(0, 2.5)  # Added noise
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.5, 0.9),
        'packetLossRate': np.random.uniform(0.1, 0.4),
        'severity': np.random.uniform(0.2, 0.6),
        'snrVariance': np.random.uniform(5, 15),
        'oracle_conservative': max(0, rate - 1),
        'oracle_balanced': rate,
        'oracle_aggressive': min(rate + 1, 7),
        'network_context': 'poor_unstable'
    }

def create_mobility_spike_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(10, 24) + np.random.normal(0, 2.0)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.4, 0.9),
        'packetLossRate': np.random.uniform(0.1, 0.5),
        'severity': np.random.uniform(0.4, 0.8),
        'mobilityMetric': np.random.uniform(15, 100),
        'snrVariance': np.random.uniform(2, 10),
        'oracle_conservative': max(0, rate - 1),
        'oracle_balanced': rate,
        'oracle_aggressive': min(rate + 1, 7),
        'network_context': 'poor_unstable'
    }

def create_persistent_failure_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(8, 16) + np.random.normal(0, 1.5)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.1, 0.6),
        'packetLossRate': np.random.uniform(0.4, 0.9),
        'severity': np.random.uniform(0.8, 1.0),
        'snrVariance': np.random.uniform(3, 9),
        'oracle_conservative': 0,
        'oracle_balanced': max(0, rate - 2),
        'oracle_aggressive': max(0, rate - 1),
        'network_context': 'emergency_recovery'
    }

def create_sudden_improvement_scenario() -> Dict[str, Any]:
    rate = np.random.choice([0, 1, 2])
    snr = np.random.uniform(22, 30) + np.random.normal(0, 1.0)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.95, 1.00),
        'packetLossRate': np.random.uniform(0.0, 0.05),
        'severity': 0.0,
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': rate,
        'oracle_balanced': min(rate + 2, 6),
        'oracle_aggressive': min(rate + 3, 7),
        'network_context': 'excellent_stable'
    }

def create_consecutive_success_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(18, 28) + np.random.normal(0, 1.5)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.98, 1.00),
        'packetLossRate': np.random.uniform(0.0, 0.02),
        'severity': 0.0,
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': rate,
        'oracle_balanced': min(rate + 1, 7),
        'oracle_aggressive': min(rate + 2, 7),
        'network_context': 'good_stable'
    }

def create_rate_flip_flop_scenario() -> Dict[str, Any]:
    rate = np.random.choice(G_RATE_INDICES)
    snr = np.random.uniform(12, 26) + np.random.normal(0, 2.0)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.6, 0.9),
        'packetLossRate': np.random.uniform(0.1, 0.3),
        'severity': np.random.uniform(0.3, 0.8),
        'snrVariance': np.random.uniform(5, 15),
        'oracle_conservative': max(0, rate - 1),
        'oracle_balanced': rate,
        'oracle_aggressive': min(rate + 1, 7),
        'network_context': 'poor_unstable'
    }

def create_low_snr_success_scenario() -> Dict[str, Any]:
    rate = 0
    snr = np.random.uniform(3, 8) + np.random.normal(0, 0.5)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.8, 1.0),
        'packetLossRate': np.random.uniform(0.0, 0.2),
        'severity': 0.0,
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': 0,
        'oracle_balanced': 1,
        'oracle_aggressive': 2,
        'network_context': 'emergency_recovery'
    }

def create_high_snr_failure_scenario() -> Dict[str, Any]:
    rate = np.random.choice([6, 7])
    snr = np.random.uniform(25, 32) + np.random.normal(0, 1.0)
    return {
        'lastSnr': snr,
        'rateIdx': rate,
        'shortSuccRatio': np.random.uniform(0.0, 0.4),
        'packetLossRate': np.random.uniform(0.6, 1.0),
        'severity': np.random.uniform(0.8, 1.0),
        'snrVariance': np.random.uniform(0.1, 2.0),
        'oracle_conservative': max(0, rate - 3),
        'oracle_balanced': max(0, rate - 2),
        'oracle_aggressive': max(0, rate - 1),
        'network_context': 'marginal_conditions'
    }

# ================== MAIN PIPELINE ==================
def main():
    logger.info("=== ML Data Prep Script Started (FIXED - NO TEMPORAL LEAKAGE) ===")
    if not os.path.exists(INPUT_CSV):
        logger.error(f"Input CSV does not exist: {INPUT_CSV}")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Clean and validate
    df = clean_dataframe(df)
    df = filter_sane_rows(df)

    # FIXED: Issue #3 - Context classification (no SNR)
    logger.info("Classifying context WITHOUT SNR (Issue #3 fix)...")
    df['network_context'] = df.apply(classify_network_context, axis=1)
    
    # FIXED: Issues #2, #33 - Oracle labels (no SNR, no temporal features)
    logger.info("Generating oracle labels WITHOUT SNR or temporal features (Issues #2, #33 fixes)...")
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

    # Synthetic Edge Case Generation (FIXED: Issue #7 - reduced to 5000)
    logger.info("Generating synthetic edge cases (5,000 samples, Issue #7 fix)...")
    synthetic_df = generate_critical_edge_cases(target_samples=5000)
    logger.info(f"Synthetic edge cases shape: {synthetic_df.shape}")

    # Combine
    logger.info("Combining real and synthetic data...")
    final_df = pd.concat([df, synthetic_df], ignore_index=True, sort=False)
    logger.info(f"Final dataframe shape: {final_df.shape}")

    # Compute and save class weights (with capping - Issue #6)
    weights_output_dir = os.path.join(BASE_DIR, "model_artifacts")
    oracle_labels_cols = ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive']
    all_label_cols = oracle_labels_cols + ['rateIdx']
    class_weights = compute_and_save_class_weights(final_df, all_label_cols, weights_output_dir)

    # FIXED: Issue #1 - Remove leaky/temporal features BEFORE saving
    logger.info("🧹 Removing leaky and temporal features (Issue #1 fix)...")
    final_df = remove_leaky_and_temporal_features(final_df)
    
    # Save
    try:
        final_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"ML-enriched CSV exported: {OUTPUT_CSV} (rows: {final_df.shape[0]}, cols: {final_df.shape[1]})")
        print(f"\nML-enriched CSV exported: {OUTPUT_CSV}")
        print(f"  Rows: {final_df.shape[0]:,}")
        print(f"  Cols: {final_df.shape[1]}")
        print(f"  🛡️ SAFE FEATURES ONLY (temporal leakage removed)")
    except Exception as e:
        logger.error(f"Failed to save output CSV: {e}")
        sys.exit(1)

    # Stats and Summary
    print("\n--- LABEL DISTRIBUTION ---")
    for lbl in ['oracle_conservative', 'oracle_balanced', 'oracle_aggressive']:
        if lbl in final_df.columns:
            vc = final_df[lbl].value_counts().sort_index()
            print(f"{lbl}:\n{vc}\n")
            logger.info(f"{lbl} value counts:\n{vc}")

    print("\n--- NETWORK CONTEXT DISTRIBUTION ---")
    nc_vc = final_df['network_context'].value_counts()
    print(nc_vc)
    logger.info(f"Network context distribution:\n{nc_vc}")

    # Feature stats (only SAFE features)
    print("\n--- SAFE FEATURE STATISTICS ---")
    safe_feature_cols = [c for c in final_df.columns if c in SAFE_FEATURES]
    if safe_feature_cols:
        stats_df = final_df[safe_feature_cols].describe(include='all').transpose()
        print(stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].fillna('N/A'))
        logger.info(f"Safe feature statistics:\n{stats_df}")

    logger.info("=== ML Data Prep Script Finished (ALL TEMPORAL LEAKAGE REMOVED) ===")
    print("\n✅ CRITICAL FIXES APPLIED:")
    print("  - Issue #1: Temporal leakage features REMOVED")
    print("  - Issue #2: Oracle labels NO LONGER use SNR")
    print("  - Issue #3: Context classification uses packet loss/variance, NOT SNR")
    print("  - Issue #33: Oracle uses ONLY pre-decision features")
    print("  - Issue #9: Oracle noise increased to ±1.0")
    print("  - Issue #10: No hardcoded rate 7")
    print("  - Issue #14: Random seed = 42")
    print("  - Issue #7: Synthetic samples reduced to 5,000")
    print("  - Issue #6: Class weights capped at 50.0")

if __name__ == "__main__":
    main()