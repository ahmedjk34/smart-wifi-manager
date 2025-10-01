"""
Data Leakage Validation Script - ENHANCED VERSION
Validates that no data leakage exists in the cleaned dataset before training.
NOW DETECTS: Temporal leakage, SNR-based circular reasoning, and correlation-based leakage

Author: ahmedjk34
Date: 2025-09-28
FIXED: 2025-10-01 (Enhanced validation for Issues #1, #2, #3, #18, #33)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
INPUT_CSV = "smart-v3-ml-enriched.csv"

# FIXED: Issue #18 - Lowered correlation threshold from 0.8 to 0.4
CORRELATION_THRESHOLD = 0.4  # Catches temporal leakage (0.3-0.5 correlations)
HIGH_CORRELATION_THRESHOLD = 0.7  # For critical warnings

AVAILABLE_TARGETS = ["rateIdx", "oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# FIXED: Issue #1 - Define temporal leakage features (should be ABSENT)
TEMPORAL_LEAKAGE_FEATURES = [
    "consecSuccess",      # Outcome of CURRENT rate choice
    "consecFailure",      # Outcome of CURRENT rate choice
    "retrySuccessRatio",  # Success metric from outcomes
    "timeSinceLastRateChange",  # Encodes rate performance history
    "rateStabilityScore", # Derived from rate change history
    "recentRateChanges",  # Rate history
    "packetSuccess"       # Literal packet outcome
]

# Known leaky features (should be ABSENT)
KNOWN_LEAKY_FEATURES = [
    "phyRate",               # Perfect correlation with rateIdx
    "optimalRateDistance",   # 8 unique values = 8 rate classes
    "recentThroughputTrend", # High correlation (0.853)
    "conservativeFactor",    # Inverse correlation (-0.809)
    "aggressiveFactor",      # Inverse of conservative
    "recommendedSafeRate"    # Direct target hint
]

# Constant/useless features (should be ABSENT)
USELESS_FEATURES = [
    "T1", "T2", "T3",        # Always constant
    "decisionReason",         # Always 0
    "offeredLoad",            # Always 0 in data
    "queueLen",               # Mostly 0 in data
    "retryCount"              # Always 0 in data
]

# Safe features (SHOULD be present)
SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort",
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "shortSuccRatio", "medSuccRatio",
    "packetLossRate",
    "channelWidth", "mobilityMetric",
    "severity", "confidence"
]

# FIXED: Issue #3 - Context should NOT correlate highly with SNR
CONTEXT_LABEL = "network_context"

# ================== VALIDATION FUNCTIONS ==================

def check_temporal_leakage_removal(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    FIXED: Issue #1 - Verify ALL temporal leakage features are removed
    """
    print("\n" + "="*80)
    print("1. TEMPORAL LEAKAGE VALIDATION (Issue #1)")
    print("="*80)
    
    found_temporal = []
    for feature in TEMPORAL_LEAKAGE_FEATURES:
        if feature in df.columns:
            found_temporal.append(feature)
            print(f"❌ CRITICAL: Temporal leakage feature '{feature}' still present!")
        else:
            print(f"✅ Temporal feature '{feature}' properly removed")
    
    if found_temporal:
        print(f"\n🚨 VALIDATION FAILED: {len(found_temporal)} temporal leakage features found!")
        return False, found_temporal
    else:
        print(f"\n✅ PASS: All {len(TEMPORAL_LEAKAGE_FEATURES)} temporal features removed")
        return True, []

def check_known_leaky_features_removal(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify known leaky features are removed"""
    print("\n" + "="*80)
    print("2. KNOWN LEAKY FEATURES VALIDATION")
    print("="*80)
    
    found_leaky = []
    for feature in KNOWN_LEAKY_FEATURES:
        if feature in df.columns:
            found_leaky.append(feature)
            print(f"❌ CRITICAL: Leaky feature '{feature}' still present!")
        else:
            print(f"✅ Leaky feature '{feature}' properly removed")
    
    if found_leaky:
        print(f"\n🚨 VALIDATION FAILED: {len(found_leaky)} leaky features found!")
        return False, found_leaky
    else:
        print(f"\n✅ PASS: All {len(KNOWN_LEAKY_FEATURES)} leaky features removed")
        return True, []

def check_useless_features_removal(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    FIXED: Issue #28 - Verify useless features are removed early
    """
    print("\n" + "="*80)
    print("3. USELESS/CONSTANT FEATURES VALIDATION (Issue #28)")
    print("="*80)
    
    found_useless = []
    for feature in USELESS_FEATURES:
        if feature in df.columns:
            found_useless.append(feature)
            unique_count = df[feature].nunique()
            print(f"⚠️ WARNING: Useless feature '{feature}' still present ({unique_count} unique values)")
        else:
            print(f"✅ Useless feature '{feature}' properly removed")
    
    if found_useless:
        print(f"\n⚠️ WARNING: {len(found_useless)} useless features found (not critical)")
        return True, found_useless  # Warning, not failure
    else:
        print(f"\n✅ PASS: All {len(USELESS_FEATURES)} useless features removed")
        return True, []

def check_safe_features_present(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify safe features are still present"""
    print("\n" + "="*80)
    print("4. SAFE FEATURES PRESENCE VALIDATION")
    print("="*80)
    
    missing_safe = []
    for feature in SAFE_FEATURES:
        if feature not in df.columns:
            missing_safe.append(feature)
            print(f"⚠️ WARNING: Safe feature '{feature}' is missing!")
        else:
            print(f"✅ Safe feature '{feature}' present")
    
    if missing_safe:
        print(f"\n⚠️ WARNING: {len(missing_safe)} safe features missing")
        return True, missing_safe  # Warning, not failure
    else:
        print(f"\n✅ PASS: All {len(SAFE_FEATURES)} safe features present")
        return True, []

def check_feature_target_correlations(df: pd.DataFrame, target: str, 
                                     correlation_threshold: float = CORRELATION_THRESHOLD,
                                     high_threshold: float = HIGH_CORRELATION_THRESHOLD) -> Tuple[bool, List[Tuple[str, float]]]:
    """
    FIXED: Issue #18 - Lowered threshold to 0.4 to catch temporal leakage
    Checks for suspicious correlations between features and target
    """
    print("\n" + "="*80)
    print(f"5. FEATURE-TARGET CORRELATION VALIDATION (Issue #18)")
    print(f"   Target: {target}")
    print(f"   Thresholds: Moderate={correlation_threshold}, High={high_threshold}")
    print("="*80)
    
    if target not in df.columns:
        print(f"⚠️ Target '{target}' not found in dataset")
        return True, []
    
    suspicious_corrs = []
    critical_corrs = []
    
    # Check correlations for all remaining features
    for col in df.columns:
        if col in [target] + AVAILABLE_TARGETS + [CONTEXT_LABEL, 'scenario_file']:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            try:
                corr = df[col].corr(df[target])
                
                if pd.notnull(corr):
                    abs_corr = abs(corr)
                    
                    if abs_corr > high_threshold:
                        critical_corrs.append((col, corr))
                        print(f"🚨 CRITICAL: {col} has correlation {corr:.3f} with {target}")
                    elif abs_corr > correlation_threshold:
                        suspicious_corrs.append((col, corr))
                        print(f"⚠️ SUSPICIOUS: {col} has correlation {corr:.3f} with {target}")
                    else:
                        print(f"✅ Safe: {col} = {corr:.3f}")
            except Exception as e:
                print(f"⚠️ Could not compute correlation for {col}: {e}")
    
    if critical_corrs:
        print(f"\n🚨 VALIDATION FAILED: {len(critical_corrs)} critical correlations (>{high_threshold})")
        for feat, corr in critical_corrs:
            print(f"   {feat}: {corr:.3f}")
        return False, critical_corrs + suspicious_corrs
    elif suspicious_corrs:
        print(f"\n⚠️ WARNING: {len(suspicious_corrs)} suspicious correlations (>{correlation_threshold})")
        for feat, corr in suspicious_corrs:
            print(f"   {feat}: {corr:.3f}")
        return True, suspicious_corrs
    else:
        print(f"\n✅ PASS: No concerning correlations found")
        return True, []

def check_context_snr_independence(df: pd.DataFrame) -> Tuple[bool, float]:
    """
    FIXED: Issue #3 - Verify context classification is NOT SNR-dependent
    """
    print("\n" + "="*80)
    print("6. CONTEXT-SNR INDEPENDENCE VALIDATION (Issue #3)")
    print("="*80)
    
    if CONTEXT_LABEL not in df.columns:
        print(f"⚠️ Context label '{CONTEXT_LABEL}' not found")
        return True, 0.0
    
    if 'lastSnr' not in df.columns:
        print(f"⚠️ SNR feature 'lastSnr' not found")
        return True, 0.0
    
    # Encode context as numeric for correlation
    context_encoded = pd.factorize(df[CONTEXT_LABEL])[0]
    snr_corr = pd.Series(context_encoded).corr(df['lastSnr'])
    
    print(f"📊 Context-SNR correlation: {snr_corr:.3f}")
    
    if abs(snr_corr) > 0.5:
        print(f"🚨 CRITICAL: Context is highly correlated with SNR ({snr_corr:.3f})!")
        print(f"   This indicates SNR is being used in context classification")
        return False, snr_corr
    elif abs(snr_corr) > 0.3:
        print(f"⚠️ WARNING: Context has moderate correlation with SNR ({snr_corr:.3f})")
        return True, snr_corr
    else:
        print(f"✅ PASS: Context is independent of SNR ({snr_corr:.3f})")
        return True, snr_corr

def check_oracle_label_quality(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    FIXED: Issue #31, #32 - Check oracle label quality and distribution
    """
    print("\n" + "="*80)
    print("7. ORACLE LABEL QUALITY VALIDATION (Issues #31, #32)")
    print("="*80)
    
    oracle_labels = [l for l in AVAILABLE_TARGETS if 'oracle' in l and l in df.columns]
    
    if not oracle_labels:
        print("⚠️ No oracle labels found")
        return True, {}
    
    quality_metrics = {}
    
    for oracle_label in oracle_labels:
        print(f"\n📊 Analyzing {oracle_label}...")
        
        # Check class distribution
        label_dist = df[oracle_label].value_counts().sort_index()
        total_samples = len(df[oracle_label].dropna())
        
        # Check if all classes present
        missing_classes = []
        for class_id in range(8):
            if class_id not in label_dist.index:
                missing_classes.append(class_id)
        
        if missing_classes:
            print(f"   ⚠️ Missing classes: {missing_classes}")
        
        # Check class balance
        min_class_pct = (label_dist.min() / total_samples) * 100
        max_class_pct = (label_dist.max() / total_samples) * 100
        imbalance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')
        
        print(f"   Class distribution:")
        for class_id, count in label_dist.items():
            pct = (count / total_samples) * 100
            print(f"      Class {class_id}: {count:,} ({pct:.1f}%)")
        
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")
        
        if imbalance_ratio > 100:
            print(f"   🚨 SEVERE IMBALANCE: {imbalance_ratio:.1f}x")
        elif imbalance_ratio > 50:
            print(f"   ⚠️ HIGH IMBALANCE: {imbalance_ratio:.1f}x")
        else:
            print(f"   ✅ Reasonable balance: {imbalance_ratio:.1f}x")
        
        quality_metrics[oracle_label] = {
            'missing_classes': missing_classes,
            'imbalance_ratio': imbalance_ratio,
            'min_pct': min_class_pct,
            'max_pct': max_class_pct
        }
    
    return True, quality_metrics

def check_class_balance_all_targets(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    FIXED: Issue #5, #27 - Check class balance for all targets
    """
    print("\n" + "="*80)
    print("8. CLASS BALANCE VALIDATION (Issues #5, #27)")
    print("="*80)
    
    balance_metrics = {}
    all_passed = True
    
    for target in AVAILABLE_TARGETS:
        if target not in df.columns:
            continue
        
        print(f"\n📊 {target} class distribution:")
        class_dist = df[target].value_counts().sort_index()
        total_samples = len(df[target].dropna())
        
        missing_classes = []
        low_sample_classes = []
        
        for class_id in range(8):
            count = class_dist.get(class_id, 0)
            pct = (count / total_samples) * 100 if total_samples > 0 else 0
            
            if count == 0:
                missing_classes.append(class_id)
                print(f"   ❌ Class {class_id}: MISSING")
            elif count < 10:
                low_sample_classes.append(class_id)
                print(f"   ⚠️ Class {class_id}: {count} samples ({pct:.3f}%) - TOO FEW")
            else:
                status = "✅" if pct > 1.0 else "⚠️"
                print(f"   {status} Class {class_id}: {count:,} samples ({pct:.1f}%)")
        
        if missing_classes:
            print(f"\n   🚨 CRITICAL: {len(missing_classes)} missing classes: {missing_classes}")
            all_passed = False
        
        if low_sample_classes:
            print(f"   ⚠️ WARNING: {len(low_sample_classes)} classes with <10 samples: {low_sample_classes}")
        
        balance_metrics[target] = {
            'missing_classes': missing_classes,
            'low_sample_classes': low_sample_classes,
            'total_samples': total_samples
        }
    
    return all_passed, balance_metrics

def check_scenario_file_presence(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    FIXED: Issue #4, #12 - Verify scenario_file exists for proper splitting
    """
    print("\n" + "="*80)
    print("9. SCENARIO FILE VALIDATION (Issues #4, #12)")
    print("="*80)
    
    if 'scenario_file' not in df.columns:
        print("🚨 CRITICAL: 'scenario_file' column is MISSING!")
        print("   This means train/test split will be RANDOM (temporal leakage!)")
        return False, {}
    
    scenario_dist = df['scenario_file'].value_counts()
    num_scenarios = len(scenario_dist)
    
    print(f"✅ 'scenario_file' column present")
    print(f"📊 Number of scenarios: {num_scenarios}")
    print(f"📊 Samples per scenario (top 10):")
    
    for scenario, count in scenario_dist.head(10).items():
        print(f"   {scenario}: {count:,} samples")
    
    # Check if scenarios have reasonable size
    min_scenario_size = scenario_dist.min()
    max_scenario_size = scenario_dist.max()
    
    if min_scenario_size < 10:
        print(f"⚠️ WARNING: Smallest scenario has only {min_scenario_size} samples")
    
    metrics = {
        'num_scenarios': num_scenarios,
        'min_size': min_scenario_size,
        'max_size': max_scenario_size,
        'mean_size': scenario_dist.mean()
    }
    
    return True, metrics

# ================== MAIN VALIDATION ==================
def main():
    print("="*80)
    print("DATA LEAKAGE VALIDATION - ENHANCED VERSION")
    print("Author: ahmedjk34")
    print("Date: 2025-10-01")
    print("Validates: Issues #1, #2, #3, #4, #5, #12, #18, #27, #28, #31, #32, #33")
    print("="*80)
    
    if not Path(INPUT_CSV).exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        return False
    
    print(f"\n📂 Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"📊 Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Track validation results
    all_passed = True
    critical_issues = []
    warnings = []
    
    # 1. Temporal leakage check (Issue #1)
    passed, found = check_temporal_leakage_removal(df)
    if not passed:
        all_passed = False
        critical_issues.extend(found)
    
    # 2. Known leaky features check
    passed, found = check_known_leaky_features_removal(df)
    if not passed:
        all_passed = False
        critical_issues.extend(found)
    
    # 3. Useless features check (Issue #28)
    passed, found = check_useless_features_removal(df)
    if found:
        warnings.extend(found)
    
    # 4. Safe features presence check
    passed, missing = check_safe_features_present(df)
    if missing:
        warnings.extend(missing)
    
    # 5. Feature-target correlations (Issue #18)
    for target in AVAILABLE_TARGETS:
        if target in df.columns:
            passed, corrs = check_feature_target_correlations(df, target)
            if not passed:
                all_passed = False
                critical_issues.extend([f"{feat} (corr={corr:.3f})" for feat, corr in corrs])
    
    # 6. Context-SNR independence (Issue #3)
    passed, corr = check_context_snr_independence(df)
    if not passed:
        all_passed = False
        critical_issues.append(f"Context-SNR correlation: {corr:.3f}")
    
    # 7. Oracle label quality (Issues #31, #32)
    passed, metrics = check_oracle_label_quality(df)
    
    # 8. Class balance (Issues #5, #27)
    passed, balance = check_class_balance_all_targets(df)
    if not passed:
        all_passed = False
    
    # 9. Scenario file presence (Issues #4, #12)
    passed, scenario_metrics = check_scenario_file_presence(df)
    if not passed:
        all_passed = False
        critical_issues.append("scenario_file column missing")
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Critical Issues: {len(critical_issues)}")
    if critical_issues:
        for issue in critical_issues:
            print(f"   🚨 {issue}")
    
    print(f"\n⚠️ Warnings: {len(warnings)}")
    if warnings:
        for warning in warnings[:10]:  # Show first 10
            print(f"   ⚠️ {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more")
    
    print("\n" + "="*80)
    if all_passed and len(critical_issues) == 0:
        print("✅ VALIDATION PASSED: Dataset is clean and ready for training!")
        print("🚀 Safe to proceed with training")
        return True
    else:
        print("❌ VALIDATION FAILED: Critical issues found!")
        print("🛑 Fix issues before training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)