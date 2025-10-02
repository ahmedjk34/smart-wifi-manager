"""
Data Leakage Validation Script - FIXED FOR SNR-BASED ORACLES
Validates that no data leakage exists in the cleaned dataset before training.
NOW UNDERSTANDS: SNR-based oracles have HIGH SNR correlation (this is CORRECT!)

CRITICAL FIX (2025-10-02 14:18:00 UTC):
- SNR correlation with oracle labels is EXPECTED (0.85-0.95) - NOT leakage!
- Context-SNR correlation is EXPECTED (SNR defines context) - NOT leakage!
- Only checks for OUTCOME feature correlations (which are now removed)

Author: ahmedjk34
Date: 2025-10-02 (FIXED FOR NEW PIPELINE)
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

# 🔧 FIXED: Different correlation thresholds for different feature types
CORRELATION_THRESHOLDS = {
    'snr_features': 0.95,       # SNR can be highly correlated (oracle uses SNR!)
    'outcome_features': 0.5,    # Outcome features should be ABSENT
    'other_features': 0.7       # Other features shouldn't be too correlated
}

AVAILABLE_TARGETS = ["rateIdx", "oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# Temporal leakage features (should be ABSENT)
TEMPORAL_LEAKAGE_FEATURES = [
    "consecSuccess", "consecFailure", "retrySuccessRatio",
    "timeSinceLastRateChange", "rateStabilityScore", "recentRateChanges",
    "packetSuccess"
]

# Known leaky features (should be ABSENT)
KNOWN_LEAKY_FEATURES = [
    "phyRate", "optimalRateDistance", "recentThroughputTrend",
    "conservativeFactor", "aggressiveFactor", "recommendedSafeRate"
]

# Constant/useless features (should be ABSENT)
USELESS_FEATURES = [
    "T1", "T2", "T3", "decisionReason", "offeredLoad", "queueLen", "retryCount"
]

# 🔧 FIXED: Outcome features that should be REMOVED (File 3 fix)
OUTCOME_FEATURES_REMOVED = [
    "shortSuccRatio", "medSuccRatio", "packetLossRate", 
    "severity", "confidence"
]

# 🔧 FIXED: Safe features (9 features after File 3 fix)
SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort",
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "channelWidth", "mobilityMetric"
]

# SNR features (EXPECTED to correlate with oracle labels!)
SNR_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort",
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance"
]

CONTEXT_LABEL = "network_context"

# ================== VALIDATION FUNCTIONS ==================

def check_temporal_leakage_removal(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify ALL temporal leakage features are removed"""
    print("\n" + "="*80)
    print("1. TEMPORAL LEAKAGE VALIDATION")
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

def check_outcome_features_removed(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    🔧 NEW: Verify outcome features were removed by File 3
    """
    print("\n" + "="*80)
    print("3. OUTCOME FEATURES REMOVAL VALIDATION (File 3 Fix)")
    print("="*80)
    
    found_outcome = []
    for feature in OUTCOME_FEATURES_REMOVED:
        if feature in df.columns:
            found_outcome.append(feature)
            print(f"❌ CRITICAL: Outcome feature '{feature}' still present! (File 3 didn't work)")
        else:
            print(f"✅ Outcome feature '{feature}' properly removed")
    
    if found_outcome:
        print(f"\n🚨 VALIDATION FAILED: {len(found_outcome)} outcome features found!")
        print(f"   These should have been removed by File 3!")
        return False, found_outcome
    else:
        print(f"\n✅ PASS: All {len(OUTCOME_FEATURES_REMOVED)} outcome features removed")
        return True, []

def check_safe_features_present(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    🔧 FIXED: Verify ONLY the 9 safe features are present
    """
    print("\n" + "="*80)
    print("4. SAFE FEATURES PRESENCE VALIDATION (9 features expected)")
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
        return False, missing_safe
    else:
        print(f"\n✅ PASS: All {len(SAFE_FEATURES)} safe features present (NO outcome features)")
        return True, []

def check_feature_target_correlations(df: pd.DataFrame, target: str) -> Tuple[bool, List[Tuple[str, float]]]:
    """
    🔧 FIXED: Understands that SNR SHOULD correlate with oracle labels!
    
    Only flags:
    - Outcome features with high correlation (shouldn't exist!)
    - Non-SNR features with suspiciously high correlation
    
    Does NOT flag:
    - SNR features (they're SUPPOSED to correlate with oracle!)
    """
    print("\n" + "="*80)
    print(f"5. FEATURE-TARGET CORRELATION VALIDATION (SNR-AWARE)")
    print(f"   Target: {target}")
    print("="*80)
    
    if target not in df.columns:
        print(f"⚠️ Target '{target}' not found in dataset")
        return True, []
    
    critical_corrs = []
    expected_corrs = []
    
    # Check correlations for all remaining features
    for col in df.columns:
        if col in [target] + AVAILABLE_TARGETS + [CONTEXT_LABEL, 'scenario_file']:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            try:
                corr = df[col].corr(df[target])
                
                if pd.notnull(corr):
                    abs_corr = abs(corr)
                    
                    # Determine which threshold to use
                    if col in SNR_FEATURES:
                        # SNR features - EXPECTED to be high for oracle labels!
                        threshold = CORRELATION_THRESHOLDS['snr_features']
                        if abs_corr > threshold:
                            print(f"🚨 CRITICAL: {col} correlation {corr:.3f} > {threshold:.2f} (TOO high even for SNR!)")
                            critical_corrs.append((col, corr))
                        elif abs_corr > 0.7 and 'oracle' in target:
                            print(f"✅ EXPECTED: {col} = {corr:.3f} (oracle uses SNR - this is CORRECT!)")
                            expected_corrs.append((col, corr))
                        else:
                            print(f"✅ Safe: {col} = {corr:.3f}")
                    
                    elif col in OUTCOME_FEATURES_REMOVED:
                        # Outcome features - should NOT exist!
                        print(f"🚨 CRITICAL: {col} exists and correlates {corr:.3f} (should be removed!)")
                        critical_corrs.append((col, corr))
                    
                    else:
                        # Other features - moderate threshold
                        threshold = CORRELATION_THRESHOLDS['other_features']
                        if abs_corr > threshold:
                            print(f"🚨 CRITICAL: {col} has correlation {corr:.3f} with {target}")
                            critical_corrs.append((col, corr))
                        else:
                            print(f"✅ Safe: {col} = {corr:.3f}")
                            
            except Exception as e:
                print(f"⚠️ Could not compute correlation for {col}: {e}")
    
    if critical_corrs:
        print(f"\n🚨 VALIDATION FAILED: {len(critical_corrs)} critical correlations")
        for feat, corr in critical_corrs:
            print(f"   {feat}: {corr:.3f}")
        return False, critical_corrs
    else:
        if expected_corrs and 'oracle' in target:
            print(f"\n✅ PASS: {len(expected_corrs)} expected SNR correlations (oracle is SNR-based)")
            print(f"   This is CORRECT behavior - oracle uses SNR thresholds!")
        else:
            print(f"\n✅ PASS: No concerning correlations found")
        return True, []

def check_context_snr_relationship(df: pd.DataFrame) -> Tuple[bool, float]:
    """
    🔧 FIXED: Context-SNR correlation is EXPECTED (context is defined by SNR ranges)
    """
    print("\n" + "="*80)
    print("6. CONTEXT-SNR RELATIONSHIP VALIDATION (EXPECTED CORRELATION)")
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
    
    # Context SHOULD correlate with SNR (it's defined by SNR ranges!)
    if abs(snr_corr) > 0.5:
        print(f"✅ EXPECTED: Context correlates with SNR ({snr_corr:.3f})")
        print(f"   This is CORRECT - context is defined by SNR ranges (emergency, poor, good, excellent)")
        print(f"   Negative correlation is encoding artifact (doesn't indicate problem)")
        return True, snr_corr
    else:
        print(f"⚠️ WARNING: Context has LOW correlation with SNR ({snr_corr:.3f})")
        print(f"   Expected higher correlation since context is SNR-based")
        return True, snr_corr

def check_oracle_label_quality(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Check oracle label quality and distribution
    """
    print("\n" + "="*80)
    print("7. ORACLE LABEL QUALITY VALIDATION")
    print("="*80)
    
    oracle_labels = [l for l in AVAILABLE_TARGETS if 'oracle' in l and l in df.columns]
    
    if not oracle_labels:
        print("⚠️ No oracle labels found")
        return True, {}
    
    quality_metrics = {}
    has_issues = False
    
    for oracle_label in oracle_labels:
        print(f"\n📊 Analyzing {oracle_label}...")
        
        label_dist = df[oracle_label].value_counts().sort_index()
        total_samples = len(df[oracle_label].dropna())
        
        # Check missing classes
        missing_classes = [i for i in range(8) if i not in label_dist.index]
        
        if missing_classes:
            print(f"   🚨 CRITICAL: Missing classes: {missing_classes}")
            print(f"      Oracle will NEVER predict these rates!")
            has_issues = True
        
        # Check severe imbalance
        if len(label_dist) > 0:
            min_class_pct = (label_dist.min() / total_samples) * 100
            max_class_pct = (label_dist.max() / total_samples) * 100
            imbalance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')
            
            print(f"   Class distribution:")
            for class_id, count in label_dist.items():
                pct = (count / total_samples) * 100
                if count < 1000:
                    print(f"      ⚠️ Class {class_id}: {count:,} ({pct:.1f}%) - TOO FEW!")
                else:
                    print(f"      ✅ Class {class_id}: {count:,} ({pct:.1f}%)")
            
            print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")
            
            if imbalance_ratio > 100:
                print(f"   🚨 SEVERE IMBALANCE: {imbalance_ratio:.1f}x")
                print(f"      Model will struggle to learn rare classes!")
                has_issues = True
            elif imbalance_ratio > 30:
                print(f"   ⚠️ HIGH IMBALANCE: {imbalance_ratio:.1f}x (but manageable)")
            else:
                print(f"   ✅ Reasonable balance: {imbalance_ratio:.1f}x")
            
            quality_metrics[oracle_label] = {
                'missing_classes': missing_classes,
                'imbalance_ratio': imbalance_ratio,
                'min_pct': min_class_pct,
                'max_pct': max_class_pct
            }
    
    return not has_issues, quality_metrics

def check_scenario_file_presence(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Verify scenario_file exists for proper splitting"""
    print("\n" + "="*80)
    print("8. SCENARIO FILE VALIDATION")
    print("="*80)
    
    if 'scenario_file' not in df.columns:
        print("🚨 CRITICAL: 'scenario_file' column is MISSING!")
        print("   This means train/test split will be RANDOM (temporal leakage!)")
        return False, {}
    
    scenario_dist = df['scenario_file'].value_counts()
    num_scenarios = len(scenario_dist)
    
    print(f"✅ 'scenario_file' column present")
    print(f"📊 Number of scenarios: {num_scenarios}")
    
    if num_scenarios < 10:
        print(f"⚠️ WARNING: Only {num_scenarios} scenarios (need 10+ for reliable splitting)")
    
    metrics = {
        'num_scenarios': num_scenarios,
        'min_size': scenario_dist.min(),
        'max_size': scenario_dist.max(),
        'mean_size': scenario_dist.mean()
    }
    
    return True, metrics

# ================== MAIN VALIDATION ==================
def main():
    print("="*80)
    print("DATA LEAKAGE VALIDATION - FIXED FOR SNR-BASED ORACLES")
    print("Author: ahmedjk34")
    print("Date: 2025-10-02 14:18:00 UTC")
    print("Understands: SNR correlation is EXPECTED (oracle uses SNR)")
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
    
    # Run validations
    passed, found = check_temporal_leakage_removal(df)
    if not passed:
        all_passed = False
        critical_issues.extend(found)
    
    passed, found = check_known_leaky_features_removal(df)
    if not passed:
        all_passed = False
        critical_issues.extend(found)
    
    passed, found = check_outcome_features_removed(df)
    if not passed:
        all_passed = False
        critical_issues.extend(found)
    
    passed, missing = check_safe_features_present(df)
    if not passed:
        all_passed = False
        critical_issues.extend(missing)
    
    # Feature-target correlations (SNR-aware)
    for target in AVAILABLE_TARGETS:
        if target in df.columns:
            passed, corrs = check_feature_target_correlations(df, target)
            if not passed:
                all_passed = False
                critical_issues.extend([f"{feat} (corr={corr:.3f})" for feat, corr in corrs])
    
    # Context-SNR relationship (expected!)
    passed, corr = check_context_snr_relationship(df)
    # Don't fail on this - it's expected!
    
    # Oracle label quality
    passed, metrics = check_oracle_label_quality(df)
    if not passed:
        all_passed = False
        critical_issues.append("Oracle label quality issues (missing classes/severe imbalance)")
    
    # Scenario file
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
    
    print("\n" + "="*80)
    if all_passed and len(critical_issues) == 0:
        print("✅ VALIDATION PASSED: Dataset is clean and ready for training!")
        print("🚀 Safe to proceed with training")
        print("\nℹ️ NOTE: High SNR correlation with oracle labels is EXPECTED (oracle uses SNR)")
        return True
    else:
        print("❌ VALIDATION FAILED: Critical issues found!")
        print("🛑 Fix issues before training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)