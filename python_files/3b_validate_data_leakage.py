"""
Data Leakage Validation Script
Validates that no data leakage exists in the cleaned dataset before training.

Author: ahmedjk34
Date: 2025-09-28
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Configuration
INPUT_CSV = "smart-v3-ml-enriched.csv"
CORRELATION_THRESHOLD = 0.8  # Any correlation above this is suspicious
AVAILABLE_TARGETS = ["rateIdx", "oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# Guaranteed safe features (should have low correlation with targets)
SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "shortSuccRatio", "medSuccRatio", "consecSuccess", "consecFailure",
    "packetLossRate", "retrySuccessRatio", 
    "recentRateChanges", "timeSinceLastRateChange", "rateStabilityScore",
    "severity", "confidence", "packetSuccess",
    "channelWidth", "mobilityMetric"
]

# Known leaky features (should be removed)
KNOWN_LEAKY_FEATURES = [
    "phyRate", "optimalRateDistance", "recentThroughputTrend",
    "conservativeFactor", "aggressiveFactor", "recommendedSafeRate"
]

# Useless constant features (should be removed)
USELESS_FEATURES = [
    "T1", "T2", "T3", "decisionReason", "offeredLoad", "queueLen", "retryCount"
]

def validate_no_leakage(df):
    """Validate that no data leakage exists"""
    print("🔍 VALIDATING DATA LEAKAGE...")
    
    leakage_found = False
    
    # Check 1: Verify leaky features are removed
    print("\n📋 Checking for removed leaky features...")
    for leaky_feature in KNOWN_LEAKY_FEATURES:
        if leaky_feature in df.columns:
            print(f"❌ CRITICAL: Leaky feature '{leaky_feature}' still in dataset!")
            leakage_found = True
        else:
            print(f"✅ Leaky feature '{leaky_feature}' properly removed")
    
    # Check 2: Verify useless features are removed  
    print("\n📋 Checking for removed useless features...")
    for useless_feature in USELESS_FEATURES:
        if useless_feature in df.columns:
            print(f"⚠️ Useless feature '{useless_feature}' still in dataset")
            # Check if it's actually constant
            if df[useless_feature].nunique() <= 1:
                print(f"  📊 Confirmed constant: {df[useless_feature].iloc[0]}")
    
    # Check 3: Correlation analysis with targets
    print(f"\n📊 Checking correlations with targets (threshold: {CORRELATION_THRESHOLD})...")
    
    for target in AVAILABLE_TARGETS:
        if target not in df.columns:
            print(f"⚠️ Target '{target}' not found, skipping correlation check")
            continue
            
        print(f"\n🎯 Target: {target}")
        high_correlations = []
        
        for feature in SAFE_FEATURES:
            if feature not in df.columns:
                continue
                
            try:
                correlation = df[feature].corr(df[target])
                if pd.notnull(correlation) and abs(correlation) > CORRELATION_THRESHOLD:
                    high_correlations.append((feature, correlation))
                    print(f"  ❌ HIGH CORRELATION: {feature} = {correlation:.3f}")
                    leakage_found = True
                elif pd.notnull(correlation):
                    print(f"  ✅ Safe: {feature} = {correlation:.3f}")
            except Exception as e:
                print(f"  ⚠️ Could not compute correlation for {feature}: {e}")
        
        if not high_correlations:
            print(f"  ✅ No concerning correlations found for {target}")
    
    # Check 4: Class balance analysis
    print(f"\n📊 Analyzing class balance...")
    for target in AVAILABLE_TARGETS:
        if target in df.columns:
            class_dist = df[target].value_counts().sort_index()
            total_samples = len(df[target].dropna())
            
            print(f"\n🎯 {target} distribution:")
            min_class_pct = float('inf')
            for class_val, count in class_dist.items():
                pct = (count / total_samples) * 100
                min_class_pct = min(min_class_pct, pct)
                status = "❌" if pct < 0.5 else "⚠️" if pct < 2.0 else "✅"
                print(f"  {status} Class {class_val}: {count:,} samples ({pct:.1f}%)")
            
            if min_class_pct < 0.1:
                print(f"  ❌ SEVERE IMBALANCE: Smallest class has {min_class_pct:.3f}% of samples")
                leakage_found = True
    
    return not leakage_found

def main():
    print("="*60)
    print("DATA LEAKAGE VALIDATION")
    print("="*60)
    
    if not Path(INPUT_CSV).exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        return False
    
    print(f"📂 Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"📊 Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Run validation
    validation_passed = validate_no_leakage(df)
    
    print("\n" + "="*60)
    if validation_passed:
        print("✅ VALIDATION PASSED: No data leakage detected!")
        print("🚀 Safe to proceed with training")
        return True
    else:
        print("❌ VALIDATION FAILED: Data leakage issues found!")
        print("🛑 Fix issues before training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)