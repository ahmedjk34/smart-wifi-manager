"""
Intermediate ML Data Cleaning and Statistical Analysis Pipeline
Processes smart-v3-logged-ALL.csv with advanced cleaning, filtering, and comprehensive statistics.

CRITICAL: This file does NOT balance classes - that happens in File 1 (CSV combiner)
          Balancing here would destroy real WiFi traffic characteristics!

Features:
- Advanced duplicate detection and removal
- Extensive outlier detection and filtering
- Data type enforcement and validation
- Comprehensive statistical analysis and visualization
- NO CLASS BALANCING (maintains real WiFi distribution)
- Detailed logging and reporting
- Export cleaned data and statistics
- FIXED: Early removal of constant/useless features (Issue #28)
- FIXED: Reproducible random seed (Issue #14)
- FIXED: No class balancing in File 2 (Issue #TBD)

Author: ahmedjk34
Date: 2025-09-22
FIXED: 2025-10-01 (Issues #14, #28, no-balancing)
"""

import warnings
import sys
import logging

# CRITICAL: Suppress ALL warnings BEFORE any other imports
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Redirect stderr to devnull to suppress C-level warnings
import os
sys.stderr = open(os.devnull, 'w')

# NOW import pandas and numpy
import pandas as pd
import numpy as np

# Suppress pandas warnings
pd.options.mode.chained_assignment = None
pd.set_option('mode.copy_on_write', True)

# Suppress numpy warnings
np.seterr(all='ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from scipy import stats
import json

# ================== CONFIGURATION ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-BALANCED.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-ml-cleaned.csv")
STATS_DIR = os.path.join(BASE_DIR, "cleaning_stats")
LOG_FILE = os.path.join(BASE_DIR, "intermediate_cleaning.log")

# FIXED: Issue #14 - Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# WiFi-specific constraints
VALID_RATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
VALID_PHY_RATES = [
    1000000, 2000000, 5500000, 6000000, 9000000, 11000000, 12000000, 18000000,
    24000000, 36000000, 48000000, 54000000  # Complete 802.11g support
]
VALID_CHANNEL_WIDTHS = [20, 40, 80, 160]

# FIXED: Issue #28 - Define constant/useless features to remove early
CONSTANT_USELESS_FEATURES = [
    'T1', 'T2', 'T3',
    'decisionReason',
    'offeredLoad',
    'queueLen',
    'retryCount'
]

# Data validation ranges
VALIDATION_RANGES = {
    'lastSnr': (-10, 50),  # Expanded to allow slightly negative SNR
    'snrFast': (-10, 50),
    'snrSlow': (-10, 50),
    'snrTrendShort': (-20, 20),
    'snrStabilityIndex': (0, 50),
    'snrPredictionConfidence': (0, 1),
    'shortSuccRatio': (0, 1),
    'medSuccRatio': (0, 1),
    'consecSuccess': (0, 10000),
    'consecFailure': (0, 100),
    'recentThroughputTrend': (0, 10),
    'packetLossRate': (0, 1),
    'retrySuccessRatio': (0, 100),
    'recentRateChanges': (0, 100),
    'timeSinceLastRateChange': (0, 10000),
    'rateStabilityScore': (0, 1),
    'optimalRateDistance': (0, 10),
    'severity': (0, 1),
    'confidence': (0, 1),
    'channelWidth': (10, 160),
    'mobilityMetric': (0, 1000),
    'snrVariance': (0, 1000),
    'packetSuccess': (0, 1),
}

# CRITICAL: Class balancing DISABLED - happens in File 1 instead!
ENABLE_CLASS_BALANCING = False

# ================== SETUP ==================
def setup_logging():
    """Setup comprehensive logging"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("INTERMEDIATE ML DATA CLEANING - STATISTICAL CLEANING ONLY")
    logger.info("="*80)
    logger.info(f"Author: ahmedjk34")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random Seed: {RANDOM_SEED} (Issue #14 - Reproducibility)")
    logger.info(f"Class Balancing: {'ENABLED' if ENABLE_CLASS_BALANCING else 'DISABLED (correct!)'}")
    logger.info("="*80)
    logger.info("")
    logger.info("⚠️ IMPORTANT: This file does NOT balance classes!")
    logger.info("   Class balancing happens in File 1 (CSV combiner) to maintain real WiFi distribution.")
    logger.info("   File 2 only performs statistical cleaning (outliers, duplicates, validation).")
    logger.info("")
    return logger

def create_stats_directory():
    """Create directory for statistics outputs"""
    os.makedirs(STATS_DIR, exist_ok=True)
    return STATS_DIR

# ================== DATA LOADING ==================
def load_and_validate_data(filepath: str, logger) -> pd.DataFrame:
    """Load data with comprehensive validation"""
    logger.info(f"📂 Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"❌ Input file does not exist: {filepath}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data: {str(e)}")
        sys.exit(1)

# ================== EARLY FEATURE REMOVAL ==================
def remove_constant_useless_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    FIXED: Issue #28 - Remove constant/useless features early in pipeline
    """
    logger.info("🧹 EARLY REMOVAL: Checking for constant/useless features...")
    
    features_to_remove = []
    
    for feature in CONSTANT_USELESS_FEATURES:
        if feature in df.columns:
            unique_count = df[feature].nunique()
            
            if unique_count <= 1:
                unique_val = df[feature].iloc[0] if len(df) > 0 else None
                logger.info(f"  ❌ {feature}: CONSTANT (value={unique_val}) - REMOVING")
                features_to_remove.append(feature)
            else:
                if df[feature].dtype in ['int64', 'float64']:
                    zero_pct = (df[feature] == 0).sum() / len(df) * 100
                    if zero_pct > 99:
                        logger.info(f"  ⚠️ {feature}: {zero_pct:.1f}% zeros - REMOVING (useless)")
                        features_to_remove.append(feature)
                    else:
                        logger.info(f"  ℹ️ {feature}: Has {unique_count} unique values, keeping")
    
    if features_to_remove:
        df_clean = df.drop(columns=features_to_remove)
        logger.info(f"✅ Removed {len(features_to_remove)} constant/useless features: {features_to_remove}")
        logger.info(f"📊 Dataset shape after removal: {df_clean.shape}")
        return df_clean
    else:
        logger.info("✅ No constant/useless features found to remove")
        return df

# ================== COMPREHENSIVE STATISTICS ==================
def generate_comprehensive_statistics(df: pd.DataFrame, stage: str, logger) -> Dict[str, Any]:
    """Generate extensive statistical analysis"""
    logger.info(f"📈 Generating comprehensive statistics for {stage} stage...")
    
    stats_dict = {
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'basic_info': {},
        'column_info': {},
        'numerical_stats': {},
        'categorical_stats': {},
        'missing_data': {},
        'outliers': {},
        'wifi_specific': {}
    }
    
    # Basic information
    stats_dict['basic_info'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Column information
    stats_dict['column_info'] = {
        'column_names': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict(),
        'non_null_counts': df.count().to_dict()
    }
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        numerical_stats = df[numerical_cols].describe(include='all')
        stats_dict['numerical_stats'] = numerical_stats.to_dict()
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df) * 100).round(2)
    stats_dict['missing_data'] = {
        'missing_counts': missing_data.to_dict(),
        'missing_percentages': missing_percentages.to_dict(),
        'total_missing_values': int(missing_data.sum())
    }
    
    # WiFi-specific statistics
    wifi_stats = {}
    
    if 'rateIdx' in df.columns:
        rate_dist = df['rateIdx'].value_counts().sort_index()
        wifi_stats['rate_index_distribution'] = rate_dist.to_dict()
        
        # CRITICAL: Check if distribution looks natural
        if len(rate_dist) >= 8:
            rate_imbalance = rate_dist.max() / rate_dist.min()
            wifi_stats['rate_imbalance_ratio'] = float(rate_imbalance)
            
            if rate_imbalance < 2.0:
                logger.warning("⚠️ WARNING: Rate distribution looks artificially balanced!")
                logger.warning("   Real WiFi should have 5-20x more high rates than low rates.")
                logger.warning("   If you see this, check File 1 balancing logic!")
    
    if 'lastSnr' in df.columns:
        snr_data = df['lastSnr'].dropna()
        wifi_stats['snr_analysis'] = {
            'mean': float(snr_data.mean()),
            'median': float(snr_data.median()),
            'std': float(snr_data.std()),
            'min': float(snr_data.min()),
            'max': float(snr_data.max()),
        }
    
    if 'scenario_file' in df.columns:
        scenario_dist = df['scenario_file'].value_counts()
        wifi_stats['scenario_distribution'] = {
            'total_scenarios': int(df['scenario_file'].nunique()),
            'top_10_scenarios': scenario_dist.head(10).to_dict(),
        }
    
    stats_dict['wifi_specific'] = wifi_stats
    
    return stats_dict

def save_statistics(stats_dict: Dict[str, Any], stage: str, stats_dir: str, logger):
    """Save statistics to JSON format"""
    logger.info(f"💾 Saving statistics for {stage} stage...")
    
    def convert_numpy(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    stats_dict_converted = convert_numpy(stats_dict)
    
    json_file = os.path.join(stats_dir, f"statistics_{stage}.json")
    try:
        with open(json_file, 'w') as f:
            json.dump(stats_dict_converted, f, indent=2)
        logger.info(f"✅ Statistics saved to {json_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save statistics: {str(e)}")

def generate_visualizations(df: pd.DataFrame, stage: str, stats_dir: str, logger):
    """Generate key visualizations"""
    logger.info(f"📊 Generating visualizations for {stage} stage...")
    
    plt.style.use('default')
    
    # 1. Rate distribution (CRITICAL - shows if balancing happened)
    if 'rateIdx' in df.columns:
        plt.figure(figsize=(12, 6))
        rate_counts = df['rateIdx'].value_counts().sort_index()
        
        plt.subplot(1, 2, 1)
        plt.bar(rate_counts.index, rate_counts.values, color='steelblue')
        plt.title(f'Rate Index Distribution - {stage}')
        plt.xlabel('Rate Index')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add imbalance ratio annotation
        if len(rate_counts) >= 8:
            imbalance = rate_counts.max() / rate_counts.min()
            plt.text(0.5, 0.95, f'Imbalance Ratio: {imbalance:.1f}x', 
                    transform=plt.gca().transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.subplot(1, 2, 2)
        percentages = (rate_counts / rate_counts.sum() * 100).round(1)
        plt.bar(percentages.index, percentages.values, color='coral')
        plt.title(f'Rate Distribution (Percentage) - {stage}')
        plt.xlabel('Rate Index')
        plt.ylabel('Percentage (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f'rate_distribution_{stage}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. SNR distribution
    if 'lastSnr' in df.columns:
        plt.figure(figsize=(10, 6))
        df['lastSnr'].hist(bins=50, alpha=0.7, color='green')
        plt.title(f'SNR Distribution - {stage}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(stats_dir, f'snr_distribution_{stage}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✅ Visualizations saved to {stats_dir}")

# ================== DATA CLEANING FUNCTIONS ==================
def remove_duplicates(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Remove duplicate rows"""
    logger.info("🔍 Checking for duplicate rows...")
    
    initial_count = len(df)
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        logger.info(f"Found {duplicate_count} duplicate rows ({duplicate_count/initial_count*100:.2f}%)")
        df_clean = df.drop_duplicates()
        logger.info(f"✅ Removed {duplicate_count} duplicate rows")
        return df_clean
    else:
        logger.info("✅ No duplicate rows found")
        return df

def enforce_data_types(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Enforce proper data types"""
    logger.info("🔧 Enforcing data types...")
    
    df_clean = df.copy()
    type_conversions = 0
    
    expected_types = {
        'time': 'float64',
        'stationId': 'int64',
        'rateIdx': 'int64',
        'phyRate': 'int64',
        'lastSnr': 'float64',
        'snrFast': 'float64',
        'snrSlow': 'float64',
        'shortSuccRatio': 'float64',
        'medSuccRatio': 'float64',
        'channelWidth': 'int64',
    }
    
    for col, expected_type in expected_types.items():
        if col in df_clean.columns:
            try:
                if df_clean[col].dtype != expected_type:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce', downcast=None)
                    if expected_type.startswith('int'):
                        df_clean[col] = df_clean[col].round().astype('Int64')
                    type_conversions += 1
            except Exception:
                pass
    
    logger.info(f"✅ Completed {type_conversions} type conversions")
    return df_clean

def validate_wifi_constraints(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Validate WiFi-specific constraints"""
    logger.info("📡 Validating WiFi constraints...")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    if 'rateIdx' in df_clean.columns:
        valid_mask = df_clean['rateIdx'].notna() & df_clean['rateIdx'].isin(VALID_RATE_INDICES)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.info(f"  ⚠️ Removing {invalid_count} rows with invalid rate indices")
            df_clean = df_clean[valid_mask]
    
    if 'phyRate' in df_clean.columns:
        valid_mask = df_clean['phyRate'].notna() & df_clean['phyRate'].isin(VALID_PHY_RATES)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.info(f"  ⚠️ Removing {invalid_count} rows with invalid PHY rates")
            df_clean = df_clean[valid_mask]
    
    if 'channelWidth' in df_clean.columns:
        valid_mask = df_clean['channelWidth'].notna() & df_clean['channelWidth'].isin(VALID_CHANNEL_WIDTHS)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.info(f"  ⚠️ Removing {invalid_count} rows with invalid channel widths")
            df_clean = df_clean[valid_mask]
    
    removed_count = initial_count - len(df_clean)
    logger.info(f"✅ WiFi validation complete. Removed {removed_count} invalid rows")
    return df_clean

def filter_outliers(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Filter outliers based on validation ranges"""
    logger.info("🎯 Filtering outliers based on validation ranges...")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in df_clean.columns:
            outliers = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"  ⚠️ {col}: Removing {outlier_count} outliers "
                          f"(outside [{min_val}, {max_val}])")
                df_clean = df_clean[~outliers]
    
    total_removed = initial_count - len(df_clean)
    logger.info(f"✅ Outlier filtering complete. Removed {total_removed} rows "
               f"({total_removed/initial_count*100:.2f}%)")
    return df_clean

def handle_missing_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Handle missing data"""
    logger.info("🔧 Handling missing data...")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    critical_columns = ['rateIdx', 'phyRate', 'lastSnr', 'shortSuccRatio']
    
    missing_critical = df_clean[critical_columns].isnull().any(axis=1)
    critical_missing_count = missing_critical.sum()
    
    if critical_missing_count > 0:
        logger.info(f"  ❌ Removing {critical_missing_count} rows missing critical columns")
        df_clean = df_clean[~missing_critical]
    
    logger.info(f"✅ Missing data handling complete. Retained {len(df_clean)}/{initial_count} rows")
    return df_clean

# ================== MAIN PIPELINE ==================
def main():
    """Main pipeline execution - Statistical cleaning ONLY"""
    logger = setup_logging()
    stats_dir = create_stats_directory()
    
    try:
        # Load data
        df_raw = load_and_validate_data(INPUT_CSV, logger)
        
        # Remove constant/useless features early
        df_raw = remove_constant_useless_features(df_raw, logger)
        
        # Generate initial statistics
        initial_stats = generate_comprehensive_statistics(df_raw, "initial", logger)
        save_statistics(initial_stats, "initial", stats_dir, logger)
        generate_visualizations(df_raw, "initial", stats_dir, logger)
        
        # Start cleaning pipeline
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA CLEANING PIPELINE (STATISTICAL ONLY)")
        logger.info("="*60)
        
        df_clean = remove_duplicates(df_raw, logger)
        df_clean = enforce_data_types(df_clean, logger)
        df_clean = validate_wifi_constraints(df_clean, logger)
        df_clean = filter_outliers(df_clean, logger)
        df_clean = handle_missing_data(df_clean, logger)
        
        # CRITICAL: NO CLASS BALANCING HERE!
        logger.info("\n⚠️ SKIPPING class balancing (correct behavior!)")
        logger.info("   Class balancing happens in File 1 to maintain realistic WiFi distribution")
        
        # Generate final statistics
        final_stats = generate_comprehensive_statistics(df_clean, "final", logger)
        save_statistics(final_stats, "final", stats_dir, logger)
        generate_visualizations(df_clean, "final", stats_dir, logger)
        
        # Check final distribution
        if 'rateIdx' in df_clean.columns:
            rate_dist = df_clean['rateIdx'].value_counts().sort_index()
            logger.info("\n📊 Final rate distribution:")
            for rate, count in rate_dist.items():
                pct = (count / len(df_clean)) * 100
                logger.info(f"   Rate {rate}: {count:,} ({pct:.1f}%)")
            
            imbalance = rate_dist.max() / rate_dist.min()
            logger.info(f"\n📊 Rate imbalance ratio: {imbalance:.1f}x")
            
            if imbalance < 3.0:
                logger.warning("⚠️ WARNING: Distribution looks artificially balanced!")
                logger.warning("   Real WiFi should have 5-20x imbalance (high rates dominate)")
            else:
                logger.info("✅ Distribution looks realistic (high rates dominate)")
        
        # Save cleaned data
        logger.info(f"\n💾 Saving cleaned data to: {OUTPUT_CSV}")
        df_clean.to_csv(OUTPUT_CSV, index=False)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("CLEANING PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Initial rows: {len(df_raw):,}")
        logger.info(f"📊 Final rows: {len(df_clean):,}")
        logger.info(f"📊 Rows removed: {len(df_raw) - len(df_clean):,} "
                   f"({(len(df_raw) - len(df_clean))/len(df_raw)*100:.1f}%)")
        logger.info(f"📁 Output file: {OUTPUT_CSV}")
        logger.info(f"🔧 Class balancing: DISABLED (correct!)")
        
        print(f"\n✅ CLEANING COMPLETE!")
        print(f"📊 {len(df_raw):,} → {len(df_clean):,} rows")
        print(f"📁 Cleaned data: {OUTPUT_CSV}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# def balance_classes(
# df: pd.DataFrame, 
# target_col: str, 
# logger, 
# major_classes: List[int] = [0, 7],
# major_min: int = 200_000,
# minor_min: int = 100_000,
# enable_balancing: bool = False
# ) -> pd.DataFrame:
# """Balance classes with realistic ratios for major/minor classes."""

# if not enable_balancing:
#     logger.info(f"⚖️ Class balancing for {target_col} is DISABLED - skipping")
#     return df

# logger.info(f"⚖️ Advanced class balancing for {target_col} ...")

# if target_col not in df.columns:
#     logger.warning(f"⚠️ Target column {target_col} not found. Skipping balancing.")
#     return df

# class_counts = df[target_col].value_counts().sort_index()
# logger.info(f"📊 Current class distribution:")
# for class_val, count in class_counts.items():
#     logger.info(f"  {class_val}: {count} samples")

# dfs = []
# for c in class_counts.index:
#     if c in major_classes:
#         n = min(class_counts[c], major_min)
#         logger.info(f"  Major class {c}: keeping up to {n} samples")
#     else:
#         n = min(class_counts[c], minor_min)
#         logger.info(f"  Minor class {c}: keeping up to {n} samples")
#     if class_counts[c] > n:
#         # FIXED: Issue #14 - Use global random seed
#         sampled = df[df[target_col]==c].sample(n=n, random_state=RANDOM_SEED)
#         logger.info(f"    ✅ {c}: Downsampled from {class_counts[c]} to {n}")
#     else:
#         sampled = df[df[target_col]==c]
#         logger.info(f"    ✅ {c}: Kept all {class_counts[c]} samples")
#     dfs.append(sampled)

# balanced_df = pd.concat(dfs, ignore_index=True)
# logger.info(f"✅ Balancing complete. Final dataset: {len(balanced_df)} rows")
# logger.info(f"New class distribution:\n{balanced_df[target_col].value_counts().sort_index()}")
# return balanced_df