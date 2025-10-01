"""
Intermediate ML Data Cleaning and Statistical Analysis Pipeline
Processes smart-v3-logged-ALL.csv with advanced cleaning, filtering, balancing, and comprehensive statistics.

Features:
- Advanced duplicate detection and removal
- Extensive outlier detection and filtering
- Data type enforcement and validation
- Comprehensive statistical analysis and visualization
- Class balancing options
- Detailed logging and reporting
- Export cleaned data and statistics
- FIXED: Early removal of constant/useless features (Issue #28)
- FIXED: Reproducible random seed (Issue #14)

Author: ahmedjk34
Date: 2025-09-22
FIXED: 2025-10-01 (Issues #14, #28)
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

import sys
import logging
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
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-ml-cleaned.csv")
STATS_DIR = os.path.join(BASE_DIR, "cleaning_stats")
LOG_FILE = os.path.join(BASE_DIR, "intermediate_cleaning.log")

# FIXED: Issue #14 - Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# WiFi-specific constraints
VALID_RATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
# FIXED: Issue #38 - Complete 802.11g PHY rates
VALID_PHY_RATES = [
    1000000, 2000000, 5500000, 6000000, 9000000, 11000000, 12000000, 18000000,
    24000000, 36000000, 48000000, 54000000  # Complete 802.11g support
]
VALID_CHANNEL_WIDTHS = [20, 40, 80, 160]

# FIXED: Issue #28 - Define constant/useless features to remove early
CONSTANT_USELESS_FEATURES = [
    'T1', 'T2', 'T3',           # Always constant thresholds
    'decisionReason',            # Always 0 in data
    'offeredLoad',               # Always 0 in data
    'queueLen',                  # Always 0 in data (mostly)
    'retryCount'                 # Always 0 in data
]

# Data validation ranges
VALIDATION_RANGES = {
    'lastSnr': (0, 50),
    'snrFast': (0, 50), 
    'snrSlow': (0, 50),
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
    'offeredLoad': (0, 1000000),
    'queueLen': (0, 1000),
    'retryCount': (0, 100),
    'channelWidth': (10, 160),
    'mobilityMetric': (0, 1000),
    'snrVariance': (0, 1000),
    'T1': (0, 100),
    'T2': (0, 100),
    'T3': (0, 100),
    'decisionReason': (0, 10),
    'packetSuccess': (0, 1),
}

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
    logger.info("INTERMEDIATE ML DATA CLEANING AND STATISTICAL ANALYSIS PIPELINE")
    logger.info(f"Author: ahmedjk34")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random Seed: {RANDOM_SEED} (Issue #14 - Reproducibility)")
    logger.info("="*80)
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
        
        # Log basic info about the dataset
        logger.info(f"📊 Dataset shape: {df.shape}")
        logger.info(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data: {str(e)}")
        sys.exit(1)

# ================== EARLY FEATURE REMOVAL ==================
def remove_constant_useless_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    FIXED: Issue #28 - Remove constant/useless features early in pipeline
    These features waste space and confuse feature importance analysis
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
                # Check if mostly zeros (>99%)
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
        'random_seed': RANDOM_SEED,  # FIXED: Issue #14 - Log seed
        'basic_info': {},
        'column_info': {},
        'numerical_stats': {},
        'categorical_stats': {},
        'missing_data': {},
        'outliers': {},
        'correlations': {},
        'wifi_specific': {}
    }
    
    # Basic information
    stats_dict['basic_info'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
        'completely_null_rows': df.isnull().all(axis=1).sum()
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
        
        # Additional numerical insights
        for col in numerical_cols:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats_dict['numerical_stats'][f'{col}_skewness'] = float(col_data.skew())
                    stats_dict['numerical_stats'][f'{col}_kurtosis'] = float(col_data.kurtosis())
                    stats_dict['numerical_stats'][f'{col}_unique_values'] = int(col_data.nunique())
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                stats_dict['categorical_stats'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head().to_dict()
                }
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df) * 100).round(2)
    stats_dict['missing_data'] = {
        'missing_counts': missing_data.to_dict(),
        'missing_percentages': missing_percentages.to_dict(),
        'columns_with_missing': missing_data[missing_data > 0].index.tolist(),
        'total_missing_values': int(missing_data.sum())
    }
    
    # Outlier detection (IQR method for numerical columns)
    outlier_info = {}
    for col in numerical_cols:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': round(len(outliers) / len(col_data) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
    stats_dict['outliers'] = outlier_info
    
    # WiFi-specific statistics
    wifi_stats = {}
    
    # Rate index distribution
    if 'rateIdx' in df.columns:
        rate_dist = df['rateIdx'].value_counts().sort_index()
        wifi_stats['rate_index_distribution'] = rate_dist.to_dict()
        wifi_stats['invalid_rate_indices'] = int((~df['rateIdx'].isin(VALID_RATE_INDICES)).sum())
    
    # PHY rate distribution
    if 'phyRate' in df.columns:
        phy_rate_dist = df['phyRate'].value_counts().sort_index()
        wifi_stats['phy_rate_distribution'] = phy_rate_dist.to_dict()
        wifi_stats['invalid_phy_rates'] = int((~df['phyRate'].isin(VALID_PHY_RATES)).sum())
    
    # SNR analysis
    if 'lastSnr' in df.columns:
        snr_data = df['lastSnr'].dropna()
        wifi_stats['snr_analysis'] = {
            'mean': float(snr_data.mean()),
            'median': float(snr_data.median()),
            'std': float(snr_data.std()),
            'min': float(snr_data.min()),
            'max': float(snr_data.max()),
            'below_10db': int((snr_data < 10).sum()),
            'above_30db': int((snr_data > 30).sum())
        }
    
    # Success ratio analysis
    if 'shortSuccRatio' in df.columns:
        success_data = df['shortSuccRatio'].dropna()
        wifi_stats['success_ratio_analysis'] = {
            'mean': float(success_data.mean()),
            'median': float(success_data.median()),
            'below_50_percent': int((success_data < 0.5).sum()),
            'above_90_percent': int((success_data > 0.9).sum()),
            'perfect_success': int((success_data == 1.0).sum())
        }
    
    # Scenario file analysis
    if 'scenario_file' in df.columns:
        scenario_dist = df['scenario_file'].value_counts()
        wifi_stats['scenario_distribution'] = {
            'total_scenarios': int(df['scenario_file'].nunique()),
            'top_10_scenarios': scenario_dist.head(10).to_dict(),
            'scenarios_with_single_entry': int((scenario_dist == 1).sum())
        }
    
    stats_dict['wifi_specific'] = wifi_stats
    
    return stats_dict

def save_statistics(stats_dict: Dict[str, Any], stage: str, stats_dir: str, logger):
    """Save statistics to multiple formats with robust numpy type conversion"""
    logger.info(f"💾 Saving statistics for {stage} stage...")
    
    # FIXED: Recursive numpy type converter
    def convert_numpy(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
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
    
    # Convert the entire stats dictionary
    stats_dict_converted = convert_numpy(stats_dict)
    
    # Save as JSON
    json_file = os.path.join(stats_dir, f"statistics_{stage}.json")
    try:
        with open(json_file, 'w') as f:
            json.dump(stats_dict_converted, f, indent=2)
        logger.info(f"✅ Statistics saved to {json_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save statistics: {str(e)}")
        
        # Fallback: Save as pickle if JSON fails
        pickle_file = os.path.join(stats_dir, f"statistics_{stage}.pkl")
        try:
            import pickle
            with open(pickle_file, 'wb') as f:
                pickle.dump(stats_dict, f)
            logger.warning(f"⚠️ Saved as pickle instead: {pickle_file}")
        except Exception as e2:
            logger.error(f"❌ Pickle fallback also failed: {str(e2)}")

def generate_visualizations(df: pd.DataFrame, stage: str, stats_dir: str, logger):
    """Generate comprehensive visualizations"""
    logger.info(f"📊 Generating visualizations for {stage} stage...")
    
    plt.style.use('default')
    
    # Set up the plotting parameters
    fig_size = (15, 10)
    
    # 1. Rate Index Distribution
    if 'rateIdx' in df.columns:
        plt.figure(figsize=(10, 6))
        rate_counts = df['rateIdx'].value_counts().sort_index()
        plt.bar(rate_counts.index, rate_counts.values)
        plt.title(f'Rate Index Distribution - {stage}')
        plt.xlabel('Rate Index')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(stats_dir, f'rate_distribution_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. SNR Distribution
    if 'lastSnr' in df.columns:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        df['lastSnr'].hist(bins=50, alpha=0.7)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        df.boxplot(column='lastSnr', ax=plt.gca())
        plt.title('SNR Box Plot')
        
        plt.subplot(2, 2, 3)
        if 'rateIdx' in df.columns:
            df.boxplot(column='lastSnr', by='rateIdx', ax=plt.gca())
            plt.title('SNR by Rate Index')
            plt.suptitle('')
        
        plt.subplot(2, 2, 4)
        df['lastSnr'].plot(kind='kde')
        plt.title('SNR Density')
        plt.xlabel('SNR (dB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f'snr_analysis_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Success Ratio Analysis
    if 'shortSuccRatio' in df.columns:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        df['shortSuccRatio'].hist(bins=50, alpha=0.7)
        plt.title('Success Ratio Distribution')
        plt.xlabel('Success Ratio')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        if 'rateIdx' in df.columns:
            df.boxplot(column='shortSuccRatio', by='rateIdx', ax=plt.gca())
            plt.title('Success Ratio by Rate Index')
            plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f'success_ratio_analysis_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Correlation Heatmap for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        plt.figure(figsize=(15, 12))
        correlation_matrix = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title(f'Feature Correlation Matrix - {stage}')
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f'correlation_matrix_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Missing Data Visualization
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(missing_data) > 0:
        plt.figure(figsize=(12, 6))
        missing_data.plot(kind='bar')
        plt.title(f'Missing Data by Column - {stage}')
        plt.xlabel('Columns')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f'missing_data_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✅ Visualizations saved to {stats_dir}")

# ================== DATA CLEANING FUNCTIONS ==================
def remove_duplicates(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Remove duplicate rows with detailed logging"""
    logger.info("🔍 Checking for duplicate rows...")
    
    initial_count = len(df)
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        logger.info(f"Found {duplicate_count} duplicate rows ({duplicate_count/initial_count*100:.2f}%)")
        df_clean = df.drop_duplicates()
        logger.info(f"✅ Removed {duplicate_count} duplicate rows")
        return df_clean
    else:
        logger.info("✅ No duplicate rows found")
        return df

def enforce_data_types(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Enforce proper data types with error handling"""
    logger.info("🔧 Enforcing data types...")
    
    df_clean = df.copy()
    type_conversions = 0
    
    # Define expected types
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
        'consecSuccess': 'int64',
        'consecFailure': 'int64',
        'packetSuccess': 'int64',
        'offeredLoad': 'float64',
        'queueLen': 'int64',
        'retryCount': 'int64',
        'channelWidth': 'int64',
        'mobilityMetric': 'float64',
        'snrVariance': 'float64'
    }
    
    for col, expected_type in expected_types.items():
        if col in df_clean.columns:
            try:
                if df_clean[col].dtype != expected_type:
                    # FIXED: Use errors='coerce' and downcast to suppress warnings
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce', downcast=None)
                    if expected_type.startswith('int'):
                        df_clean[col] = df_clean[col].round().astype('Int64')
                    type_conversions += 1
                    # Only log actual conversions, not every row
                    if type_conversions <= 10:  # Limit logging
                        logger.info(f"  ✅ Converted {col} to {expected_type}")
            except Exception as e:
                if type_conversions <= 10:
                    logger.warning(f"  ⚠️ Failed to convert {col}: {str(e)}")
    
    logger.info(f"✅ Completed {type_conversions} type conversions")
    return df_clean

def validate_wifi_constraints(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Validate WiFi-specific constraints with EXPANDED PHY rate support"""
    logger.info("📡 Validating WiFi constraints...")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Validate rate indices
    if 'rateIdx' in df_clean.columns:
        invalid_rates = ~df_clean['rateIdx'].isin(VALID_RATE_INDICES)
        invalid_count = invalid_rates.sum()
        if invalid_count > 0:
            logger.info(f"  ⚠️ Found {invalid_count} rows with invalid rate indices")
            df_clean = df_clean[~invalid_rates]
    
    # FIXED: Use expanded PHY rates (complete 802.11g)
    if 'phyRate' in df_clean.columns:
        invalid_phy = ~df_clean['phyRate'].isin(VALID_PHY_RATES)
        invalid_count = invalid_phy.sum()
        if invalid_count > 0:
            logger.info(f"  ⚠️ Found {invalid_count} rows with invalid PHY rates")
            unique_invalid = df_clean[invalid_phy]['phyRate'].unique()
            logger.info(f"  📊 Invalid PHY rates found: {sorted(unique_invalid)}")
            df_clean = df_clean[~invalid_phy]
    
    # Validate channel widths
    if 'channelWidth' in df_clean.columns:
        invalid_channels = ~df_clean['channelWidth'].isin(VALID_CHANNEL_WIDTHS)
        invalid_count = invalid_channels.sum()
        if invalid_count > 0:
            logger.info(f"  ⚠️ Found {invalid_count} rows with invalid channel widths")
            df_clean = df_clean[~invalid_channels]
    
    removed_count = initial_count - len(df_clean)
    logger.info(f"✅ WiFi validation complete. Removed {removed_count} invalid rows")
    return df_clean

def filter_outliers(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Filter outliers based on validation ranges"""
    logger.info("🎯 Filtering outliers based on validation ranges...")
    logger.info("📊 DATA ANALYSIS - Min/Max/Avg values for each column:")
    
    # First, analyze ALL numerical columns to see their actual ranges
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if col in df.columns and not df[col].isnull().all():
            col_data = df[col].dropna()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                avg_val = col_data.mean()
                logger.info(f"    {col}: MIN={min_val:.6f}, MAX={max_val:.6f}, AVG={avg_val:.6f}")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in df_clean.columns:
            before_count = len(df_clean)
            outliers = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            outlier_count = outliers.sum()
            
            # Log actual data range vs validation range
            actual_min = df_clean[col].min()
            actual_max = df_clean[col].max()
            logger.info(f"  🔍 {col}: Validation range [{min_val}, {max_val}], Actual range [{actual_min:.6f}, {actual_max:.6f}]")
            
            if outlier_count > 0:
                logger.info(f"    ⚠️ Found {outlier_count} outliers ({outlier_count/before_count*100:.2f}%)")
                df_clean = df_clean[~outliers]
                logger.info(f"    ✅ Removed {before_count - len(df_clean)} rows")
            else:
                logger.info(f"    ✅ No outliers found")
    
    total_removed = initial_count - len(df_clean)
    logger.info(f"✅ Outlier filtering complete. Removed {total_removed} total rows ({total_removed/initial_count*100:.2f}%)")
    return df_clean

def handle_missing_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Handle missing data with domain-specific logic"""
    logger.info("🔧 Handling missing data...")
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Define critical columns that cannot be missing
    critical_columns = ['rateIdx', 'phyRate', 'lastSnr', 'shortSuccRatio']
    
    # Remove rows missing critical columns
    missing_critical = df_clean[critical_columns].isnull().any(axis=1)
    critical_missing_count = missing_critical.sum()
    
    if critical_missing_count > 0:
        logger.info(f"  ❌ Removing {critical_missing_count} rows missing critical columns")
        df_clean = df_clean[~missing_critical]
    
    # Handle missing values in non-critical columns
    for col in df_clean.columns:
        if col not in critical_columns and df_clean[col].isnull().any():
            missing_count = df_clean[col].isnull().sum()
            missing_pct = missing_count / len(df_clean) * 100
            
            if missing_pct > 50:
                logger.info(f"  ⚠️ {col}: {missing_pct:.1f}% missing - keeping as is")
            elif df_clean[col].dtype in ['int64', 'float64', 'Int64']:
                # Fill numerical columns with median
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info(f"  ✅ {col}: Filled {missing_count} missing values with median ({median_val:.2f})")
            else:
                # Fill categorical columns with mode
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                    logger.info(f"  ✅ {col}: Filled {missing_count} missing values with mode ({mode_val[0]})")
    
    final_count = len(df_clean)
    logger.info(f"✅ Missing data handling complete. Retained {final_count}/{initial_count} rows")
    return df_clean

def balance_classes(
    df: pd.DataFrame, 
    target_col: str, 
    logger, 
    major_classes: List[int] = [0, 7],
    major_min: int = 200_000,
    minor_min: int = 100_000,
    enable_balancing: bool = False
) -> pd.DataFrame:
    """Balance classes with realistic ratios for major/minor classes."""
    
    if not enable_balancing:
        logger.info(f"⚖️ Class balancing for {target_col} is DISABLED - skipping")
        return df
    
    logger.info(f"⚖️ Advanced class balancing for {target_col} ...")

    if target_col not in df.columns:
        logger.warning(f"⚠️ Target column {target_col} not found. Skipping balancing.")
        return df

    class_counts = df[target_col].value_counts().sort_index()
    logger.info(f"📊 Current class distribution:")
    for class_val, count in class_counts.items():
        logger.info(f"  {class_val}: {count} samples")

    dfs = []
    for c in class_counts.index:
        if c in major_classes:
            n = min(class_counts[c], major_min)
            logger.info(f"  Major class {c}: keeping up to {n} samples")
        else:
            n = min(class_counts[c], minor_min)
            logger.info(f"  Minor class {c}: keeping up to {n} samples")
        if class_counts[c] > n:
            # FIXED: Issue #14 - Use global random seed
            sampled = df[df[target_col]==c].sample(n=n, random_state=RANDOM_SEED)
            logger.info(f"    ✅ {c}: Downsampled from {class_counts[c]} to {n}")
        else:
            sampled = df[df[target_col]==c]
            logger.info(f"    ✅ {c}: Kept all {class_counts[c]} samples")
        dfs.append(sampled)
    
    balanced_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ Balancing complete. Final dataset: {len(balanced_df)} rows")
    logger.info(f"New class distribution:\n{balanced_df[target_col].value_counts().sort_index()}")
    return balanced_df

# ================== MAIN PIPELINE ==================
def main():
    """Main pipeline execution"""
    logger = setup_logging()
    stats_dir = create_stats_directory()
    
    try:
        # Load data
        df_raw = load_and_validate_data(INPUT_CSV, logger)
        
        # FIXED: Issue #28 - Remove constant/useless features EARLY
        df_raw = remove_constant_useless_features(df_raw, logger)
        
        # Generate initial statistics
        initial_stats = generate_comprehensive_statistics(df_raw, "initial", logger)
        save_statistics(initial_stats, "initial", stats_dir, logger)
        generate_visualizations(df_raw, "initial", stats_dir, logger)
        
        # Start cleaning pipeline
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Remove duplicates
        df_clean = remove_duplicates(df_raw, logger)
        
        # Step 2: Enforce data types
        df_clean = enforce_data_types(df_clean, logger)
        
        # Step 3: Validate WiFi constraints
        df_clean = validate_wifi_constraints(df_clean, logger)
        
        # Step 4: Filter outliers
        df_clean = filter_outliers(df_clean, logger)
        
        # Step 5: Handle missing data
        df_clean = handle_missing_data(df_clean, logger)
        
        # Step 6: Optional class balancing (for rateIdx)
        if 'rateIdx' in df_clean.columns:
            df_clean = balance_classes(df_clean, 'rateIdx', logger, enable_balancing=ENABLE_CLASS_BALANCING)
                
        # Generate final statistics
        final_stats = generate_comprehensive_statistics(df_clean, "final", logger)
        save_statistics(final_stats, "final", stats_dir, logger)
        generate_visualizations(df_clean, "final", stats_dir, logger)
        
        # Save cleaned data
        logger.info(f"\n💾 Saving cleaned data to: {OUTPUT_CSV}")
        df_clean.to_csv(OUTPUT_CSV, index=False)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("CLEANING PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Initial rows: {len(df_raw):,}")
        logger.info(f"📊 Final rows: {len(df_clean):,}")
        logger.info(f"📊 Rows removed: {len(df_raw) - len(df_clean):,} ({(len(df_raw) - len(df_clean))/len(df_raw)*100:.1f}%)")
        logger.info(f"📊 Final columns: {len(df_clean.columns)}")
        logger.info(f"📁 Output file: {OUTPUT_CSV}")
        logger.info(f"📁 Statistics: {stats_dir}")
        logger.info(f"🔧 Random seed used: {RANDOM_SEED}")
        
        # Print summary to console
        print(f"\n✅ CLEANING COMPLETE!")
        print(f"📊 {len(df_raw):,} → {len(df_clean):,} rows ({(len(df_raw) - len(df_clean))/len(df_raw)*100:.1f}% removed)")
        print(f"📁 Cleaned data: {OUTPUT_CSV}")
        print(f"📁 Statistics: {stats_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)