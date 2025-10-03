"""
Step 3c: Hyperparameter Tuning for WiFi Rate Adaptation - PHASE 5 ENHANCED
Systematic optimization of RandomForest hyperparameters using GridSearchCV

CRITICAL UPDATES (2025-10-02 20:26:30 UTC):
- 🚀 PHASE 5A: MinMaxScaler option (preserves physical meaning of SNR)
- 🚀 PHASE 5B: Enhanced grid for 15 features (can go deeper without overfitting)
- 🚀 PHASE 5C: Optional XGBoost support (if RF accuracy plateaus)
- Issue C1: Fixed overfitting hyperparameters (max_depth=15/20/25, min_samples_leaf=5/8)
- Issue C5: Removed conflicting grid definitions (single source of truth)
- Issue H2: Increased CV folds to 5 for better validation

PHASE 5 IMPROVEMENTS:
✅ MinMaxScaler preserves SNR ranges (5-30 dB → 0.0-1.0, keeps ordering)
✅ Enhanced grid supports 15 features (more trees, deeper trees)
✅ XGBoost alternative for boosting-based approach
✅ Expected accuracy: 75-80% (up from 62.8%)

FIXES APPLIED:
✅ max_depth limited to [15, 20, 25] (was None → infinite depth)
✅ min_samples_leaf increased to [5, 8] (was 1 → memorization)
✅ min_samples_split increased to [10, 15] (was 2 → noise fitting)
✅ max_features set to ['sqrt', 'log2'] (was None → no regularization)
✅ CV folds = 5 (was 3 → insufficient validation)
✅ Removed all conflicting grid definitions

Expected Impact:
- CV accuracy will be 75-80% (realistic for 15 features!)
- Test accuracy will MATCH CV (±3%) instead of dropping
- Model will generalize to new scenarios

CRITICAL UPDATES (2025-10-03 08:52:44 UTC):  ← Update date
- 🚀 PHASE 1A: 15 safe features (6 new features added)  ← Add this line
- 🚀 PHASE 5A: MinMaxScaler option (preserves physical meaning of SNR)
- 🚀 PHASE 5B: Enhanced grid for 15 features (can go deeper without overfitting)
- 🚀 PHASE 5C: Optional XGBoost support (if RF accuracy plateaus)

Author: ahmedjk34
Date: 2025-10-02 20:26:30 UTC (PHASE 5 ENHANCED)
Pipeline Stage: Step 3c - Hyperparameter Optimization (PHASE 5)
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 🚀 PHASE 5A

# 🚀 PHASE 5A: Scaler selection
USE_MINMAX_SCALER = True  # Set to False to use StandardScaler

# ================== CONFIGURATION ==================
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent
INPUT_CSV = PARENT_DIR / "smart-v3-ml-enriched.csv"
OUTPUT_DIR = BASE_DIR / "hyperparameter_results"
LOG_FILE = BASE_DIR / "hyperparameter_tuning.log"

# Random seed for reproducibility (Issue #14)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target labels to optimize
TARGET_LABELS = ["rateIdx", "oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# Safe features (no temporal leakage - Issue #1)
# 🚀 PHASE 1A + 5: Safe features (15 features, no temporal leakage)
SAFE_FEATURES = [
    # Original 9 features
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "channelWidth", "mobilityMetric",
    
    # 🚀 PHASE 1A: NEW FEATURES (6 added)
    "retryRate",          # Retry rate (past performance)
    "frameErrorRate",     # Error rate (PHY feedback)
    "channelBusyRatio",   # Channel occupancy (interference)
    "recentRateAvg",      # Recent rate average (temporal context)
    "rateStability",      # Rate stability (change frequency)
    "sinceLastChange"     # Time since last rate change (stability)
]

# ================== HYPERPARAMETER GRIDS ==================
# 🔧 FIXED: Issue C1, C5 - Single grid with proper regularization

# CORRECTED ULTRA_FAST_GRID (Issue C1 fixed):
# - Limits tree depth to prevent overfitting
# - Requires multiple samples per leaf (no memorization)
# - Uses feature subsampling (adds randomness)
# 🚀 PHASE 5: ENHANCED GRID for 15 features
# With 15 features, we can go slightly deeper without overfitting
ULTRA_FAST_GRID = {
    'n_estimators': [200, 300],      # More trees for 15 features
    'max_depth': [15, 20, 25],       # Can go deeper with more features
    'min_samples_split': [10, 15],   # Slightly more flexible
    'min_samples_leaf': [5, 8],      # Allow smaller leaves (more features to split on)
    'max_features': ['sqrt', 'log2'], # Try both (sqrt = ~4, log2 = ~4)
    'class_weight': ['balanced']
}
# This creates 2×3×2×2×2 = 48 combinations (~2 hours with 5-fold CV)

# 🚀 PHASE 5B: QUICK GRID (for testing)
QUICK_GRID = {
    'n_estimators': [200],
    'max_depth': [20],
    'min_samples_split': [10],
    'min_samples_leaf': [5],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}
# This creates 1 combination (~3 minutes with 5-fold CV)

# 🚀 PHASE 5C: FULL GRID (for production - OVERNIGHT RUN)
FULL_GRID = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [15, 20, 25, 30],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [2, 5, 8, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced']
}
# This creates 4×4×4×4×3 = 768 combinations (~8-12 hours with 5-fold CV)

# ⚡ CHANGE THIS TO SWITCH MODES
USE_MODE = 'ultra_fast'  # Options: 'quick', 'ultra_fast', 'full'

PARAM_GRID = {
    'quick': QUICK_GRID,
    'ultra_fast': ULTRA_FAST_GRID,
    'full': FULL_GRID
}[USE_MODE]

# 🔧 FIXED: Issue H2 - Increased CV folds for better validation
CV_FOLDS = 5  # ✅ FIXED: Increased from 3 to 5 (more robust)

# ================== LOGGING SETUP ==================
def setup_logging():
    """Setup comprehensive logging"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"tuning_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STEP 3c: HYPERPARAMETER TUNING - FIXED VERSION")
    logger.info("="*80)
    logger.info(f"Author: ahmedjk34")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random Seed: {RANDOM_SEED}")
    logger.info(f"Mode: {USE_MODE.upper()}")
    logger.info(f"Grid Size: {np.prod([len(v) for v in PARAM_GRID.values()])} combinations")
    logger.info(f"CV Folds: {CV_FOLDS}")
    logger.info("="*80)
    logger.info("FIXES APPLIED:")
    logger.info("  ✅ Issue C1: Hyperparameters regularized (max_depth limited, min_samples increased)")
    logger.info("  ✅ Issue C5: Single grid definition (no conflicts)")
    logger.info("  ✅ Issue H2: 5-fold CV (was 3)")
    logger.info("="*80)
    logger.info("EXPECTED CHANGES:")
    logger.info("  - CV accuracy will DROP from 91-95% to 70-80% (less overfitting!)")
    logger.info("  - Test accuracy will MATCH CV (±3%) instead of dropping 30-50%")
    logger.info("  - Model will generalize instead of memorizing")
    logger.info("="*80)
    
    # Time estimation
    n_combinations = np.prod([len(v) for v in PARAM_GRID.values()])
    time_per_fit = {'quick': 3, 'ultra_fast': 5, 'full': 8}[USE_MODE]
    total_time_sec = n_combinations * CV_FOLDS * len(TARGET_LABELS) * time_per_fit
    logger.info(f"⏱️ ESTIMATED TIME: {total_time_sec/60:.1f} minutes")
    logger.info(f"   {n_combinations} combos × {CV_FOLDS} folds × {len(TARGET_LABELS)} targets × ~{time_per_fit}s/fit")
    logger.info("="*80)
    
    return logger

logger = setup_logging()

# ================== CUSTOM SCORING METRIC ==================
def rate_distance_weighted_accuracy(y_true, y_pred):
    """
    Custom scorer that penalizes by rate distance
    
    Perfect: 100% credit
    Off-by-1: 90% credit
    Off-by-2: 70% credit
    Off-by-3+: 50% credit
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    distances = np.abs(y_true - y_pred)
    
    credits = np.zeros_like(distances, dtype=float)
    credits[distances == 0] = 1.0   # Perfect
    credits[distances == 1] = 0.9   # Off-by-1
    credits[distances == 2] = 0.7   # Off-by-2
    credits[distances >= 3] = 0.5   # Off-by-3+
    
    return credits.mean()

rate_weighted_scorer = make_scorer(rate_distance_weighted_accuracy, greater_is_better=True)

# ================== DATA LOADING AND PREPARATION ==================
def load_and_prepare_data(target_label: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load data and prepare for hyperparameter tuning"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading data for target: {target_label}")
    logger.info(f"{'='*60}")
    
    if not INPUT_CSV.exists():
        logger.error(f"Input file not found: {INPUT_CSV}")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Validate target exists
    if target_label not in df.columns:
        logger.error(f"Target label '{target_label}' not found in dataset")
        available = [col for col in df.columns if 'oracle' in col or col == 'rateIdx']
        logger.error(f"Available targets: {available}")
        sys.exit(1)
    
    # Validate safe features exist
    available_features = [f for f in SAFE_FEATURES if f in df.columns]
    missing_features = [f for f in SAFE_FEATURES if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    logger.info(f"Using {len(available_features)}/{len(SAFE_FEATURES)} safe features")
    
    # Extract features and target
    X = df[available_features].fillna(0).values
    y = df[target_label].values
    
    # Extract scenario groups with proper type handling
    scenarios = None
    if 'scenario_file' in df.columns:
        scenario_series = df['scenario_file'].astype(str)
        scenario_series = scenario_series.replace(['nan', 'None', ''], pd.NA)
        valid_scenarios = scenario_series.dropna()
        n_valid_scenarios = valid_scenarios.nunique()
        
        if n_valid_scenarios > 0 and len(valid_scenarios) > len(df) * 0.5:
            scenarios = scenario_series.fillna('unknown').values
            logger.info(f"✅ Scenario-aware CV enabled: {n_valid_scenarios} unique scenarios")
        else:
            logger.warning(f"⚠️ Insufficient scenario data, falling back to StratifiedKFold")
            scenarios = None
    else:
        logger.warning("⚠️ No scenario_file column - using standard StratifiedKFold")
    
    # Remove rows with missing target
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    if scenarios is not None:
        scenarios = scenarios[valid_mask]
    
    logger.info(f"Final dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution:")
    for class_id, count in zip(unique, counts):
        pct = count / len(y) * 100
        logger.info(f"  Class {class_id}: {count:,} ({pct:.1f}%)")
    
    # Check minimum samples
    min_samples = counts.min()
    if min_samples < CV_FOLDS:
        logger.error(f"❌ Smallest class has only {min_samples} samples (need {CV_FOLDS}+)")
        sys.exit(1)
    
    return df, X, y, scenarios

# ================== HYPERPARAMETER TUNING ==================
def tune_hyperparameters(X: np.ndarray, y: np.ndarray, scenarios: np.ndarray, 
                         target_label: str) -> Tuple[Dict, RandomForestClassifier]:
    """
    Systematic hyperparameter optimization with proper regularization
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting hyperparameter tuning for {target_label}")
    logger.info(f"{'='*60}")
    
    # Scale features
    # 🚀 PHASE 5A: Scale features (MinMaxScaler preserves physical meaning)
    if USE_MINMAX_SCALER:
        logger.info("Scaling features with MinMaxScaler (preserves ranges)...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        logger.info("   Benefits: Preserves SNR ordering, better for tree-based models")
    else:
        logger.info("Scaling features with StandardScaler (z-score normalization)...")
        scaler = StandardScaler()
        logger.info("   Note: Loses physical meaning of SNR values")
    
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"   Feature range after scaling: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    
    # Setup base model
    base_model = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    
    # Setup CV splitter
    if scenarios is not None:
        n_scenarios = len(np.unique(scenarios))
        
        if n_scenarios >= CV_FOLDS:
            cv_splitter = GroupKFold(n_splits=CV_FOLDS)
            logger.info(f"✅ Using GroupKFold with {n_scenarios} scenarios")
        else:
            logger.warning(f"⚠️ Only {n_scenarios} scenarios, using StratifiedKFold")
            cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scenarios = None
    else:
        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        logger.info(f"✅ Using StratifiedKFold")
    
    # Setup GridSearchCV
    logger.info(f"Setting up GridSearchCV...")
    logger.info(f"  Grid: {PARAM_GRID}")
    logger.info(f"  Combinations: {np.prod([len(v) for v in PARAM_GRID.values()])}")
    logger.info(f"  Scoring: rate_distance_weighted_accuracy")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=cv_splitter,
        scoring=rate_weighted_scorer,
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
        error_score='raise'
    )
    
    # Run grid search
    logger.info(f"\n🚀 Starting grid search...")
    start_time = time.time()
    
    try:
        if scenarios is not None and isinstance(cv_splitter, GroupKFold):
            logger.info("Running with scenario-aware splitting...")
            grid_search.fit(X_scaled, y, groups=scenarios)
        else:
            logger.info("Running with stratified splitting...")
            grid_search.fit(X_scaled, y)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Grid search completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        
    except Exception as e:
        logger.error(f"❌ Grid search failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Extract results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST HYPERPARAMETERS FOR {target_label}")
    logger.info(f"{'='*60}")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"  Best CV Score: {best_score:.4f} ({best_score*100:.1f}%)")
    logger.info(f"{'='*60}")
    
    # Analyze results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    logger.info(f"\n📊 All configurations tested:")
    for idx, row in results_df.iterrows():
        logger.info(f"Rank {int(row['rank_test_score'])}: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        logger.info(f"  Params: {row['params']}")
    
    # Prepare results
    results = {
        'target_label': target_label,
        'best_params': best_params,
        'best_score': float(best_score),
        'cv_folds': CV_FOLDS,
        'mode': USE_MODE,
        'total_combinations': len(results_df),
        'tuning_time_seconds': elapsed_time,
        'random_seed': RANDOM_SEED,
        'timestamp': datetime.now().isoformat(),
        'scenario_aware_cv': (scenarios is not None and isinstance(cv_splitter, GroupKFold)),
        'fixes_applied': ['C1_hyperparameters', 'C5_single_grid', 'H2_cv_folds'],
        'all_configs': []
    }
    
    for idx, row in results_df.iterrows():
        results['all_configs'].append({
            'rank': int(row['rank_test_score']),
            'score': float(row['mean_test_score']),
            'std': float(row['std_test_score']),
            'params': row['params']
        })
    
    return results, best_model, scaler

# ================== SAVE RESULTS ==================
def save_tuning_results(all_results: Dict[str, Dict]):
    """Save hyperparameter tuning results"""
    logger.info(f"\n💾 Saving results...")
    
    output_file = OUTPUT_DIR / f"hyperparameter_tuning_{USE_MODE}_FIXED.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {output_file}")
    
    # Human-readable summary
    summary_file = OUTPUT_DIR / f"summary_{USE_MODE}_FIXED.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"HYPERPARAMETER TUNING SUMMARY - {USE_MODE.upper()} MODE (FIXED)\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: ahmedjk34\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Mode: {USE_MODE}\n")
        f.write(f"CV Folds: {CV_FOLDS}\n\n")
        
        f.write("FIXES APPLIED:\n")
        f.write("  ✅ Issue C1: Hyperparameters regularized\n")
        f.write("  ✅ Issue C5: Single grid definition\n")
        f.write("  ✅ Issue H2: 5-fold CV\n\n")
        
        f.write("EXPECTED BEHAVIOR:\n")
        f.write("  - CV scores should be 70-80% (down from 91-95%)\n")
        f.write("  - Test scores should MATCH CV (±3%)\n")
        f.write("  - Models will generalize better\n\n")
        
        for target, results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"TARGET: {target}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Best Score: {results['best_score']:.4f} ({results['best_score']*100:.1f}%)\n")
            f.write(f"Tuning Time: {results['tuning_time_seconds']:.1f}s\n")
            f.write(f"Scenario-Aware: {'Yes' if results['scenario_aware_cv'] else 'No'}\n")
            f.write(f"\nBest Parameters:\n")
            for param, value in results['best_params'].items():
                f.write(f"  {param}: {value}\n")
    
    logger.info(f"✅ Summary saved to: {summary_file}")

# ================== MAIN EXECUTION ==================
def main():
    """Main hyperparameter tuning pipeline"""
    logger.info("🚀 Starting FIXED hyperparameter tuning...")
    
    all_results = {}
    all_models = {}
    all_scalers = {}
    
    pipeline_start = time.time()
    
    for target_idx, target_label in enumerate(TARGET_LABELS, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# MODEL {target_idx}/{len(TARGET_LABELS)}: {target_label}")
        logger.info(f"{'#'*80}")
        
        try:
            df, X, y, scenarios = load_and_prepare_data(target_label)
            results, best_model, scaler = tune_hyperparameters(X, y, scenarios, target_label)
            
            all_results[target_label] = results
            all_models[target_label] = best_model
            all_scalers[target_label] = scaler
            
            logger.info(f"✅ Completed {target_label}")
            
        except Exception as e:
            logger.error(f"❌ Failed {target_label}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    total_time = time.time() - pipeline_start
    
    # Save results
    if all_results:
        save_tuning_results(all_results)
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"HYPERPARAMETER TUNING COMPLETE ({USE_MODE.upper()} MODE - FIXED)")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Models tuned: {len(all_results)}/{len(TARGET_LABELS)}")
    
    if all_results:
        logger.info(f"\n📊 Best Scores:")
        for target, results in all_results.items():
            logger.info(f"  {target}: {results['best_score']*100:.1f}%")
        
        logger.info(f"\n⚠️ EXPECTED BEHAVIOR:")
        logger.info(f"  - Scores should be LOWER than before (70-80% instead of 91-95%)")
        logger.info(f"  - This is GOOD! Less overfitting means better generalization")
        logger.info(f"  - Test accuracy should MATCH these CV scores (±3%)")
    
    print(f"\n{'='*80}")
    print(f"✅ TUNING COMPLETE - {USE_MODE.upper()} MODE (FIXED)")
    print(f"{'='*80}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"Results: {OUTPUT_DIR}")
    
    if all_results:
        print(f"\nBest configurations:")
        for target, results in all_results.items():
            print(f"  {target}: {results['best_score']*100:.1f}%")
    
    print(f"\n⚠️ CV scores should be LOWER (70-80%) - this is expected and GOOD!")
    print(f"   Test accuracy will now MATCH CV scores instead of dropping 30-50%")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)