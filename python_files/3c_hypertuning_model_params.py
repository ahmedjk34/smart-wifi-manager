"""
Step 3c: Hyperparameter Tuning for WiFi Rate Adaptation - FIXED
Systematic optimization of RandomForest hyperparameters using GridSearchCV

FIXES:
- Issue #20: Model hyperparameters not tuned (arbitrary values)
- Issue #8: Cross-validation uses scenario-aware splits when available
- Issue #47: Grid search uses weighted accuracy
- Issue #48: Explicit handling of class imbalance during CV
- BUGFIX: Handles missing scenario_file gracefully

Features:
- Fallback to StratifiedKFold if scenario_file missing
- Custom scoring metric (penalizes large rate prediction errors)
- Class weight optimization
- Comprehensive hyperparameter grid search
- Results saved for all oracle strategies

Author: ahmedjk34
Date: 2025-10-01
Pipeline Stage: Step 3c - Hyperparameter Optimization (FIXED)
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
TARGET_LABELS = ["oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# Safe features (no temporal leakage - Issue #1)
SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "shortSuccRatio", "medSuccRatio",
    "packetLossRate",
    "channelWidth", "mobilityMetric",
    "severity", "confidence"
]

# FIXED: Comprehensive hyperparameter grid
HYPERPARAMETER_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced']
}

# Quick test grid (for debugging)
QUICK_TEST_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}

# Use quick test for development, full grid for production
USE_QUICK_TEST = False
PARAM_GRID = QUICK_TEST_GRID if USE_QUICK_TEST else HYPERPARAMETER_GRID

# Cross-validation configuration (Issue #8)
CV_FOLDS = 5

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
    logger.info(f"Grid Size: {np.prod([len(v) for v in PARAM_GRID.values()])} combinations")
    logger.info(f"CV Folds: {CV_FOLDS}")
    logger.info(f"Fixes: Issues #20, #8, #47, #48")
    logger.info("="*80)
    
    return logger

logger = setup_logging()

# ================== CUSTOM SCORING METRIC ==================
def rate_distance_weighted_accuracy(y_true, y_pred):
    """
    FIXED: Issue #47 - Custom scorer that penalizes by rate distance
    
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
    
    # FIXED: Extract scenario groups with proper type handling
    scenarios = None
    if 'scenario_file' in df.columns:
        # Convert to string and handle NaN/missing values
        scenario_series = df['scenario_file'].astype(str)
        
        # Replace 'nan' strings with None
        scenario_series = scenario_series.replace(['nan', 'None', ''], pd.NA)
        
        # Check if we have valid scenario data
        valid_scenarios = scenario_series.dropna()
        n_valid_scenarios = valid_scenarios.nunique()
        
        if n_valid_scenarios > 0 and len(valid_scenarios) > len(df) * 0.5:
            # More than 50% of rows have valid scenario info
            scenarios = scenario_series.fillna('unknown').values
            logger.info(f"✅ Scenario-aware CV enabled: {n_valid_scenarios} unique scenarios")
            logger.info(f"   {len(valid_scenarios):,}/{len(df):,} rows have scenario info ({len(valid_scenarios)/len(df)*100:.1f}%)")
        else:
            logger.warning(f"⚠️ scenario_file column exists but has insufficient valid data")
            logger.warning(f"   Only {n_valid_scenarios} unique scenarios, {len(valid_scenarios):,}/{len(df):,} valid rows")
            logger.warning(f"   Falling back to StratifiedKFold")
            scenarios = None
    else:
        logger.warning("⚠️ No scenario_file column - using standard StratifiedKFold")
        logger.warning("   This may allow temporal leakage in CV!")
    
    # Remove rows with missing target
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    if scenarios is not None:
        scenarios = scenarios[valid_mask]
    
    logger.info(f"Final dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution for {target_label}:")
    for class_id, count in zip(unique, counts):
        pct = count / len(y) * 100
        logger.info(f"  Class {class_id}: {count:,} samples ({pct:.1f}%)")
    
    # Check for classes with too few samples
    min_samples = counts.min()
    if min_samples < CV_FOLDS:
        logger.error(f"❌ Smallest class has only {min_samples} samples (need at least {CV_FOLDS} for {CV_FOLDS}-fold CV)")
        logger.error(f"   Consider merging rare classes or reducing CV_FOLDS")
        sys.exit(1)
    
    return df, X, y, scenarios

# ================== HYPERPARAMETER TUNING ==================
def tune_hyperparameters(X: np.ndarray, y: np.ndarray, scenarios: np.ndarray, 
                         target_label: str) -> Tuple[Dict, RandomForestClassifier]:
    """
    FIXED: Issue #20 - Systematic hyperparameter optimization
    FIXED: Handles missing scenario_file gracefully
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting hyperparameter tuning for {target_label}")
    logger.info(f"{'='*60}")
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Setup base model
    base_model = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    
    # FIXED: Setup CV splitter based on data availability
    if scenarios is not None:
        # Use GroupKFold for scenario-aware splitting
        n_scenarios = len(np.unique(scenarios))
        
        if n_scenarios >= CV_FOLDS:
            cv_splitter = GroupKFold(n_splits=CV_FOLDS)
            logger.info(f"✅ Using GroupKFold with {n_scenarios} scenarios (Issue #8 fix)")
        else:
            logger.warning(f"⚠️ Only {n_scenarios} scenarios (need {CV_FOLDS}+ for GroupKFold)")
            logger.warning(f"   Falling back to StratifiedKFold")
            cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scenarios = None  # Don't use groups
    else:
        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        logger.info(f"✅ Using StratifiedKFold (no scenario_file available)")
    
    # Setup GridSearchCV
    logger.info(f"Setting up GridSearchCV...")
    logger.info(f"  Parameter grid: {PARAM_GRID}")
    logger.info(f"  Total combinations: {np.prod([len(v) for v in PARAM_GRID.values()])}")
    logger.info(f"  CV folds: {CV_FOLDS}")
    logger.info(f"  Scoring: rate_distance_weighted_accuracy (Issue #47 fix)")
    
    # FIXED: GridSearchCV with proper cv parameter
    if scenarios is not None and isinstance(cv_splitter, GroupKFold):
        # GroupKFold requires groups parameter in fit()
        logger.info("  Using scenario groups for CV splitting")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=PARAM_GRID,
            cv=cv_splitter,
            scoring=rate_weighted_scorer,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
            error_score='raise'
        )
    else:
        # StratifiedKFold doesn't use groups
        logger.info("  Using stratified CV splitting (no groups)")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=PARAM_GRID,
            cv=cv_splitter,
            scoring=rate_weighted_scorer,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
            error_score='raise'
        )
    
    # Run grid search
    logger.info(f"\n🚀 Starting grid search (this may take a while)...")
    start_time = time.time()
    
    try:
        if scenarios is not None and isinstance(cv_splitter, GroupKFold):
            # Fit with groups for scenario-aware CV
            logger.info("Running grid search with scenario-aware splitting...")
            grid_search.fit(X_scaled, y, groups=scenarios)
        else:
            # Fit without groups for stratified CV
            logger.info("Running grid search with stratified splitting...")
            grid_search.fit(X_scaled, y)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Grid search completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
    except Exception as e:
        logger.error(f"❌ Grid search failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Provide helpful error message
        logger.error("\nDEBUG INFO:")
        logger.error(f"  X_scaled shape: {X_scaled.shape}")
        logger.error(f"  y shape: {y.shape}")
        logger.error(f"  Unique y values: {np.unique(y)}")
        logger.error(f"  scenarios is None: {scenarios is None}")
        if scenarios is not None:
            logger.error(f"  scenarios shape: {scenarios.shape}")
            logger.error(f"  Unique scenarios: {len(np.unique(scenarios))}")
        logger.error(f"  CV splitter type: {type(cv_splitter).__name__}")
        logger.error(f"  CV n_splits: {cv_splitter.get_n_splits()}")
        
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
    
    # Analyze top 5 configurations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    logger.info(f"\n📊 Top 5 configurations:")
    for idx, row in results_df.head(5).iterrows():
        logger.info(f"\nRank {int(row['rank_test_score'])}:")
        logger.info(f"  Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        logger.info(f"  Params: {row['params']}")
    
    # Prepare results dictionary
    results = {
        'target_label': target_label,
        'best_params': best_params,
        'best_score': float(best_score),
        'cv_folds': CV_FOLDS,
        'total_combinations_tested': len(results_df),
        'tuning_time_seconds': elapsed_time,
        'random_seed': RANDOM_SEED,
        'timestamp': datetime.now().isoformat(),
        'scenario_aware_cv': (scenarios is not None and isinstance(cv_splitter, GroupKFold)),
        'top_5_configs': []
    }
    
    for idx, row in results_df.head(5).iterrows():
        results['top_5_configs'].append({
            'rank': int(row['rank_test_score']),
            'score': float(row['mean_test_score']),
            'std': float(row['std_test_score']),
            'params': row['params']
        })
    
    return results, best_model, scaler

# ================== SAVE RESULTS ==================
def save_tuning_results(all_results: Dict[str, Dict]):
    """Save hyperparameter tuning results to JSON"""
    logger.info(f"\n💾 Saving tuning results...")
    
    output_file = OUTPUT_DIR / "hyperparameter_tuning_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {output_file}")
    
    # Also save a human-readable summary
    summary_file = OUTPUT_DIR / "hyperparameter_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER TUNING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: ahmedjk34\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        
        for target, results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"TARGET: {target}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Best Score: {results['best_score']:.4f} ({results['best_score']*100:.1f}%)\n")
            f.write(f"Tuning Time: {results['tuning_time_seconds']:.1f}s\n")
            f.write(f"Scenario-Aware CV: {'Yes' if results['scenario_aware_cv'] else 'No'}\n")
            f.write(f"\nBest Parameters:\n")
            for param, value in results['best_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nTop 5 Configurations:\n")
            for config in results['top_5_configs']:
                f.write(f"\n  Rank {config['rank']}: {config['score']:.4f} (±{config['std']:.4f})\n")
                f.write(f"    Params: {config['params']}\n")
    
    logger.info(f"✅ Summary saved to: {summary_file}")

# ================== MAIN EXECUTION ==================
def main():
    """Main hyperparameter tuning pipeline"""
    logger.info("🚀 Starting hyperparameter tuning pipeline...")
    
    all_results = {}
    all_models = {}
    all_scalers = {}
    
    pipeline_start = time.time()
    
    for target_idx, target_label in enumerate(TARGET_LABELS, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# TUNING MODEL {target_idx}/{len(TARGET_LABELS)}: {target_label}")
        logger.info(f"{'#'*80}")
        
        try:
            # Load data
            df, X, y, scenarios = load_and_prepare_data(target_label)
            
            # Tune hyperparameters
            results, best_model, scaler = tune_hyperparameters(X, y, scenarios, target_label)
            
            # Store results
            all_results[target_label] = results
            all_models[target_label] = best_model
            all_scalers[target_label] = scaler
            
            logger.info(f"✅ Completed tuning for {target_label}")
            
        except Exception as e:
            logger.error(f"❌ Tuning failed for {target_label}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    total_time = time.time() - pipeline_start
    
    # Save all results
    if all_results:
        save_tuning_results(all_results)
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"HYPERPARAMETER TUNING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Models tuned: {len(all_results)}/{len(TARGET_LABELS)}")
    
    if all_results:
        logger.info(f"\n📊 Best Scores Summary:")
        for target, results in all_results.items():
            logger.info(f"  {target}: {results['best_score']:.4f} ({results['best_score']*100:.1f}%)")
    
    logger.info(f"\n📁 Results saved to: {OUTPUT_DIR}")
    logger.info(f"\n✅ Hyperparameter tuning pipeline completed successfully!")
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if all_results:
        print(f"\nBest configurations:")
        for target, results in all_results.items():
            print(f"\n{target}: {results['best_score']*100:.1f}%")
            print(f"  n_estimators: {results['best_params']['n_estimators']}")
            print(f"  max_depth: {results['best_params']['max_depth']}")
            print(f"  min_samples_split: {results['best_params']['min_samples_split']}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)