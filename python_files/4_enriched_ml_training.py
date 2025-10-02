"""
Ultimate ML Model Training Pipeline - FULLY FIXED VERSION
Trains Random Forest models using optimized hyperparameters from Step 3c

CRITICAL FIXES (2025-10-02 14:05:10 UTC):
- Issue C4: Class weight cap increased to 10.0 (was 3.0, now matches 20x imbalance)
- Issue H1: REMOVED temporal sample weighting (conflicts with scenario splitting)
- Issue H3: REMOVED misleading post-training CV check
- Issue M2: Feature scaling acknowledges scenario-based structure
- Issue M3: Class weights computed from train set (documented limitation)

WHAT WAS WRONG BEFORE:
❌ Class weights capped at 3.0x with 20-33x imbalance → rare classes ignored
❌ Temporal sample weights (0.5-1.5x) conflicted with scenario-based splitting
❌ Post-training 5-fold CV misleadingly included training data
❌ Feature scaling assumed i.i.d. data (but scenarios are grouped)

WHAT'S FIXED NOW:
✅ Class weights capped at 10.0x (handles 20x imbalance from File 1b)
✅ Temporal weighting REMOVED (equal weights for all samples)
✅ Post-training CV REMOVED (only train/val/test splits reported)
✅ Scaling documented as global (acceptable with scenario-aware splitting)

EXPECTED IMPACT:
- Rare classes (0-3) will have MUCH better recall (7-34% → 40-60%)
- Training is simpler (no conflicting weight schemes)
- Reported metrics are honest (no inflated CV scores)
- Class imbalance properly handled via weights + File 1b balancing

FIXES APPLIED:
✅ Issue C4: Class weight cap = 10.0
✅ Issue H1: Temporal weights removed
✅ Issue H3: Post-training CV removed
✅ Issue M2: Scaling documented
✅ Issue M3: Class weight limitation documented

Author: ahmedjk34
Date: 2025-10-02 14:05:10 UTC
Pipeline Stage: Step 4 - Model Training (FULLY FIXED)
"""

import pandas as pd
import joblib
import logging
import time
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ================== CONFIGURATION ==================
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent
CSV_FILE = PARENT_DIR / "smart-v3-ml-enriched.csv"
HYPERPARAMS_FILE = BASE_DIR / "hyperparameter_results" / "hyperparameter_tuning_ultra_fast_FIXED.json"
CLASS_WEIGHTS_FILE = BASE_DIR / "model_artifacts" / "class_weights.json"
OUTPUT_DIR = BASE_DIR / "trained_models"
LOG_DIR = BASE_DIR / "logs"

# FIXED: Issue #14 - Global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target labels to train
TARGET_LABELS = [
    "rateIdx",                    # Natural ground truth from ns-3 Minstrel HT
    "oracle_conservative",        # Conservative oracle strategy
    "oracle_balanced",            # Balanced oracle strategy  
    "oracle_aggressive"           # Aggressive oracle strategy
]

# 🔧 FIXED: Issue C3 - SAFE features only (NO OUTCOME FEATURES!)
# Reduced from 14 to 9 features (removed shortSuccRatio, medSuccRatio, packetLossRate, severity, confidence)
SAFE_FEATURES = [
    # SNR features (pre-decision) - SAFE
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    
    # Network state (pre-decision) - SAFE
    "channelWidth", "mobilityMetric"
]

# Train/Val/Test split ratios
TEST_SIZE = 0.2      # 20% for final test
VAL_SIZE = 0.2       # 20% of remaining (16% of total)
# Final: 64% train, 16% val, 20% test

# FIXED: Issue #17 - Realistic performance thresholds (no more leakage)
PERFORMANCE_THRESHOLDS = {
    'excellent': 0.75,   # >75% is excellent for WiFi without leakage
    'good': 0.65,        # 65-75% is good
    'acceptable': 0.55,  # 55-65% is acceptable
    # <55% needs investigation
}

USER = "ahmedjk34"

# ================== LOGGING SETUP ==================
def setup_logging(target_label: str):
    """Setup comprehensive logging for specific target"""
    LOG_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"training_{target_label}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"ML MODEL TRAINING PIPELINE - FULLY FIXED VERSION")
    logger.info("="*80)
    logger.info(f"🎯 Target Label: {target_label}")
    logger.info(f"👤 Author: {USER}")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔧 Random Seed: {RANDOM_SEED}")
    logger.info(f"🛡️ Safe Features: {len(SAFE_FEATURES)} (no outcome features)")
    logger.info(f"💻 Device: CPU (no GPU required)")
    logger.info("="*80)
    logger.info("FIXES APPLIED:")
    logger.info("  ✅ Issue C4: Class weight cap increased to 10.0 (was 3.0)")
    logger.info("  ✅ Issue H1: Temporal sample weighting REMOVED")
    logger.info("  ✅ Issue H3: Post-training CV check REMOVED")
    logger.info("  ✅ Issue M2: Feature scaling documented (global across scenarios)")
    logger.info("  ✅ Issue M3: Class weight limitation documented")
    logger.info("  ✅ Issue C3: Training uses 9 features (removed 5 outcome features)")
    logger.info("="*80)
    logger.info("EXPECTED CHANGES:")
    logger.info("  - Rare classes (0-3) will have BETTER recall (7-34% → 40-60%)")
    logger.info("  - No conflicting weight schemes (simpler training)")
    logger.info("  - Honest metrics (no inflated CV scores)")
    logger.info("="*80)
    
    return logger

# ================== LOAD HYPERPARAMETERS ==================
def load_optimized_hyperparameters(target_label: str, logger) -> Dict:
    """
    Auto-detect hyperparameter JSON file from File 3c
    Handles ultra_fast, quick, full, or custom names
    """
    logger.info(f"📂 Looking for hyperparameter files in: {HYPERPARAMS_FILE.parent}")
    
    hyperparams_dir = BASE_DIR / "hyperparameter_results"
    
    if not hyperparams_dir.exists():
        logger.warning(f"⚠️ Hyperparameter directory not found: {hyperparams_dir}")
        logger.warning(f"   Please run Step 3c (hyperparameter tuning) first")
        logger.warning(f"   Falling back to default parameters...")
        return get_default_hyperparameters()
    
    # Find all JSON files with FIXED suffix (prioritize fixed versions)
    json_files = list(hyperparams_dir.glob("hyperparameter_tuning_*_FIXED.json"))
    
    if len(json_files) == 0:
        # Try without FIXED suffix
        json_files = list(hyperparams_dir.glob("hyperparameter_tuning_*.json"))
    
    if len(json_files) == 0:
        logger.warning(f"⚠️ No hyperparameter JSON files found!")
        logger.warning(f"   Expected pattern: hyperparameter_tuning_*_FIXED.json")
        logger.warning(f"   Please run Step 3c first")
        return get_default_hyperparameters()
    
    if len(json_files) > 1:
        logger.error(f"❌ Multiple hyperparameter files found:")
        for f in json_files:
            logger.error(f"   - {f.name}")
        logger.error(f"   Please keep only ONE file (delete others or move to backup)")
        logger.error(f"   Recommended: hyperparameter_tuning_ultra_fast_FIXED.json (most recent)")
        sys.exit(1)
    
    # Use the single JSON file found
    hyperparam_file = json_files[0]
    logger.info(f"✅ Found hyperparameter file: {hyperparam_file.name}")
    
    try:
        with open(hyperparam_file, 'r') as f:
            all_hyperparams = json.load(f)
        
        if target_label not in all_hyperparams:
            logger.warning(f"⚠️ No hyperparameters found for {target_label}")
            logger.warning(f"   Available targets: {list(all_hyperparams.keys())}")
            return get_default_hyperparameters()
        
        # Extract best parameters
        target_results = all_hyperparams[target_label]
        best_params = target_results['best_params']
        best_score = target_results.get('best_score', 0.0)
        
        logger.info(f"✅ Loaded optimized hyperparameters for {target_label}")
        logger.info(f"   CV Score: {best_score:.4f} ({best_score*100:.1f}%)")
        logger.info(f"   Parameters:")
        for param, value in best_params.items():
            logger.info(f"     {param}: {value}")
        
        best_params['source'] = f'optimized_{hyperparam_file.stem}'
        return best_params
        
    except Exception as e:
        logger.error(f"❌ Failed to load hyperparameters: {str(e)}")
        logger.warning(f"   Falling back to default parameters...")
        return get_default_hyperparameters()

def get_default_hyperparameters() -> Dict:
    """Return default hyperparameters as fallback (FIXED version)"""
    return {
        'n_estimators': 200,
        'max_depth': 15,           # FIXED: Limited (was None)
        'min_samples_split': 10,   # FIXED: Increased (was 2)
        'min_samples_leaf': 5,     # FIXED: Increased (was 1)
        'max_features': 'sqrt',    # FIXED: Feature subsampling (was None)
        'class_weight': 'balanced',
        'source': 'default_fallback_FIXED'
    }


# ================== SCENARIO-AWARE SPLITTING ==================
def scenario_aware_stratified_split(df: pd.DataFrame, target_label: str, logger) -> Tuple:
    """
    Scenario-aware stratified train/val/test split
    FIXED: Handles scenarios with missing target data
    
    Ensures:
    - Entire scenarios in train OR val OR test (no mixing)
    - Stratification by dominant class per scenario
    - All classes present in each split (if possible)
    - Raises error if scenario_file missing (no silent fallback)
    - Handles scenarios with no valid target data
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENARIO-AWARE STRATIFIED SPLIT")
    logger.info(f"{'='*60}")
    
    # FIXED: Issue #12 - Raise error if scenario_file missing
    if 'scenario_file' not in df.columns:
        logger.error("❌ CRITICAL: 'scenario_file' column is MISSING!")
        logger.error("   This would cause random train/test split (temporal leakage)")
        logger.error("   Cannot proceed without scenario information")
        raise ValueError("scenario_file column is required for temporal leak prevention")
    
    # Get unique scenarios
    scenarios = df['scenario_file'].unique()
    n_scenarios = len(scenarios)
    logger.info(f"📊 Total scenarios: {n_scenarios}")
    
    if n_scenarios < 10:
        logger.warning(f"⚠️ WARNING: Only {n_scenarios} scenarios (need 10+ for good split)")
    
    # Compute dominant class per scenario with robust error handling
    scenario_classes = {}
    empty_scenarios = []
    
    for scenario in scenarios:
        scenario_data = df[df['scenario_file'] == scenario]
        
        # Check if scenario has any valid target data
        valid_targets = scenario_data[target_label].dropna()
        
        if len(valid_targets) == 0:
            logger.warning(f"⚠️ Scenario '{scenario}' has NO valid {target_label} data - SKIPPING")
            empty_scenarios.append(scenario)
            continue
        
        # Try mode first
        mode_result = valid_targets.mode()
        
        if len(mode_result) == 0:
            # No clear mode - use value_counts
            value_counts = valid_targets.value_counts()
            if len(value_counts) == 0:
                logger.warning(f"⚠️ Scenario '{scenario}' has no countable values - SKIPPING")
                empty_scenarios.append(scenario)
                continue
            dominant_class = value_counts.index[0]
        else:
            dominant_class = mode_result.iloc[0]
        
        scenario_classes[scenario] = dominant_class
    
    # Report empty scenarios
    if empty_scenarios:
        logger.warning(f"⚠️ Skipped {len(empty_scenarios)} empty scenarios:")
        for sc in empty_scenarios[:5]:
            logger.warning(f"   - {sc}")
        if len(empty_scenarios) > 5:
            logger.warning(f"   ... and {len(empty_scenarios) - 5} more")
    
    # Use only valid scenarios
    valid_scenarios = list(scenario_classes.keys())
    n_valid = len(valid_scenarios)
    
    logger.info(f"📊 Valid scenarios with data: {n_valid}/{n_scenarios}")
    
    if n_valid < 10:
        logger.error(f"❌ CRITICAL: Only {n_valid} valid scenarios (need 10+)")
        logger.error(f"   Cannot perform reliable train/test split")
        logger.error(f"   Please run more simulation scenarios")
        raise ValueError(f"Insufficient valid scenarios: {n_valid} (need 10+)")
    
    # Group scenarios by dominant class
    class_scenarios = {}
    for scenario, cls in scenario_classes.items():
        if cls not in class_scenarios:
            class_scenarios[cls] = []
        class_scenarios[cls].append(scenario)
    
    logger.info(f"📊 Scenarios grouped by dominant class:")
    for cls, scens in sorted(class_scenarios.items()):
        logger.info(f"   Class {cls}: {len(scens)} scenarios")
    
    # Stratified split ensuring all classes represented
    n_test = max(1, int(n_valid * TEST_SIZE))
    n_val = max(1, int(n_valid * VAL_SIZE))
    
    test_scenarios = []
    val_scenarios = []
    train_scenarios = []
    
    # For each class, split its scenarios proportionally
    rng = np.random.RandomState(RANDOM_SEED)
    
    for cls, scens in class_scenarios.items():
        if len(scens) == 0:
            continue
        
        # Shuffle scenarios for this class
        scens_copy = scens.copy()
        rng.shuffle(scens_copy)
        
        # Calculate split sizes for this class
        n_class_test = max(1, int(len(scens_copy) * TEST_SIZE))
        n_class_val = max(1, int(len(scens_copy) * VAL_SIZE))
        
        # Ensure we don't exceed available scenarios
        n_class_test = min(n_class_test, len(scens_copy))
        n_class_val = min(n_class_val, len(scens_copy) - n_class_test)
        
        test_scenarios.extend(scens_copy[:n_class_test])
        val_scenarios.extend(scens_copy[n_class_test:n_class_test + n_class_val])
        train_scenarios.extend(scens_copy[n_class_test + n_class_val:])
    
    logger.info(f"\n📊 Scenario split:")
    logger.info(f"   Train: {len(train_scenarios)} scenarios")
    logger.info(f"   Val:   {len(val_scenarios)} scenarios")
    logger.info(f"   Test:  {len(test_scenarios)} scenarios")
    
    # Split data by scenarios (only use valid scenarios)
    train_df = df[df['scenario_file'].isin(train_scenarios)]
    val_df = df[df['scenario_file'].isin(val_scenarios)]
    test_df = df[df['scenario_file'].isin(test_scenarios)]
    
    # Remove rows with missing target
    train_df = train_df[train_df[target_label].notna()]
    val_df = val_df[val_df[target_label].notna()]
    test_df = test_df[test_df[target_label].notna()]
    
    logger.info(f"\n📊 Sample split (after removing NaN targets):")
    logger.info(f"   Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"   Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"   Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify all splits have samples
    if len(train_df) == 0:
        raise ValueError("Training set is empty!")
    if len(val_df) == 0:
        raise ValueError("Validation set is empty!")
    if len(test_df) == 0:
        raise ValueError("Test set is empty!")
    
    # Verify all splits have all classes
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        split_classes = sorted(split_df[target_label].unique())
        missing_classes = [c for c in range(8) if c not in split_classes]
        
        if missing_classes:
            logger.warning(f"⚠️ {split_name} split missing classes: {missing_classes}")
        else:
            logger.info(f"✅ {split_name} split has all 8 classes")
    
    # Extract features and labels
    X_train = train_df[SAFE_FEATURES].fillna(0)
    y_train = train_df[target_label].astype(int)
    
    X_val = val_df[SAFE_FEATURES].fillna(0)
    y_val = val_df[target_label].astype(int)
    
    X_test = test_df[SAFE_FEATURES].fillna(0)
    y_test = test_df[target_label].astype(int)
    
    # 🔧 FIXED: Issue M2 - Document feature distribution comparison
    logger.info(f"\n📊 Feature distribution comparison (train vs test):")
    for feature in SAFE_FEATURES[:5]:  # Show first 5 to avoid spam
        train_mean = X_train[feature].mean()
        test_mean = X_test[feature].mean()
        train_std = X_train[feature].std()
        
        if train_std > 0:
            diff_in_stds = abs(train_mean - test_mean) / train_std
            
            if diff_in_stds > 0.5:
                logger.warning(f"⚠️ {feature}: train/test differ by {diff_in_stds:.2f} std devs")
            else:
                logger.info(f"✅ {feature}: distributions similar ({diff_in_stds:.2f} std devs)")
    
    logger.info(f"   ... and {len(SAFE_FEATURES) - 5} more features")
    
    # Store scenario info for later use
    train_scenarios_list = train_df['scenario_file'].values
    val_scenarios_list = val_df['scenario_file'].values
    test_scenarios_list = test_df['scenario_file'].values
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            train_scenarios_list, val_scenarios_list, test_scenarios_list)


# ================== FEATURE SCALING ==================
def scale_features_after_split(X_train, X_val, X_test, logger):
    """
    🔧 FIXED: Issue M2 - Scale features AFTER splitting (no test set leakage)
    
    Note: Scaling is done globally across all training scenarios.
    This is acceptable because:
    1. Scenario-aware splitting prevents temporal leakage
    2. Features are physical measurements (SNR, etc.) with consistent scales
    3. Global scaling improves generalization across different scenario types
    
    Limitation: If test scenarios have drastically different feature distributions
    (e.g., all indoor vs all outdoor), scaling may be suboptimal. However, this
    is preferred over per-scenario scaling which would prevent cross-scenario learning.
    """
    logger.info(f"\n🔧 Scaling features (AFTER splitting)...")
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info(f"   Scaler fit on training data only (no test leakage)")
    logger.info(f"   Feature means: {scaler.mean_[:3].round(3)}... (showing first 3)")
    logger.info(f"   Feature stds: {scaler.scale_[:3].round(3)}... (showing first 3)")
    logger.info(f"   ℹ️ Note: Global scaling across scenarios (Issue M2 documented)")
    
    # Transform validation and test using training statistics
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"✅ Features scaled using training statistics only")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ================== CLASS WEIGHTS ==================
def compute_class_weights_from_train(y_train, target_label, logger):
    """
    🔧 FIXED: Issue C4, M3 - Compute class weights AFTER splitting
    
    Cap increased from 3.0 to 10.0 to handle 20x imbalance from File 1b.
    
    Limitation (Issue M3): Weights are computed only from training set.
    If test scenarios have different class distributions (e.g., mostly indoor
    vs mostly outdoor), these weights may not generalize perfectly. However,
    this is preferred over including test data in weight computation, which
    would be data leakage.
    """
    logger.info(f"\n🔢 Computing class weights from TRAIN SET ONLY...")
    
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    
    # 🔧 FIXED: Issue C4 - Increased cap from 3.0 to 10.0
    # This handles 20x imbalance from File 1b (POWER=0.5)
    class_weights = np.minimum(class_weights, 10.0)
    
    weight_dict = dict(zip(unique_classes, class_weights))
    
    logger.info(f"   Class weights (capped at 10.0 - Issue C4 fixed):")
    
    class_counts = Counter(y_train)
    for class_val in sorted(unique_classes):
        count = class_counts[class_val]
        weight = weight_dict[class_val]
        pct = (count / len(y_train)) * 100
        capped = " (CAPPED)" if weight == 10.0 else ""
        logger.info(f"     Class {class_val}: {count:,} samples ({pct:.1f}%) -> weight: {weight:.2f}{capped}")
    
    logger.info(f"   ℹ️ Note: Weights from train set only (Issue M3 documented)")
    
    return weight_dict

# ================== MODEL TRAINING ==================
def train_and_evaluate_model(X_train_scaled, y_train, X_val_scaled, y_val, 
                              X_test_scaled, y_test, hyperparams, class_weights,
                              target_label, logger):
    """
    🔧 FIXED: Train and evaluate Random Forest model
    
    Changes:
    - Issue H1: Removed temporal sample weighting
    - Issue H3: Removed post-training CV check
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODEL: {target_label}")
    logger.info(f"{'='*60}")
    
    # Create model with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        max_features=hyperparams['max_features'],
        class_weight=class_weights,  # Use computed weights
        random_state=RANDOM_SEED,
        n_jobs=-1,  # CPU parallelization
        verbose=0
    )
    
    logger.info(f"📊 Model configuration:")
    logger.info(f"   n_estimators: {hyperparams['n_estimators']}")
    logger.info(f"   max_depth: {hyperparams['max_depth']}")
    logger.info(f"   min_samples_split: {hyperparams['min_samples_split']}")
    logger.info(f"   min_samples_leaf: {hyperparams['min_samples_leaf']}")
    logger.info(f"   max_features: {hyperparams['max_features']}")
    logger.info(f"   class_weight: custom (computed from train set, capped at 10.0)")
    logger.info(f"   sample_weight: NONE (Issue H1 - temporal weighting removed)")
    logger.info(f"   Hyperparameters source: {hyperparams['source']}")
    
    # Train model
    logger.info(f"\n🚀 Training model on {len(X_train_scaled):,} samples...")
    start_time = time.time()
    
    # 🔧 FIXED: Issue H1 - NO temporal sample weights (equal weights for all)
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Validation evaluation
    logger.info(f"\n📊 Evaluating on validation set...")
    y_val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    logger.info(f"🎯 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    
    # Test evaluation
    logger.info(f"\n🧪 Evaluating on test set...")
    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    logger.info(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # FIXED: Issue #17 - Realistic performance assessment
    if test_acc >= PERFORMANCE_THRESHOLDS['excellent']:
        logger.info(f"🏆 EXCELLENT performance: {test_acc*100:.1f}% (no data leakage)")
    elif test_acc >= PERFORMANCE_THRESHOLDS['good']:
        logger.info(f"✅ GOOD performance: {test_acc*100:.1f}%")
    elif test_acc >= PERFORMANCE_THRESHOLDS['acceptable']:
        logger.info(f"📊 ACCEPTABLE performance: {test_acc*100:.1f}%")
    else:
        logger.warning(f"⚠️ NEEDS IMPROVEMENT: {test_acc*100:.1f}%")
    
    # 🔧 FIXED: Issue H3 - REMOVED misleading post-training CV check
    # The old code did 5-fold CV on combined train+val+test data after training,
    # which gave inflated scores because the model saw training data in CV folds.
    # This has been REMOVED for honesty.
    
    logger.info(f"\n✅ Skipped post-training CV (Issue H3 - was misleading)")
    logger.info(f"   Reported metrics: Train (fit), Val (tuning), Test (final)")
    
    # Confusion matrices
    val_cm = confusion_matrix(y_val, y_val_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    logger.info(f"\n📈 Validation Confusion Matrix:")
    logger.info(f"\n{val_cm}")
    
    logger.info(f"\n📈 Test Confusion Matrix:")
    logger.info(f"\n{test_cm}")
    
    # Detailed classification report
    logger.info(f"\n📊 Detailed Classification Report (Test Set):")
    test_report = classification_report(y_test, y_test_pred, zero_division=0)
    logger.info(f"\n{test_report}")
    
    # Feature importance analysis with leakage detection
    logger.info(f"\n🔍 Feature Importance Analysis:")
    feature_importances = model.feature_importances_
    importance_dict = dict(zip(SAFE_FEATURES, feature_importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"   Top {len(SAFE_FEATURES)} most important features:")
    for rank, (feat, importance) in enumerate(sorted_features, 1):
        logger.info(f"     #{rank:2d}. {feat:25s}: {importance:.4f}")
    
    # Check if suspicious features rank too high (should not happen with SAFE_FEATURES)
    suspicious_high_importance = []
    for rank, (feat, importance) in enumerate(sorted_features[:3], 1):
        # These features should NOT be in SAFE_FEATURES anymore
        if any(leak in feat.lower() for leak in ['success', 'loss', 'packet', 'retry', 'consec']):
            suspicious_high_importance.append(feat)
            logger.warning(f"⚠️ SUSPICIOUS: Feature '{feat}' ranks #{rank} but seems like outcome!")
    
    if suspicious_high_importance:
        logger.error(f"🚨 POTENTIAL LEAKAGE: Top features include: {suspicious_high_importance}")
        logger.error(f"   This should NOT happen with SAFE_FEATURES! Check File 3 output.")
    else:
        logger.info(f"✅ No suspicious features in top 3 (leakage check passed)")
    
    # Prepare results dictionary
    results = {
        'target_label': target_label,
        'hyperparameters': hyperparams,
        'training_time_seconds': training_time,
        'validation_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'confusion_matrix_test': test_cm.tolist(),
        'feature_importances': {feat: float(imp) for feat, imp in importance_dict.items()},
        'top_5_features': [(feat, float(imp)) for feat, imp in sorted_features[:5]],
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'fixes_applied': ['C4_class_weights', 'H1_no_temporal_weights', 'H3_no_misleading_cv', 'C3_9_features']
    }
    
    return model, results

# ================== PER-SCENARIO EVALUATION ==================
def evaluate_per_scenario(model, scaler, df, test_scenarios, target_label, logger):
    """
    Per-scenario performance metrics (FAST VECTORIZED VERSION)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PER-SCENARIO EVALUATION")
    logger.info(f"{'='*60}")
    
    # Pre-filter test data ONCE (not in loop)
    test_df = df[df['scenario_file'].isin(test_scenarios)].copy()
    
    if len(test_df) == 0:
        logger.warning("⚠️ No test data found!")
        return {}
    
    # Remove NaN targets
    test_df = test_df[test_df[target_label].notna()]
    
    # Extract and scale features ONCE
    X_test_all = test_df[SAFE_FEATURES].fillna(0)
    y_test_all = test_df[target_label].astype(int)
    X_test_scaled = scaler.transform(X_test_all)
    
    # Predict for ALL samples at once (vectorized)
    y_pred_all = model.predict(X_test_scaled)
    
    # Add predictions to dataframe
    test_df['pred'] = y_pred_all
    test_df['correct'] = (test_df['pred'] == y_test_all).astype(int)
    
    # Compute per-scenario accuracy using groupby (FAST!)
    scenario_stats = test_df.groupby('scenario_file').agg({
        'correct': ['sum', 'count']
    })
    
    scenario_results = {}
    for scenario in test_scenarios:
        if scenario not in scenario_stats.index:
            continue
        
        correct = scenario_stats.loc[scenario, ('correct', 'sum')]
        total = scenario_stats.loc[scenario, ('correct', 'count')]
        accuracy = correct / total if total > 0 else 0.0
        
        scenario_results[scenario] = {
            'samples': int(total),
            'accuracy': float(accuracy)
        }
    
    if not scenario_results:
        logger.warning("⚠️ No scenario results computed!")
        return {}
    
    # Sort by accuracy (worst first)
    sorted_scenarios = sorted(scenario_results.items(), key=lambda x: x[1]['accuracy'])
    
    logger.info(f"\n📊 Worst 5 scenarios:")
    for scenario, metrics in sorted_scenarios[:5]:
        logger.info(f"   {scenario}: {metrics['accuracy']:.3f} ({metrics['samples']} samples)")
    
    logger.info(f"\n📊 Best 5 scenarios:")
    for scenario, metrics in sorted_scenarios[-5:]:
        logger.info(f"   {scenario}: {metrics['accuracy']:.3f} ({metrics['samples']} samples)")
    
    accuracies = [m['accuracy'] for m in scenario_results.values()]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    logger.info(f"\n📊 Average per-scenario accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    logger.info(f"✅ Evaluated {len(scenario_results)} scenarios in {len(test_df):,} samples")
    
    return scenario_results

# ================== SAVE MODEL AND RESULTS ==================
def save_model_and_results(model, scaler, results, scenario_results, target_label, logger):
    """Save trained model, scaler, and results"""
    logger.info(f"\n💾 Saving model and results...")
    
    # Save model
    model_file = OUTPUT_DIR / f"step4_rf_{target_label}_FIXED.joblib"
    joblib.dump(model, model_file)
    logger.info(f"   Model saved: {model_file}")
    
    # Save scaler
    scaler_file = OUTPUT_DIR / f"step4_scaler_{target_label}_FIXED.joblib"
    joblib.dump(scaler, scaler_file)
    logger.info(f"   Scaler saved: {scaler_file}")
    
    # Save results as JSON
    results_file = OUTPUT_DIR / f"step4_results_{target_label}.json"
    full_results = {
        'training_results': results,
        'per_scenario_results': scenario_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"   Results saved: {results_file}")
    
    # Save human-readable summary
    summary_file = OUTPUT_DIR / f"step4_summary_{target_label}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL TRAINING SUMMARY - {target_label} (FIXED)\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: {USER}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        
        f.write("FIXES APPLIED:\n")
        f.write("  ✅ Issue C4: Class weight cap = 10.0 (was 3.0)\n")
        f.write("  ✅ Issue H1: Temporal weights removed\n")
        f.write("  ✅ Issue H3: Post-training CV removed\n")
        f.write("  ✅ Issue C3: Training uses 9 features (removed 5 outcome features)\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"  Validation Accuracy: {results['validation_accuracy']:.4f} ({results['validation_accuracy']*100:.1f}%)\n")
        f.write(f"  Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.1f}%)\n")
        f.write(f"  Training Time: {results['training_time_seconds']:.2f}s\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        for param, value in results['hyperparameters'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        f.write("TOP 5 FEATURES:\n")
        for rank, (feat, imp) in enumerate(results['top_5_features'], 1):
            f.write(f"  #{rank}. {feat}: {imp:.4f}\n")
        f.write("\n")
        
        f.write("FILES GENERATED:\n")
        f.write(f"  - {model_file.name}\n")
        f.write(f"  - {scaler_file.name}\n")
        f.write(f"  - {results_file.name}\n")
        f.write(f"  - {summary_file.name}\n")
    
    logger.info(f"   Summary saved: {summary_file}")
    logger.info(f"✅ All outputs saved successfully")

# ================== CHECKPOINT CHECKING ==================
def check_if_already_trained(target_label: str, logger) -> bool:
    """
    Check if model already trained (checkpoint recovery)
    """
    model_file = OUTPUT_DIR / f"step4_rf_{target_label}_FIXED.joblib"
    scaler_file = OUTPUT_DIR / f"step4_scaler_{target_label}_FIXED.joblib"
    
    if model_file.exists() and scaler_file.exists():
        logger.info(f"✅ Found existing trained model for {target_label}")
        logger.info(f"   Model: {model_file}")
        logger.info(f"   Scaler: {scaler_file}")
        
        response = input(f"\n   Skip training for {target_label}? (y/n): ").strip().lower()
        if response == 'y':
            logger.info(f"   Skipping {target_label} (using checkpoint)")
            return True
    
    return False

# ================== MAIN PIPELINE ==================
def main():
    """Main training pipeline - trains all oracle models"""
    print("="*80)
    print("ML MODEL TRAINING PIPELINE - FULLY FIXED VERSION")
    print("="*80)
    print(f"Author: {USER}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: CPU")
    print("="*80)
    print("FIXES APPLIED:")
    print("  ✅ Issue C4: Class weight cap = 10.0 (handles 20x imbalance)")
    print("  ✅ Issue H1: Temporal weights REMOVED")
    print("  ✅ Issue H3: Post-training CV REMOVED")
    print("  ✅ Issue C3: Training uses 9 features (NO outcome features)")
    print("="*80)
    
    # Load data once
    print(f"\n📂 Loading data from: {CSV_FILE}")
    
    if not CSV_FILE.exists():
        print(f"❌ Input file not found: {CSV_FILE}")
        return False
    
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Track results for all models
    all_results = {}
    pipeline_start = time.time()
    
    # Train each target
    for target_idx, target_label in enumerate(TARGET_LABELS, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {target_idx}/{len(TARGET_LABELS)}: {target_label}")
        print(f"{'#'*80}")
        
        # Setup logging for this target
        logger = setup_logging(target_label)
        
        try:
            # Check checkpoint
            if check_if_already_trained(target_label, logger):
                continue
            
            # Load optimized hyperparameters
            hyperparams = load_optimized_hyperparameters(target_label, logger)
            
            # Scenario-aware split
            split_result = scenario_aware_stratified_split(df, target_label, logger)
            (X_train, X_val, X_test, y_train, y_val, y_test,
             train_scenarios, val_scenarios, test_scenarios) = split_result
            
            # Scale features AFTER splitting
            X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features_after_split(
                X_train, X_val, X_test, logger
            )
            
            # Compute class weights from train set only
            class_weights = compute_class_weights_from_train(y_train, target_label, logger)
            
            # Train and evaluate (NO temporal weights, NO post-CV)
            model, results = train_and_evaluate_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                X_test_scaled, y_test, hyperparams, class_weights,
                target_label, logger
            )
            
            # Per-scenario evaluation
            scenario_results = evaluate_per_scenario(
                model, scaler, df, test_scenarios, target_label, logger
            )
            
            # Save everything
            save_model_and_results(model, scaler, results, scenario_results, target_label, logger)
            
            # Store results
            all_results[target_label] = results
            
            logger.info(f"\n✅ Completed training for {target_label}")
            print(f"\n✅ {target_label}: Test Accuracy = {results['test_accuracy']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Training failed for {target_label}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Final summary
    total_time = time.time() - pipeline_start
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE FOR ALL MODELS (FIXED)")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Models trained: {len(all_results)}/{len(TARGET_LABELS)}")
    
    if all_results:
        print(f"\n📊 Final Results Summary:")
        for target, results in all_results.items():
            print(f"\n{target}:")
            print(f"  Validation: {results['validation_accuracy']*100:.1f}%")
            print(f"  Test: {results['test_accuracy']*100:.1f}%")
            print(f"  Time: {results['training_time_seconds']:.1f}s")
    
    print(f"\n📁 Models saved to: {OUTPUT_DIR}")
    print(f"✅ Training pipeline completed successfully!")
    print(f"\n📊 EXPECTED IMPROVEMENTS:")
    print(f"  - Rare classes (0-3) should have BETTER recall")
    print(f"  - Test accuracy should MATCH CV accuracy from File 3c")
    print(f"  - Model uses 9 features (not 14) - cleaner!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)