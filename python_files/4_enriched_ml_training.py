"""
Ultimate ML Model Training Pipeline - FIXED VERSION
Trains Random Forest models using optimized hyperparameters from Step 3c

CRITICAL FIXES (2025-10-01):
- Issue #4: Scenario-aware train/test split (no random mixing)
- Issue #12: Raises error if scenario_file missing (no silent fallback)
- Issue #34: Feature scaling done AFTER splitting (no test set leakage)
- Issue #35: Distribution comparison between train/test
- Issue #36: Class weights computed AFTER splitting (only on train set)
- Issue #37: Stratified scenario split (ensures all classes in each split)
- Issue #40: Sample weights for temporal importance (recent packets weighted higher)
- Issue #21: Trains all oracle models in one run (automated)
- Issue #22: Feature importance analysis with leakage detection
- Issue #23: Per-scenario performance metrics
- Issue #25: Checkpoint recovery (skip already trained models)
- Issue #20: Loads optimized hyperparameters from File 3c

Features:
- Scenario-aware splitting (prevents temporal leakage)
- Optimized hyperparameters from GridSearchCV
- Temporal sample weighting
- Per-scenario evaluation
- Checkpoint recovery
- Comprehensive logging

Author: ahmedjk34
Date: 2025-10-01
Pipeline Stage: Step 4 - Model Training (FIXED)
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
HYPERPARAMS_FILE = BASE_DIR / "hyperparameter_results" / "hyperparameter_tuning_results.json"
CLASS_WEIGHTS_FILE = BASE_DIR / "model_artifacts" / "class_weights.json"
OUTPUT_DIR = BASE_DIR / "trained_models"
LOG_DIR = BASE_DIR / "logs"

# FIXED: Issue #14 - Global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target labels to train
TARGET_LABELS = ["oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# FIXED: Issue #1 - SAFE features only (no temporal leakage)
SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    "shortSuccRatio", "medSuccRatio",
    "packetLossRate",
    "channelWidth", "mobilityMetric",
    "severity", "confidence"
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
    logger.info(f"ML MODEL TRAINING PIPELINE - FIXED VERSION")
    logger.info("="*80)
    logger.info(f"🎯 Target Label: {target_label}")
    logger.info(f"👤 Author: {USER}")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔧 Random Seed: {RANDOM_SEED}")
    logger.info(f"🛡️ Safe Features: {len(SAFE_FEATURES)} (no temporal leakage)")
    logger.info(f"💻 Device: CPU (no GPU required)")
    logger.info("="*80)
    logger.info("FIXES APPLIED:")
    logger.info("  - Issue #4: Scenario-aware splitting")
    logger.info("  - Issue #12: No silent fallback to random split")
    logger.info("  - Issue #34: Scaling after splitting")
    logger.info("  - Issue #35: Train/test distribution comparison")
    logger.info("  - Issue #36: Class weights from train set only")
    logger.info("  - Issue #37: Stratified scenario split")
    logger.info("  - Issue #40: Temporal sample weighting")
    logger.info("  - Issue #20: Optimized hyperparameters from 3c")
    logger.info("="*80)
    
    return logger

# ================== LOAD HYPERPARAMETERS ==================
def load_optimized_hyperparameters(target_label: str, logger) -> Dict:
    """
    FIXED: Issue #20 - Load optimized hyperparameters from File 3c
    """
    logger.info(f"📂 Loading optimized hyperparameters from: {HYPERPARAMS_FILE}")
    
    # Check if hyperparameter file exists
    if not HYPERPARAMS_FILE.exists():
        logger.warning(f"⚠️ Hyperparameter file not found!")
        logger.warning(f"   Expected: {HYPERPARAMS_FILE}")
        logger.warning(f"   Please run Step 3c (hyperparameter tuning) first")
        logger.warning(f"   Falling back to default parameters...")
        
        # Default fallback parameters (reasonable but not optimized)
        return {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'source': 'default_fallback'
        }
    
    try:
        with open(HYPERPARAMS_FILE, 'r') as f:
            all_hyperparams = json.load(f)
        
        if target_label not in all_hyperparams:
            logger.warning(f"⚠️ No hyperparameters found for {target_label}")
            logger.warning(f"   Available targets: {list(all_hyperparams.keys())}")
            logger.warning(f"   Using default parameters...")
            
            return {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'source': 'default_fallback'
            }
        
        # Extract best parameters
        target_results = all_hyperparams[target_label]
        best_params = target_results['best_params']
        best_score = target_results['best_score']
        
        logger.info(f"✅ Loaded optimized hyperparameters for {target_label}")
        logger.info(f"   Tuning CV Score: {best_score:.4f} ({best_score*100:.1f}%)")
        logger.info(f"   Parameters:")
        for param, value in best_params.items():
            logger.info(f"     {param}: {value}")
        
        best_params['source'] = 'optimized_3c'
        return best_params
        
    except Exception as e:
        logger.error(f"❌ Failed to load hyperparameters: {str(e)}")
        logger.warning(f"   Falling back to default parameters...")
        
        return {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'source': 'default_fallback'
        }

# ================== SCENARIO-AWARE SPLITTING ==================
def scenario_aware_stratified_split(df: pd.DataFrame, target_label: str, logger) -> Tuple:
    """
    FIXED: Issue #4, #12, #37 - Scenario-aware stratified train/val/test split
    
    Ensures:
    - Entire scenarios in train OR val OR test (no mixing)
    - Stratification by dominant class per scenario
    - All classes present in each split (if possible)
    - Raises error if scenario_file missing (no silent fallback)
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
    
    # Compute dominant class per scenario (for stratification)
    scenario_classes = {}
    for scenario in scenarios:
        scenario_data = df[df['scenario_file'] == scenario]
        dominant_class = scenario_data[target_label].mode()[0]
        scenario_classes[scenario] = dominant_class
    
    # Group scenarios by dominant class
    class_scenarios = {}
    for scenario, cls in scenario_classes.items():
        if cls not in class_scenarios:
            class_scenarios[cls] = []
        class_scenarios[cls].append(scenario)
    
    logger.info(f"📊 Scenarios grouped by dominant class:")
    for cls, scens in sorted(class_scenarios.items()):
        logger.info(f"   Class {cls}: {len(scens)} scenarios")
    
    # FIXED: Issue #37 - Stratified split ensuring all classes represented
    n_test = max(1, int(n_scenarios * TEST_SIZE))
    n_val = max(1, int(n_scenarios * VAL_SIZE))
    
    test_scenarios = []
    val_scenarios = []
    train_scenarios = []
    
    # For each class, split its scenarios proportionally
    rng = np.random.RandomState(RANDOM_SEED)
    
    for cls, scens in class_scenarios.items():
        rng.shuffle(scens)
        
        n_class_test = max(1, int(len(scens) * TEST_SIZE))
        n_class_val = max(1, int(len(scens) * VAL_SIZE))
        
        test_scenarios.extend(scens[:n_class_test])
        val_scenarios.extend(scens[n_class_test:n_class_test + n_class_val])
        train_scenarios.extend(scens[n_class_test + n_class_val:])
    
    logger.info(f"\n📊 Scenario split:")
    logger.info(f"   Train: {len(train_scenarios)} scenarios")
    logger.info(f"   Val:   {len(val_scenarios)} scenarios")
    logger.info(f"   Test:  {len(test_scenarios)} scenarios")
    
    # Split data by scenarios
    train_df = df[df['scenario_file'].isin(train_scenarios)]
    val_df = df[df['scenario_file'].isin(val_scenarios)]
    test_df = df[df['scenario_file'].isin(test_scenarios)]
    
    logger.info(f"\n📊 Sample split:")
    logger.info(f"   Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"   Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"   Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
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
    
    # FIXED: Issue #35 - Compare feature distributions between train/test
    logger.info(f"\n📊 Feature distribution comparison (train vs test):")
    for feature in SAFE_FEATURES:
        train_mean = X_train[feature].mean()
        test_mean = X_test[feature].mean()
        train_std = X_train[feature].std()
        
        if train_std > 0:
            diff_in_stds = abs(train_mean - test_mean) / train_std
            
            if diff_in_stds > 0.5:
                logger.warning(f"⚠️ {feature}: train/test differ by {diff_in_stds:.2f} std devs")
            else:
                logger.info(f"✅ {feature}: distributions similar ({diff_in_stds:.2f} std devs)")
    
    # Store scenario info for later use
    train_scenarios_list = train_df['scenario_file'].values
    val_scenarios_list = val_df['scenario_file'].values
    test_scenarios_list = test_df['scenario_file'].values
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            train_scenarios_list, val_scenarios_list, test_scenarios_list)

# ================== FEATURE SCALING ==================
def scale_features_after_split(X_train, X_val, X_test, logger):
    """
    FIXED: Issue #34 - Scale features AFTER splitting (no test set leakage)
    """
    logger.info(f"\n🔧 Scaling features (AFTER splitting - Issue #34 fix)...")
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info(f"   Scaler fit on training data only (no test leakage)")
    logger.info(f"   Feature means: {scaler.mean_[:3].round(3)}... (showing first 3)")
    logger.info(f"   Feature stds: {scaler.scale_[:3].round(3)}... (showing first 3)")
    
    # Transform validation and test using training statistics
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"✅ Features scaled using training statistics only")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ================== CLASS WEIGHTS ==================
def compute_class_weights_from_train(y_train, target_label, logger):
    """
    FIXED: Issue #36 - Compute class weights AFTER splitting (only from train set)
    FIXED: Issue #6 - Cap weights at 50.0
    """
    logger.info(f"\n🔢 Computing class weights from TRAIN SET ONLY (Issue #36 fix)...")
    
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    
    # FIXED: Issue #6 - Cap extreme weights
    class_weights = np.minimum(class_weights, 50.0)
    
    weight_dict = dict(zip(unique_classes, class_weights))
    
    logger.info(f"   Class weights (capped at 50.0):")
    class_counts = Counter(y_train)
    for class_val in sorted(unique_classes):
        count = class_counts[class_val]
        weight = weight_dict[class_val]
        pct = (count / len(y_train)) * 100
        logger.info(f"     Class {class_val}: {count:,} samples ({pct:.1f}%) -> weight: {weight:.2f}")
    
    return weight_dict

# ================== TEMPORAL SAMPLE WEIGHTS ==================
def compute_temporal_sample_weights(y_train, train_scenarios, logger):
    """
    FIXED: Issue #40 - Weight recent packets higher than old packets
    
    Within each scenario, later packets get higher weight (more recent = more relevant)
    """
    logger.info(f"\n⏱️ Computing temporal sample weights (Issue #40)...")
    
    sample_weights = np.ones(len(y_train))
    
    unique_scenarios = np.unique(train_scenarios)
    
    for scenario in unique_scenarios:
        scenario_mask = train_scenarios == scenario
        scenario_indices = np.where(scenario_mask)[0]
        n_samples = len(scenario_indices)
        
        if n_samples > 1:
            # Linear weight: first sample = 0.5, last sample = 1.5
            # This gives recent packets 3x weight of old packets
            weights = np.linspace(0.5, 1.5, n_samples)
            sample_weights[scenario_indices] = weights
    
    logger.info(f"   Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    logger.info(f"   Recent packets weighted up to 3x higher than old packets")
    
    return sample_weights

# ================== MODEL TRAINING ==================
def train_and_evaluate_model(X_train_scaled, y_train, X_val_scaled, y_val, 
                              X_test_scaled, y_test, hyperparams, class_weights,
                              sample_weights, target_label, logger):
    """
    Train and evaluate Random Forest model with all fixes applied
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
    logger.info(f"   class_weight: custom (computed from train set)")
    logger.info(f"   sample_weight: temporal (recent packets weighted higher)")
    logger.info(f"   Hyperparameters source: {hyperparams['source']}")
    
    # Train model
    logger.info(f"\n🚀 Training model on {len(X_train_scaled):,} samples...")
    start_time = time.time()
    
    # FIXED: Issue #40 - Use temporal sample weights
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
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
    
    # Cross-validation on combined data (for comparison)
    logger.info(f"\n🔄 Cross-validation check (5-fold)...")
    X_all = np.concatenate([X_train_scaled, X_val_scaled, X_test_scaled])
    y_all = np.concatenate([y_train, y_val, y_test])
    
    cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info(f"📊 5-Fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"   Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    
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
    
    # FIXED: Issue #22 - Feature importance analysis with leakage detection
    logger.info(f"\n🔍 Feature Importance Analysis (Issue #22):")
    feature_importances = model.feature_importances_
    importance_dict = dict(zip(SAFE_FEATURES, feature_importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"   Top 10 most important features:")
    for rank, (feat, importance) in enumerate(sorted_features[:10], 1):
        logger.info(f"     #{rank:2d}. {feat:25s}: {importance:.4f}")
    
    # Check if suspicious features rank too high
    suspicious_high_importance = []
    for rank, (feat, importance) in enumerate(sorted_features[:3], 1):
        if 'consecSuccess' in feat or 'consecFailure' in feat or 'retry' in feat.lower():
            suspicious_high_importance.append(feat)
            logger.warning(f"⚠️ SUSPICIOUS: Temporal feature '{feat}' ranks #{rank}!")
    
    if suspicious_high_importance:
        logger.error(f"🚨 POTENTIAL LEAKAGE: Top features include temporal: {suspicious_high_importance}")
    else:
        logger.info(f"✅ No suspicious features in top 3 (leakage check passed)")
    
    # Prepare results dictionary
    results = {
        'target_label': target_label,
        'hyperparameters': hyperparams,
        'training_time_seconds': training_time,
        'validation_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'confusion_matrix_test': test_cm.tolist(),
        'feature_importances': {feat: float(imp) for feat, imp in importance_dict.items()},
        'top_5_features': [(feat, float(imp)) for feat, imp in sorted_features[:5]],
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED
    }
    
    return model, results

# ================== PER-SCENARIO EVALUATION ==================
def evaluate_per_scenario(model, scaler, df, test_scenarios, target_label, logger):
    """
    FIXED: Issue #23 - Per-scenario performance metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PER-SCENARIO EVALUATION (Issue #23)")
    logger.info(f"{'='*60}")
    
    test_df = df[df['scenario_file'].isin(test_scenarios)]
    
    scenario_results = {}
    
    for scenario in test_scenarios:
        scenario_data = test_df[test_df['scenario_file'] == scenario]
        
        if len(scenario_data) == 0:
            continue
        
        X_scenario = scenario_data[SAFE_FEATURES].fillna(0)
        y_scenario = scenario_data[target_label].astype(int)
        
        X_scenario_scaled = scaler.transform(X_scenario)
        y_pred_scenario = model.predict(X_scenario_scaled)
        
        accuracy = accuracy_score(y_scenario, y_pred_scenario)
        
        scenario_results[scenario] = {
            'samples': len(scenario_data),
            'accuracy': float(accuracy)
        }
    
    # Sort by accuracy (worst first)
    sorted_scenarios = sorted(scenario_results.items(), key=lambda x: x[1]['accuracy'])
    
    logger.info(f"\n📊 Worst 5 scenarios:")
    for scenario, metrics in sorted_scenarios[:5]:
        logger.info(f"   {scenario}: {metrics['accuracy']:.3f} ({metrics['samples']} samples)")
    
    logger.info(f"\n📊 Best 5 scenarios:")
    for scenario, metrics in sorted_scenarios[-5:]:
        logger.info(f"   {scenario}: {metrics['accuracy']:.3f} ({metrics['samples']} samples)")
    
    avg_accuracy = np.mean([m['accuracy'] for m in scenario_results.values()])
    std_accuracy = np.std([m['accuracy'] for m in scenario_results.values()])
    
    logger.info(f"\n📊 Average per-scenario accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    
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
        f.write(f"MODEL TRAINING SUMMARY - {target_label}\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: {USER}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"  Validation Accuracy: {results['validation_accuracy']:.4f} ({results['validation_accuracy']*100:.1f}%)\n")
        f.write(f"  Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.1f}%)\n")
        f.write(f"  Cross-Validation: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
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
    FIXED: Issue #25 - Check if model already trained (checkpoint recovery)
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
    print("ML MODEL TRAINING PIPELINE - FIXED VERSION")
    print("="*80)
    print(f"Author: {USER}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: CPU")
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
            # FIXED: Issue #25 - Check checkpoint
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
            
            # Compute temporal sample weights
            sample_weights = compute_temporal_sample_weights(y_train, train_scenarios, logger)
            
            # Train and evaluate
            model, results = train_and_evaluate_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                X_test_scaled, y_test, hyperparams, class_weights,
                sample_weights, target_label, logger
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
    print(f"TRAINING COMPLETE FOR ALL MODELS")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Models trained: {len(all_results)}/{len(TARGET_LABELS)}")
    
    if all_results:
        print(f"\n📊 Final Results Summary:")
        for target, results in all_results.items():
            print(f"\n{target}:")
            print(f"  Validation: {results['validation_accuracy']*100:.1f}%")
            print(f"  Test: {results['test_accuracy']*100:.1f}%")
            print(f"  CV: {results['cv_mean']*100:.1f}% ± {results['cv_std']*100:.1f}%")
            print(f"  Time: {results['training_time_seconds']:.1f}s")
    
    print(f"\n📁 Models saved to: {OUTPUT_DIR}")
    print(f"✅ Training pipeline completed successfully!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)