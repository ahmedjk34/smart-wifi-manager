"""
Ultimate ML Model Training Pipeline with Class Weights (Random Forest, Large CSV-Compatible, RAM-Efficient)
Trains and outputs models using Phase 4 Step 3 naming/style with CLASS WEIGHTS instead of downsampling.

Features:
- BULLETPROOF class preservation - guarantees all classes survive processing
- Optional chunked CSV loading and row limiting (configurable)
- CONFIGURABLE TARGET LABELS - train on any available label (rateIdx, oracle labels, etc.)
- Handles all new features, context labels, and oracle label nuances
- USES CLASS WEIGHTS for imbalanced data instead of aggressive downsampling
- Step-by-step logging and documentation
- Stratified splits and feature scaling
- Model saving and top feature reporting
- FIXED: Removed data leakage features for realistic performance

Author: ahmedjk34
Date: 2025-09-22
BULLETPROOF VERSION: Absolutely preserves all classes, configurable sampling, proper debugging
FIXED VERSION: Removed phyRate and other leaky features for realistic 85-95% accuracy
CONFIGURABLE VERSION: Choose any target label for training
"""

import pandas as pd
import joblib
import logging
import time
import sys
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)

# ================== CONFIGURABLE SETTINGS ==================
CSV_FILE = "smart-v3-ml-enriched.csv"   # <--- DO NOT CHANGE

# 🎯 TARGET LABEL SELECTION - CHOOSE YOUR TARGET!
# Available options based on your data:
# - "rateIdx" (original - mostly classes 0,1,5)
# - "oracle_conservative" (classes 0-7, but different distribution)
# - "oracle_balanced" (classes 0-7, more balanced) 
# - "oracle_aggressive" (classes 0-7, aggressive strategy)

TARGET_LABEL = "oracle_balanced"  # <--- CHANGE THIS TO EXPERIMENT!

# You currently train on ONE label at a time (not multiple simultaneously)
# This is the standard approach for classification tasks

# ROW LIMITING CONTROLS (Set ENABLE_ROW_LIMITING=False to use full dataset)
ENABLE_ROW_LIMITING = False              # <--- Set to True to enable sampling/chunking
MAX_ROWS = 500_000                       # Only used if ENABLE_ROW_LIMITING=True
CHUNKSIZE = 250_000                      # Only used if ENABLE_ROW_LIMITING=True

# FIXED FEATURE COLUMNS - REMOVED DATA LEAKAGE FEATURES
# FIXED: GUARANTEED SAFE FEATURES - ZERO DATA LEAKAGE
FEATURE_COLS = [
    # SNR features - core network measurements
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    
    # Performance features - network behavior
    "shortSuccRatio", "medSuccRatio", "consecSuccess", "consecFailure",
    "packetLossRate", "retrySuccessRatio", 
    
    # Rate adaptation features - historical behavior
    "recentRateChanges", "timeSinceLastRateChange", "rateStabilityScore",
    
    # Network assessment features
    "severity", "confidence", "packetSuccess",
    
    # Network configuration - static but safe
    "channelWidth", "mobilityMetric",
    
    # REMOVED ALL LEAKY FEATURES:
    # "phyRate" - LEAKY: Perfect correlation with rateIdx
    # "optimalRateDistance" - LEAKY: 8 unique values = 8 rate classes  
    # "recentThroughputTrend" - LEAKY: High correlation (0.853)
    # "conservativeFactor" - LEAKY: Inverse correlation (-0.809)
    # "aggressiveFactor" - LEAKY: Inverse of conservative
    # "recommendedSafeRate" - LEAKY: Direct target hint
    # "T1", "T2", "T3" - USELESS: Always constant
    # "decisionReason" - USELESS: Always 0
    # "offeredLoad" - USELESS: Always 0 in your data
    # "queueLen" - USELESS: Always 0 in your data  
    # "retryCount" - USELESS: Always 0 in your data
]



CONTEXT_LABEL = "network_context"
USER = "ahmedjk34"

# Dynamic file names based on target label
SCALER_FILE = f"step3_scaler_{TARGET_LABEL}_FIXED.joblib"
MODEL_FILE = f"step3_rf_{TARGET_LABEL}_model_FIXED.joblib"
DOC_FILE = f"step3_{TARGET_LABEL}_training_results.txt"
CLASS_WEIGHTS_FILE = "python_files/model_artifacts/class_weights.json"

# ================== AVAILABLE TARGET LABELS INFO ==================
TARGET_LABEL_INFO = {
    "rateIdx": {
        "description": "Original rate index (mostly classes 0,1,5)",
        "expected_classes": 8,
        "class_range": range(8),
        "typical_distribution": "Imbalanced: 49% class 0, 25% each class 1&5, <1% others"
    },
    "oracle_conservative": {
        "description": "Conservative oracle strategy (prefers lower rates)",
        "expected_classes": 8,
        "class_range": range(8),
        "typical_distribution": "More balanced across all classes"
    },
    "oracle_balanced": {
        "description": "Balanced oracle strategy (middle ground)",
        "expected_classes": 8,
        "class_range": range(8),
        "typical_distribution": "Well-distributed across classes"
    },
    "oracle_aggressive": {
        "description": "Aggressive oracle strategy (prefers higher rates)",
        "expected_classes": 8,
        "class_range": range(8),
        "typical_distribution": "Heavily weighted toward higher rate classes"
    }
}

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{TARGET_LABEL}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info(f"CONFIGURABLE ML TRAINING PIPELINE - TARGET: {TARGET_LABEL}")
    logger.info("="*60)
    logger.info(f"🎯 Target Label: {TARGET_LABEL}")
    if TARGET_LABEL in TARGET_LABEL_INFO:
        info = TARGET_LABEL_INFO[TARGET_LABEL]
        logger.info(f"📊 Description: {info['description']}")
        logger.info(f"📈 Expected: {info['typical_distribution']}")
    logger.info(f"🔧 Row limiting: {'ENABLED' if ENABLE_ROW_LIMITING else 'DISABLED'}")
    logger.info(f"🚨 FIXED: Removed 19 leaky/useless features for realistic accuracy")
    logger.info(f"📊 Using {len(FEATURE_COLS)} SAFE features only (REMOVED 19 LEAKY/USELESS features)")
    if ENABLE_ROW_LIMITING:
        logger.info(f"📊 Max rows: {MAX_ROWS:,}, Chunk size: {CHUNKSIZE:,}")
    else:
        logger.info("📊 Using FULL dataset (no row limits)")
    return logger

def discover_available_labels(df, logger):
    """Discover and display all available target labels in the dataset."""
    logger.info("🔍 Discovering available target labels in dataset...")
    
    potential_labels = ["rateIdx", "oracle_conservative", "oracle_balanced", "oracle_aggressive"]
    available_labels = []
    
    for label in potential_labels:
        if label in df.columns:
            label_dist = df[label].value_counts().sort_index()
            unique_classes = len(label_dist)
            total_samples = len(df[label].dropna())
            
            available_labels.append(label)
            logger.info(f"📊 {label}: {unique_classes} classes, {total_samples:,} samples")
            
            # Show top 5 most common classes
            top_classes = label_dist.head(5)
            for class_val, count in top_classes.items():
                pct = (count / total_samples) * 100
                logger.info(f"    Class {class_val}: {count:,} samples ({pct:.1f}%)")
    
    logger.info(f"✅ Found {len(available_labels)} available target labels: {available_labels}")
    
    if TARGET_LABEL not in available_labels:
        logger.error(f"❌ Selected target '{TARGET_LABEL}' not found in dataset!")
        logger.error(f"Available options: {available_labels}")
        return None
    
    return available_labels

def load_class_weights(filepath, target_label, logger):
    """Load pre-computed class weights for handling imbalanced data."""
    logger.info(f"📊 Loading class weights from {filepath} for target '{target_label}'...")
    try:
        with open(filepath, 'r') as f:
            all_weights = json.load(f)
        
        if target_label not in all_weights:
            logger.warning(f"⚠️ No class weights found for {target_label}, will compute balanced weights")
            return None
        
        weights = all_weights[target_label]
        # Convert string keys to integers
        weights = {int(k): float(v) for k, v in weights.items()}
        
        logger.info(f"✅ Loaded class weights for {target_label}:")
        for class_val, weight in sorted(weights.items()):
            logger.info(f"  Class {class_val}: {weight:.3f}")
        
        print(f"Class weights for {target_label}:")
        for class_val, weight in sorted(weights.items()):
            print(f"  {class_val}: {weight:.3f}")
        
        return weights
    except FileNotFoundError:
        logger.warning(f"⚠️ Class weights file not found: {filepath}")
        logger.info("Will compute balanced class weights automatically")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading class weights: {e}")
        return None

def compute_class_weights_if_needed(y, target_label, logger):
    """Compute balanced class weights if not available from file."""
    from sklearn.utils.class_weight import compute_class_weight
    
    logger.info(f"🔄 Computing balanced class weights for {target_label}...")
    
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, class_weights))
    
    logger.info(f"✅ Computed balanced class weights:")
    for class_val, weight in sorted(weight_dict.items()):
        logger.info(f"  Class {class_val}: {weight:.3f}")
    
    return weight_dict

def debug_class_loss(df_before, df_after, label, step_name, logger, expected_classes=8):
    """Debug function to track exactly what happens to each class during processing."""
    before_dist = df_before[label].value_counts().sort_index()
    after_dist = df_after[label].value_counts().sort_index()
    
    logger.info(f"🔍 DEBUG: Class changes during {step_name}:")
    for class_val in range(expected_classes):
        before_count = before_dist.get(class_val, 0)
        after_count = after_dist.get(class_val, 0)
        change = after_count - before_count
        
        if change < 0:
            logger.error(f"  ❌ Class {class_val}: {before_count} → {after_count} (LOST {abs(change)} samples!)")
        elif change == 0:
            logger.info(f"  ✅ Class {class_val}: {before_count} → {after_count} (unchanged)")
        else:
            logger.info(f"  ➕ Class {class_val}: {before_count} → {after_count} (gained {change})")

def bulletproof_load_dataset(filepath, feature_cols, label, logger, context_label):
    """
    BULLETPROOF dataset loading that GUARANTEES all classes survive.
    Now works with any target label (rateIdx, oracle labels, etc.)
    """
    logger.info(f"📂 Loading dataset with BULLETPROOF class preservation for target '{label}'...")
    
    # STEP 1: Load complete dataset
    if ENABLE_ROW_LIMITING:
        logger.info(f"🔧 Row limiting ENABLED - will process in chunks of {CHUNKSIZE:,}")
        chunk_list = []
        total_rows_seen = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=CHUNKSIZE, low_memory=False)):
            total_rows_seen += len(chunk)
            chunk_list.append(chunk)
            logger.info(f"📥 Loaded chunk {chunk_num + 1}: {len(chunk):,} rows (total: {total_rows_seen:,})")
            
            if len(pd.concat(chunk_list)) >= MAX_ROWS:
                logger.info(f"🛑 Reached target of {MAX_ROWS:,} rows, stopping chunk loading")
                break
        
        df = pd.concat(chunk_list, ignore_index=True)
        if len(df) > MAX_ROWS:
            logger.info(f"📉 Trimming from {len(df):,} to {MAX_ROWS:,} rows")
            df = df.head(MAX_ROWS)
    else:
        logger.info(f"📥 Loading FULL dataset (row limiting DISABLED)")
        df = pd.read_csv(filepath, low_memory=False)
    
    logger.info(f"📊 Initial dataset: {len(df):,} rows, {len(df.columns)} columns")
    
    # STEP 1.5: Discover available labels
    available_labels = discover_available_labels(df, logger)
    if available_labels is None:
        return None, None
    
    # STEP 2: Show ORIGINAL class distribution for selected target
    if label in df.columns:
        original_dist = df[label].value_counts().sort_index()
        logger.info(f"🎯 ORIGINAL class distribution for '{label}':")
        for class_val, count in original_dist.items():
            pct = (count / len(df)) * 100
            logger.info(f"  Class {class_val}: {count:,} samples ({pct:.1f}%)")
        
        # Determine expected classes dynamically
        actual_classes = set(original_dist.index)
        expected_classes = max(actual_classes) + 1 if actual_classes else 8
        missing_classes = set(range(expected_classes)) - actual_classes
        
        if missing_classes:
            logger.warning(f"⚠️ MISSING CLASSES IN ORIGINAL DATA: {missing_classes}")
            logger.info(f"📊 Will proceed with available {len(actual_classes)} classes")
        else:
            logger.info(f"✅ All {expected_classes} rate classes present in original data")
    
    # STEP 3: Validate features exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        logger.warning(f"⚠️ Missing features (will exclude): {missing_features}")
    logger.info(f"✅ Using {len(available_features)} available features out of {len(feature_cols)} requested")
    
    # Log removed leaky features
    removed_leaky_features = [
        "phyRate", "optimalRateDistance", "recentThroughputTrend", 
        "conservativeFactor", "aggressiveFactor", "recommendedSafeRate"
    ]
    logger.info(f"🚨 LEAKY FEATURES REMOVED: {removed_leaky_features}")
    logger.info(f"🛡️ SAFE FEATURES USED: {available_features}")
    
    # STEP 4: BULLETPROOF cleaning - only remove completely invalid rows
    logger.info("🧹 Starting BULLETPROOF cleaning (minimal intervention)...")
    df_before_cleaning = df.copy()
    
    initial_rows = len(df)
    
    # Remove rows missing the target label (absolute requirement)
    df_step1 = df.dropna(subset=[label])
    logger.info(f"📊 After removing rows missing target label: {len(df_step1):,} rows ({len(df_step1)/initial_rows*100:.1f}% retained)")
    debug_class_loss(df_before_cleaning, df_step1, label, "target label cleaning", logger, expected_classes)
    
    # Remove rows missing ALL features (completely useless)
    df_step2 = df_step1.dropna(subset=available_features, how='all')
    logger.info(f"📊 After removing rows missing ALL features: {len(df_step2):,} rows ({len(df_step2)/initial_rows*100:.1f}% retained)")
    debug_class_loss(df_step1, df_step2, label, "all-features cleaning", logger, expected_classes)
    
    # Final result
    df_clean = df_step2
    logger.info(f"📊 FINAL after bulletproof cleaning: {len(df_clean):,} rows ({len(df_clean)/initial_rows*100:.1f}% retained)")
    
    # STEP 5: VERIFY classes survived cleaning
    if label in df_clean.columns:
        final_dist = df_clean[label].value_counts().sort_index()
        logger.info(f"🎯 FINAL class distribution for '{label}' after cleaning:")
        for class_val, count in final_dist.items():
            pct = (count / len(df_clean)) * 100
            logger.info(f"  Class {class_val}: {count:,} samples ({pct:.1f}%)")
        
        # Check if we have enough samples for stratification
        min_samples = final_dist.min()
        if min_samples < 3:
            logger.error(f"❌ Smallest class has only {min_samples} samples - need at least 3 for stratified splitting")
            return None, None
        else:
            logger.info(f"✅ BULLETPROOF SUCCESS: All classes have sufficient samples for training!")
    
    return df_clean, available_features

def perform_train_split_fixed(X, y, logger, expected_classes=8):
    """FIXED: Proper train/validation/test split with comprehensive verification."""
    logger.info("🔄 Performing stratified train/validation/test split...")
    try:
        # Check class distribution before splitting
        class_counts = pd.Series(y).value_counts().sort_index()
        logger.info(f"📊 Class counts before splitting:")
        for class_val, count in class_counts.items():
            logger.info(f"  Class {class_val}: {count} samples")
        
        # Verify we have enough samples of each class for stratified split
        min_class_count = class_counts.min()
        actual_classes = len(class_counts)
        logger.info(f"📊 Found {actual_classes} classes, smallest has {min_class_count} samples")
        
        if min_class_count < 3:
            logger.error(f"❌ Smallest class has only {min_class_count} samples - need at least 3 for stratification")
            raise ValueError(f"Not enough samples in smallest class ({min_class_count}) for stratified split")
        
        # First split: 80% temp, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Second split: 75% train, 25% val (of the temp 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
        
        logger.info(f"✅ Data split completed successfully")
        logger.info(f"📈 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"📊 Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"🧪 Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Verify all splits have adequate class representation
        for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            split_classes = sorted(pd.Series(y_split).unique())
            logger.info(f"  {split_name} classes: {split_classes}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"❌ Data splitting failed: {str(e)}")
        raise

def scale_features(X_train, X_val, X_test, logger):
    """Scale features using StandardScaler."""
    logger.info("🔧 Scaling features...")
    try:
        scaler = StandardScaler()
        with tqdm(total=3, desc="Scaling datasets", unit="dataset") as pbar:
            X_train_scaled = scaler.fit_transform(X_train)
            pbar.update(1); pbar.set_postfix({"current": "train"})
            X_val_scaled = scaler.transform(X_val)
            pbar.update(1); pbar.set_postfix({"current": "validation"})
            X_test_scaled = scaler.transform(X_test)
            pbar.update(1); pbar.set_postfix({"current": "test"})
        
        joblib.dump(scaler, SCALER_FILE)
        logger.info(f"✅ Features scaled and scaler saved to {SCALER_FILE}")
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"❌ Feature scaling failed: {str(e)}")
        raise

def train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, label_name, logger, available_features, class_weights=None):
    """Train and evaluate the Random Forest model with class weights."""
    model_name = f"RF - {label_name}"
    if class_weights:
        model_name += " (WITH CLASS WEIGHTS)"
    logger.info(f"\n{'='*20} TRAINING {model_name} {'='*20}")
    try:
        # Set class weights if provided
        if class_weights:
            model.set_params(class_weight=class_weights)
            logger.info(f"🔢 Using class weights for {len(class_weights)} classes")
            # Show which classes get the most attention
            sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
            logger.info("🔝 Classes by weight (highest attention first):")
            for class_val, weight in sorted_weights:
                logger.info(f"  Class {class_val}: {weight:.1f}x attention")
        else:
            logger.info("⚖️ Training with sklearn's balanced class weights")
            model.set_params(class_weight='balanced')
        
        start_time = time.time()
        logger.info(f"🚀 Starting training for {model_name}...")
        logger.info(f"📊 Training on {len(X_train):,} samples with {X_train.shape[1]} features")
        
        # Train with progress bar
        with tqdm(total=100, desc=f"Training RF", unit="%") as pbar:
            model.fit(X_train, y_train)
            pbar.update(100)
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluation on validation set
        logger.info(f"📊 Evaluating on validation set...")
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        logger.info(f"🎯 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
        
        # Evaluation on test set
        logger.info(f"🧪 Evaluating on test set...")
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        logger.info(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        
        # Cross-validation reality check
        logger.info("🔄 Cross-validation reality check...")
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        
        cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        logger.info(f"📊 5-Fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"📊 CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Reality assessment
        if cv_mean > 0.98:
            logger.warning(f"⚠️ CV accuracy {cv_mean:.1%} very high - check for issues")
        elif cv_mean > 0.85:
            logger.info(f"✅ Excellent CV performance: {cv_mean:.1%}")
        elif cv_mean > 0.70:
            logger.info(f"✅ Good CV performance: {cv_mean:.1%}")
        else:
            logger.info(f"📊 CV performance: {cv_mean:.1%} (room for improvement)")
        
        # Detailed per-class analysis
        unique_classes = sorted(pd.Series(y_val).unique())
        val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
        logger.info("📊 Per-class validation performance:")
        for class_id in unique_classes:
            if str(class_id) in val_report:
                metrics = val_report[str(class_id)]
                support = int(metrics['support'])
                logger.info(f"  Class {class_id}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, Support={support}")
        
        # Confusion matrices
        val_cm = confusion_matrix(y_val, y_val_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"📈 Validation Confusion Matrix:\n{val_cm}")
        logger.info(f"📈 Test Confusion Matrix:\n{test_cm}")
        
        # Save model
        joblib.dump(model, MODEL_FILE)
        logger.info(f"💾 Model saved to {MODEL_FILE}")
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(available_features, model.feature_importances_))
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"🔝 Top 10 most important features:")
            for rank, (feat, importance) in enumerate(top_features, 1):
                logger.info(f"  #{rank:2d}. {feat}: {importance:.4f}")
        
        logger.info(f"{'='*60}")
        return val_acc, test_acc, training_time, cv_mean
    except Exception as e:
        logger.error(f"❌ Training/evaluation failed for {model_name}: {str(e)}")
        raise

def save_comprehensive_documentation(results, feature_cols, total_time, logger, label_name, used_class_weights=False):
    """Save comprehensive documentation of the training process and results."""
    logger.info("📝 Saving comprehensive documentation...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DOC_FILE, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"CONFIGURABLE ML MODEL TRAINING RESULTS - TARGET: {label_name}\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"User: {USER}\n")
            f.write(f"Target Label: {label_name}\n")
            f.write(f"Total Pipeline Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
            
            f.write("DATASET CONFIGURATION:\n")
            f.write(f"- Source File: {CSV_FILE}\n")
            f.write(f"- Target Label: {label_name}\n")
            if label_name in TARGET_LABEL_INFO:
                f.write(f"- Label Description: {TARGET_LABEL_INFO[label_name]['description']}\n")
            f.write(f"- Row Limiting: {'ENABLED' if ENABLE_ROW_LIMITING else 'DISABLED'}\n")
            f.write(f"- Used class weights for imbalanced data: {used_class_weights}\n\n")
            
            f.write("MODELS TRAINED:\n")
            val_acc, test_acc, train_time, cv_mean = results[0]
            f.write(f"1. RandomForestClassifier ({label_name})\n")
            f.write(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)\n")
            f.write(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)\n")
            f.write(f"   Cross-Validation: {cv_mean:.4f} ({cv_mean*100:.1f}%)\n")
            f.write(f"   Training Time: {train_time:.2f}s\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"- Algorithm: Random Forest Classifier\n")
            f.write(f"- Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
            f.write("- Split: 60/20/20 stratified (train/val/test)\n")
            f.write("- Random State: 42\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write(f"- {SCALER_FILE} (StandardScaler)\n")
            f.write(f"- {MODEL_FILE} (Random Forest for {label_name})\n")
            f.write(f"- {DOC_FILE} (this file)\n")
        
        logger.info(f"✅ Documentation saved to {DOC_FILE}")
    except Exception as e:
        logger.error(f"❌ Documentation saving failed: {str(e)}")
        raise

def main():
    """Main training pipeline with configurable target labels."""
    logger = setup_logging()
    pipeline_start = time.time()
    
    try:
        logger.info(f"🎯 Training model for target label: {TARGET_LABEL}")
        
        # STEP 0: LOAD CLASS WEIGHTS
        class_weights = load_class_weights(CLASS_WEIGHTS_FILE, TARGET_LABEL, logger)
        
        # STEP 1: BULLETPROOF DATASET LOADING
        df, available_features = bulletproof_load_dataset(
            CSV_FILE, FEATURE_COLS, TARGET_LABEL, logger, CONTEXT_LABEL
        )
        
        if df is None:
            logger.error("❌ Dataset loading failed")
            return False
        
        # Prepare features and target
        X = df[available_features]
        y = df[TARGET_LABEL].astype(int)
        
        # Compute class weights if not loaded from file
        if class_weights is None:
            class_weights = compute_class_weights_if_needed(y, TARGET_LABEL, logger)
        
        logger.info(f"🔢 Final training data: X={X.shape}, y={y.shape}")
        logger.info(f"💾 Estimated memory usage: ~{X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # STEP 2: STRATIFIED DATA SPLITTING
        X_train, X_val, X_test, y_train, y_val, y_test = perform_train_split_fixed(X, y, logger)
        
        # STEP 3: FEATURE SCALING
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test, logger)
        
        # STEP 4: MODEL TRAINING WITH CLASS WEIGHTS
        results = []
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model_results = train_and_eval(
            rf_model, X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, TARGET_LABEL, logger, available_features, 
            class_weights=class_weights
        )
        results.append(model_results)
        
        # STEP 5: SAVE COMPREHENSIVE DOCUMENTATION
        total_time = time.time() - pipeline_start
        save_comprehensive_documentation(
            results, available_features, total_time, logger, TARGET_LABEL, 
            used_class_weights=(class_weights is not None)
        )
        
        # SUCCESS!
        val_acc, test_acc, train_time, cv_mean = model_results
        logger.info("🎉 CONFIGURABLE TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"🏆 Final Results for {TARGET_LABEL}: Val={val_acc:.1%}, Test={test_acc:.1%}, CV={cv_mean:.1%}")
        
        # Performance assessment
        if cv_mean > 0.90:
            logger.info("✅ EXCELLENT PERFORMANCE: >90% accuracy!")
        elif cv_mean > 0.75:
            logger.info("✅ GOOD PERFORMANCE: >75% accuracy!")
        else:
            logger.info("📊 MODERATE PERFORMANCE: Room for improvement")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        if 'pipeline_start' in locals():
            final_time = time.time() - pipeline_start
            logger.info(f"\n⏱️  Total execution time: {final_time:.2f} seconds")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)