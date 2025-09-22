"""
Ultimate ML Model Training Pipeline with Class Weights (Random Forest, Large CSV-Compatible, RAM-Efficient)
Trains and outputs the rateIdx model using Phase 4 Step 3 naming/style with CLASS WEIGHTS instead of downsampling.

Features:
- BULLETPROOF class preservation - guarantees all 8 classes survive processing
- Optional chunked CSV loading and row limiting (configurable)
- Handles all new features, context labels, and oracle label nuances
- USES CLASS WEIGHTS for imbalanced data instead of aggressive downsampling
- Step-by-step logging and documentation
- Stratified splits and feature scaling
- Model saving and top feature reporting

Author: ahmedjk34
Date: 2025-09-22
BULLETPROOF VERSION: Absolutely preserves all classes, configurable sampling, proper debugging
"""

import pandas as pd
import joblib
import logging
import time
import sys
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)

# ================== CONFIGURABLE SETTINGS ==================
CSV_FILE = "smart-v3-ml-enriched.csv"   # <--- DO NOT CHANGE

# ROW LIMITING CONTROLS (Set ENABLE_ROW_LIMITING=False to use full dataset)
ENABLE_ROW_LIMITING = False              # <--- Set to True to enable sampling/chunking
MAX_ROWS = 500_000                       # Only used if ENABLE_ROW_LIMITING=True
CHUNKSIZE = 250_000                      # Only used if ENABLE_ROW_LIMITING=True

# CORRECTED FEATURE COLUMNS (removed rateIdx from features since it's the target)
FEATURE_COLS = [
    "phyRate", "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "shortSuccRatio", "medSuccRatio", 
    "consecSuccess", "consecFailure", "recentThroughputTrend", "packetLossRate",
    "retrySuccessRatio", "recentRateChanges", "timeSinceLastRateChange", 
    "rateStabilityScore", "optimalRateDistance", "aggressiveFactor", 
    "conservativeFactor", "recommendedSafeRate", "severity", "confidence",
    "T1", "T2", "T3", "decisionReason", "packetSuccess", "offeredLoad", 
    "queueLen", "retryCount", "channelWidth", "mobilityMetric", "snrVariance"
]

RATEIDX_LABEL = "rateIdx"             # <--- Main label for rateIdx model
CONTEXT_LABEL = "network_context"
USER = "ahmedjk34"
SCALER_FILE = "step3_scaler_FIXED.joblib"  # <--- DO NOT CHANGE
MODEL_FILE = "step3_rf_rateIdx_model_FIXED.joblib"  # <--- Matches output style of Phase 4 Step 3
DOC_FILE = "step3_ultimate_models_FIXED_versions.txt" # <--- Matches output style
CLASS_WEIGHTS_FILE = "python_files/model_artifacts/class_weights.json"  # <--- Load class weights
# ============================================

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bulletproof_ml_training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("BULLETPROOF ML TRAINING PIPELINE - GUARANTEED CLASS PRESERVATION")
    logger.info("="*60)
    logger.info(f"🔧 Row limiting: {'ENABLED' if ENABLE_ROW_LIMITING else 'DISABLED'}")
    if ENABLE_ROW_LIMITING:
        logger.info(f"📊 Max rows: {MAX_ROWS:,}, Chunk size: {CHUNKSIZE:,}")
    else:
        logger.info("📊 Using FULL dataset (no row limits)")
    return logger

def load_class_weights(filepath, target_label, logger):
    """Load pre-computed class weights for handling imbalanced data."""
    logger.info(f"📊 Loading class weights from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            all_weights = json.load(f)
        
        if target_label not in all_weights:
            logger.warning(f"⚠️ No class weights found for {target_label}, using balanced weights")
            return None
        
        weights = all_weights[target_label]
        # Convert string keys to integers for rateIdx
        if target_label == 'rateIdx':
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
        logger.info("Will train without class weights (balanced approach)")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading class weights: {e}")
        return None

def debug_class_loss(df_before, df_after, label, step_name, logger):
    """Debug function to track exactly what happens to each class during processing."""
    before_dist = df_before[label].value_counts().sort_index()
    after_dist = df_after[label].value_counts().sort_index()
    
    logger.info(f"🔍 DEBUG: Class changes during {step_name}:")
    for class_val in range(8):  # Expected classes 0-7
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
    No aggressive cleaning that could accidentally drop rare classes.
    """
    logger.info(f"📂 Loading dataset with BULLETPROOF class preservation...")
    
    # STEP 1: Load complete dataset
    if ENABLE_ROW_LIMITING:
        logger.info(f"🔧 Row limiting ENABLED - will process in chunks of {CHUNKSIZE:,}")
        # Load in chunks if row limiting is enabled
        chunk_list = []
        total_rows_seen = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=CHUNKSIZE, low_memory=False)):
            total_rows_seen += len(chunk)
            chunk_list.append(chunk)
            logger.info(f"📥 Loaded chunk {chunk_num + 1}: {len(chunk):,} rows (total: {total_rows_seen:,})")
            
            # Stop if we have enough chunks to reach MAX_ROWS
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
    
    # STEP 2: Show ORIGINAL class distribution
    if label in df.columns:
        original_dist = df[label].value_counts().sort_index()
        logger.info(f"🎯 ORIGINAL class distribution:")
        for class_val, count in original_dist.items():
            pct = (count / len(df)) * 100
            logger.info(f"  Class {class_val}: {count:,} samples ({pct:.1f}%)")
        
        # Verify we have all expected classes
        expected_classes = set(range(8))  # 0-7
        original_classes = set(original_dist.index)
        missing_from_start = expected_classes - original_classes
        
        if missing_from_start:
            logger.error(f"❌ MISSING CLASSES IN ORIGINAL DATA: {missing_from_start}")
            logger.error("Cannot proceed - source data is missing rate classes!")
            return None, None
        else:
            logger.info("✅ All 8 rate classes present in original data")
    
    # STEP 3: Validate features exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        logger.warning(f"⚠️ Missing features (will exclude): {missing_features}")
    logger.info(f"✅ Using {len(available_features)} available features out of {len(feature_cols)} requested")
    
    # STEP 4: BULLETPROOF cleaning - only remove completely invalid rows
    logger.info("🧹 Starting BULLETPROOF cleaning (minimal intervention)...")
    df_before_cleaning = df.copy()
    
    # Only remove rows that are completely unusable
    initial_rows = len(df)
    
    # Remove rows missing the target label (absolute requirement)
    df_step1 = df.dropna(subset=[label])
    logger.info(f"📊 After removing rows missing target label: {len(df_step1):,} rows ({len(df_step1)/initial_rows*100:.1f}% retained)")
    debug_class_loss(df_before_cleaning, df_step1, label, "target label cleaning", logger)
    
    # Remove rows missing ALL features (completely useless)
    df_step2 = df_step1.dropna(subset=available_features, how='all')
    logger.info(f"📊 After removing rows missing ALL features: {len(df_step2):,} rows ({len(df_step2)/initial_rows*100:.1f}% retained)")
    debug_class_loss(df_step1, df_step2, label, "all-features cleaning", logger)
    
    # Final result
    df_clean = df_step2
    logger.info(f"📊 FINAL after bulletproof cleaning: {len(df_clean):,} rows ({len(df_clean)/initial_rows*100:.1f}% retained)")
    
    # STEP 5: VERIFY all classes survived cleaning
    if label in df_clean.columns:
        final_dist = df_clean[label].value_counts().sort_index()
        logger.info(f"🎯 FINAL class distribution after cleaning:")
        for class_val, count in final_dist.items():
            pct = (count / len(df_clean)) * 100
            logger.info(f"  Class {class_val}: {count:,} samples ({pct:.1f}%)")
        
        # Critical verification
        final_classes = set(final_dist.index)
        still_missing = expected_classes - final_classes
        
        if still_missing:
            logger.error(f"❌ CLASSES LOST DURING CLEANING: {still_missing}")
            logger.error("🔍 Investigating what happened to missing classes...")
            
            # Debug: show what happened to each missing class
            for missing_class in still_missing:
                original_count = original_dist.get(missing_class, 0)
                logger.error(f"  Class {missing_class}: Started with {original_count} samples, now has 0")
                
                # Check if they were lost during each cleaning step
                step1_count = df_step1[df_step1[label] == missing_class].shape[0] if missing_class in df_step1[label].values else 0
                step2_count = df_step2[df_step2[label] == missing_class].shape[0] if missing_class in df_step2[label].values else 0
                
                logger.error(f"    After target cleaning: {step1_count}")
                logger.error(f"    After feature cleaning: {step2_count}")
            
            logger.error("❌ CANNOT PROCEED - MISSING CLASSES WILL CAUSE STRATIFIED SPLIT TO FAIL")
            return None, None
        else:
            logger.info("✅ BULLETPROOF SUCCESS: All 8 rate classes survived cleaning!")
    
    return df_clean, available_features

def perform_train_split_fixed(X, y, logger):
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
        logger.info(f"📊 Smallest class has {min_class_count} samples")
        
        if min_class_count < 3:
            logger.error(f"❌ Smallest class has only {min_class_count} samples - need at least 3 for stratification")
            logger.error("💡 Suggestion: Disable row limiting or increase MAX_ROWS to preserve more rare class samples")
            raise ValueError(f"Not enough samples in smallest class ({min_class_count}) for stratified split")
        
        # First split: 80% temp, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Second split: 75% train, 25% val (of the temp 80%)
        # This gives us: 60% train, 20% val, 20% test
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
        
        logger.info(f"✅ Data split completed successfully")
        logger.info(f"📈 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"📊 Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"🧪 Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Verify all splits have all classes
        all_splits_valid = True
        for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            split_classes = sorted(pd.Series(y_split).unique())
            logger.info(f"  {split_name} classes: {split_classes}")
            if len(split_classes) != 8:
                logger.warning(f"⚠️ {split_name} split has {len(split_classes)} classes, expected 8")
                all_splits_valid = False
        
        if not all_splits_valid:
            logger.warning("⚠️ Some splits are missing classes, but proceeding with available classes")
        
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
            logger.info("⚖️ Training without specific class weights (sklearn balanced)")
        
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
        
        # Detailed per-class analysis
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        logger.info("📊 Per-class validation performance:")
        for class_id in range(8):
            if str(class_id) in val_report:
                metrics = val_report[str(class_id)]
                support = int(metrics['support'])
                logger.info(f"  Class {class_id}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, Support={support}")
            else:
                logger.warning(f"  Class {class_id}: NO SAMPLES in validation set")
        
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
        return val_acc, test_acc, training_time
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
            f.write("BULLETPROOF ML MODEL TRAINING PIPELINE RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"User: {USER}\n")
            f.write(f"Total Pipeline Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
            
            f.write("DATASET CONFIGURATION:\n")
            f.write(f"- Source File: {CSV_FILE}\n")
            f.write(f"- Row Limiting: {'ENABLED' if ENABLE_ROW_LIMITING else 'DISABLED'}\n")
            if ENABLE_ROW_LIMITING:
                f.write(f"- Max Rows: {MAX_ROWS:,}\n")
                f.write(f"- Chunk Size: {CHUNKSIZE:,}\n")
            else:
                f.write("- Using FULL dataset (no sampling)\n")
            f.write("- Dataset includes context, oracle labels, synthetic edge cases, and new features\n")
            f.write("- BULLETPROOF class preservation - all 8 rate classes guaranteed\n")
            f.write(f"- Used class weights for imbalanced data: {used_class_weights}\n\n")
            
            f.write("MODELS TRAINED:\n")
            val_acc, test_acc, train_time = results[0]
            f.write(f"1. RandomForestClassifier ({label_name})\n")
            f.write(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)\n")
            f.write(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)\n")
            f.write(f"   Training Time: {train_time:.2f}s\n")
            f.write(f"   Used Class Weights: {used_class_weights}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"- Algorithm: Random Forest Classifier\n")
            f.write(f"- Estimators: 100 trees\n")
            f.write(f"- Max Depth: 15\n")
            f.write(f"- Scaler: StandardScaler\n")
            f.write(f"- Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
            f.write("- Split: 60/20/20 stratified (train/val/test)\n")
            f.write("- Random State: 42\n")
            f.write("- Imbalance Handling: Class Weights (preserves realistic distributions)\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write(f"- {SCALER_FILE} (StandardScaler)\n")
            f.write(f"- {MODEL_FILE} (Random Forest for {label_name})\n")
            f.write(f"- {DOC_FILE} (this file)\n\n")
            
            f.write("BULLETPROOF FEATURES:\n")
            f.write("- Guaranteed class preservation during all processing steps\n")
            f.write("- Configurable row limiting (currently disabled)\n")
            f.write("- Comprehensive debugging and class tracking\n")
            f.write("- Robust error handling and validation\n")
            f.write("- Detailed per-class performance analysis\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("- Test model performance in WiFi simulation environment\n")
            f.write("- Integrate with ns-3 rate adaptation algorithms\n")
            f.write("- Deploy to real WiFi hardware for validation\n")
            f.write("- Monitor model performance in production scenarios\n")
        
        logger.info(f"✅ Documentation saved to {DOC_FILE}")
    except Exception as e:
        logger.error(f"❌ Documentation saving failed: {str(e)}")
        raise

def main():
    """Main training pipeline with bulletproof class preservation."""
    logger = setup_logging()
    pipeline_start = time.time()
    
    try:
        # STEP 0: LOAD CLASS WEIGHTS
        class_weights = load_class_weights(CLASS_WEIGHTS_FILE, RATEIDX_LABEL, logger)
        
        # STEP 1: BULLETPROOF DATASET LOADING
        df, available_features = bulletproof_load_dataset(
            CSV_FILE, FEATURE_COLS, RATEIDX_LABEL, logger, CONTEXT_LABEL
        )
        
        if df is None:
            logger.error("❌ Dataset loading failed - missing classes")
            return False
        
        # Prepare features and target
        X = df[available_features]
        y = df[RATEIDX_LABEL].astype(int)
        
        logger.info(f"🔢 Final training data: X={X.shape}, y={y.shape}")
        logger.info(f"💾 Estimated memory usage: ~{X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # STEP 2: STRATIFIED DATA SPLITTING
        X_train, X_val, X_test, y_train, y_val, y_test = perform_train_split_fixed(X, y, logger)
        
        # STEP 3: FEATURE SCALING
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test, logger)
        
        # STEP 4: MODEL TRAINING WITH CLASS WEIGHTS
        results = []
        rf_model = RandomForestClassifier(
            n_estimators=100,      # Good balance of performance vs training time
            max_depth=15,          # Prevent overfitting while allowing complexity
            random_state=42,       # Reproducible results
            n_jobs=-1,            # Use all CPU cores
            verbose=0             # Quiet training (we have our own progress bar)
        )
        
        model_results = train_and_eval(
            rf_model, X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, RATEIDX_LABEL, logger, available_features, 
            class_weights=class_weights
        )
        results.append(model_results)
        
        # STEP 5: SAVE COMPREHENSIVE DOCUMENTATION
        total_time = time.time() - pipeline_start
        save_comprehensive_documentation(
            results, available_features, total_time, logger, RATEIDX_LABEL, 
            used_class_weights=(class_weights is not None)
        )
        
        # SUCCESS!
        val_acc, test_acc, train_time = model_results
        logger.info("🎉 BULLETPROOF TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"🏆 Final Results: Val={val_acc:.1%}, Test={test_acc:.1%}, Time={train_time:.1f}s")
        
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