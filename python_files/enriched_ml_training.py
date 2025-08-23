"""
Ultimate ML Model Training Pipeline (Random Forest, Large CSV-Compatible, RAM-Efficient)
Trains and outputs the rateIdx model using Phase 4 Step 3 naming/style.

Features:
- Chunked CSV loading with balanced sampling for large datasets.
- Handles all new features, context labels, and oracle label nuances.
- Step-by-step logging and documentation.
- Stratified splits and feature scaling.
- Model saving and top feature reporting.
- Output model is named step3_rf_rateIdx_model_FIXED.joblib (matching Phase 4 Step 3 convention).

Author: ahmedjk34
Date: 2025-08-22
"""

import pandas as pd
import joblib
import logging
import time
import sys
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

# ================== CONFIG ==================
CSV_FILE = "smart-v3-ml-enriched.csv"   # <--- DO NOT CHANGE
MAX_ROWS = 1_000_000                      # RAM friendly (~500MB usage)
CHUNKSIZE = 250_000
FEATURE_COLS = [
    "rateIdx", "phyRate", "lastSnr", "snrFast", "snrSlow", "shortSuccRatio",
    "medSuccRatio", "consecSuccess", "consecFailure", "severity", "confidence",
    "T1", "T2", "T3", "decisionReason", "packetSuccess", "offeredLoad", "queueLen",
    "retryCount", "channelWidth", "mobilityMetric", "snrVariance"
]
RATEIDX_LABEL = "rateIdx"             # <--- Main label for rateIdx model
CONTEXT_LABEL = "network_context"
USER = "ahmedjk34"
SCALER_FILE = "step3_scaler_FIXED.joblib"  # <--- DO NOT CHANGE
MODEL_FILE = "step3_rf_rateIdx_model_FIXED.joblib"  # <--- Matches output style of Phase 4 Step 3
DOC_FILE = "step3_ultimate_models_FIXED_versions.txt" # <--- Matches output style
# ============================================

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ultimate_ml_training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("ULTIMATE ML MODEL TRAINING PIPELINE STARTED (RF ONLY, RATEIDX MODEL, PATCHED FOR LARGE FILES)")
    logger.info("="*60)
    return logger

def load_balanced_dataset(filepath, feature_cols, label, logger, context_label):
    logger.info(f"📂 Loading dataset from {filepath} with chunked sampling...")
    sampled_chunks = []
    total_rows = 0
    chunk_no = 1
    for chunk in pd.read_csv(filepath, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        logger.info(f"--- Chunk {chunk_no}, {len(chunk)} rows ---")
        label_counts = chunk[label].value_counts().sort_index()
        label_pct = (label_counts / len(chunk) * 100).round(2)
        logger.info(f"Label ({label}) distribution:\n{label_counts}\nPercentages:\n{label_pct}")
        if context_label in chunk.columns:
            context_counts = chunk[context_label].value_counts().sort_index()
            logger.info(f"Context label distribution:\n{context_counts}")
        frac = MAX_ROWS / (total_rows + 1e-6)
        frac = min(frac, 1.0)
        if frac <= 0: break
        sampled_chunk = chunk.sample(frac=frac, random_state=42)
        sampled_chunks.append(sampled_chunk)
        chunk_no += 1
    df = pd.concat(sampled_chunks, ignore_index=True)
    logger.info(f"✅ Loaded sampled dataset: {df.shape[0]} rows (from ~{total_rows})")
    logger.info(f"🎯 Final label ({label}) distribution:\n{df[label].value_counts().sort_index()}")
    logger.info(f"🎯 Final label percentages:\n{(df[label].value_counts() / len(df) * 100).round(2)}")
    if context_label in df.columns:
        logger.info(f"🎯 Final context distribution:\n{df[context_label].value_counts().sort_index()}")
    print("Final sampled label balance:")
    print(df[label].value_counts(normalize=True))
    if context_label in df.columns:
        print("Final sampled context balance:")
        print(df[context_label].value_counts(normalize=True))
    return df

def perform_train_split(X, y, logger):
    logger.info("🔄 Performing stratified train/validation/test split...")
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42)
        logger.info(f"✅ Data split completed")
        logger.info(f"📈 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"📊 Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"🧪 Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        print("Train label counts:", pd.Series(y_train).value_counts())
        print("Val label counts:", pd.Series(y_val).value_counts())
        print("Test label counts:", pd.Series(y_test).value_counts())
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"❌ Data splitting failed: {str(e)}")
        raise

def scale_features(X_train, X_val, X_test, logger):
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
        print("Example training features before scaling:", X_train.iloc[0].values)
        print("Example training features after scaling:", X_train_scaled[0])
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"❌ Feature scaling failed: {str(e)}")
        raise

def train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, label_name, logger):
    model_name = f"RF - {label_name}"
    logger.info(f"\n{'='*20} TRAINING {model_name} {'='*20}")
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting training for {model_name}...")
        with tqdm(total=100, desc=f"Training RF", unit="%") as pbar:
            model.fit(X_train, y_train)
            pbar.update(100)
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        print("Feature columns used for training:", X_train.shape[1])
        logger.info(f"📊 Evaluating on validation set...")
        y_val_pred = model.predict(X_val)
        print("Predicted label distribution on validation set:", pd.Series(y_val_pred).value_counts())
        val_acc = accuracy_score(y_val, y_val_pred)
        logger.info(f"🎯 {model_name} Validation Accuracy: {val_acc:.4f}")
        logger.info(f"🧪 Evaluating on test set...")
        y_test_pred = model.predict(X_test)
        print("Test set label distribution (actual):", pd.Series(y_test).value_counts())
        print("Predicted label distribution on test set:", pd.Series(y_test_pred).value_counts())
        test_acc = accuracy_score(y_test, y_test_pred)
        val_cm = confusion_matrix(y_val, y_val_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"📈 Validation Confusion Matrix:\n{val_cm}")
        logger.info(f"📈 Test Confusion Matrix:\n{test_cm}")
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        joblib.dump(model, MODEL_FILE)
        logger.info(f"💾 Model saved to {MODEL_FILE}")
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(FEATURE_COLS, model.feature_importances_))
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:8]
            logger.info(f"🔝 Top 8 most important features:")
            for feat, importance in top_features:
                logger.info(f"  {feat}: {importance:.4f}")
        logger.info(f"{'='*60}")
        return val_acc, test_acc, training_time
    except Exception as e:
        logger.error(f"❌ Training/evaluation failed for {model_name}: {str(e)}")
        raise

def save_comprehensive_documentation(results, feature_cols, total_time, logger, label_name):
    logger.info("📝 Saving comprehensive documentation...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DOC_FILE, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("ULTIMATE ML MODEL TRAINING PIPELINE RESULTS (RF ONLY, RATEIDX MODEL)\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"User: {USER}\n")
            f.write(f"Total Pipeline Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
            f.write("DATASET USED:\n")
            f.write(f"- {CSV_FILE}\n")
            f.write("- Dataset includes context, oracle labels, synthetic edge cases, and new features\n")
            f.write("MODELS TRAINED:\n")
            val_acc, test_acc, train_time = results[0]
            f.write(f"1. RandomForestClassifier ({label_name})\n")
            f.write(f"   Validation Accuracy: {val_acc:.4f}\n")
            f.write(f"   Test Accuracy: {test_acc:.4f}\n")
            f.write(f"   Training Time: {train_time:.2f}s\n\n")
            f.write("CONFIGURATION:\n")
            f.write(f"- Scaler: StandardScaler\n")
            f.write(f"- Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
            f.write("- Split: 80/10/10 stratified\n")
            f.write("- Random State: 42\n\n")
            f.write("FILES GENERATED:\n")
            f.write(f"- {SCALER_FILE} (StandardScaler)\n")
            f.write(f"- {MODEL_FILE} (Random Forest for {label_name})\n")
            f.write(f"- {DOC_FILE} (this file)\n\n")
            f.write("NEXT STEPS:\n")
            f.write("- Test models in simulation\n")
            f.write("- Integrate with ns-3 or deployment pipeline\n")
        logger.info(f"✅ Documentation saved to {DOC_FILE}")
    except Exception as e:
        logger.error(f"❌ Documentation saving failed: {str(e)}")
        raise

def main():
    logger = setup_logging()
    pipeline_start = time.time()
    try:
        feature_cols = FEATURE_COLS
        label_name = RATEIDX_LABEL
        # STEP 1: LOAD DATA
        df = load_balanced_dataset(CSV_FILE, feature_cols, label_name, logger, CONTEXT_LABEL)
        df = df.dropna(subset=feature_cols + [label_name])
        X = df[feature_cols]
        y = df[label_name].astype(int)
        # STEP 2: SPLIT DATA
        X_train, X_val, X_test, y_train, y_val, y_test = perform_train_split(X, y, logger)
        # STEP 3: SCALE FEATURES
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test, logger)
        # STEP 4: TRAIN RF
        results = []
        rf_model = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1)
        model_results = train_and_eval(rf_model, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, label_name, logger)
        results.append(model_results)
        # STEP 5: SAVE DOCS (style and output name match Phase 4 Step 3)
        total_time = time.time() - pipeline_start
        save_comprehensive_documentation(results, feature_cols, total_time, logger, label_name)
        return True
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False
    finally:
        if 'pipeline_start' in locals():
            final_time = time.time() - pipeline_start
            logger.info(f"\n⏱️  Total execution time: {final_time:.2f} seconds")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)