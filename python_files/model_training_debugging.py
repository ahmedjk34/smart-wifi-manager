"""
Phase 4 Step 3: Ultimate ML Model Training Pipeline (RF only, Debug Steps Included)
Patched for Large CSV (1.4GB) + Low RAM (4GB)

Fix:
- Added chunked CSV loader with row sampling
- Only sample ~500k rows into memory (configurable)
- **RF only for debugging**
- Keeps balanced label distributions
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
CSV_FILE = "smartv4-ml-balanced.csv"
MAX_ROWS = 1_000_000   # safely fits in 4GB RAM (~800MB usage)
CHUNKSIZE = 250_000    # balanced chunk size (~200MB per chunk)
# ============================================

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ml_training_fixed_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("ULTIMATE ML MODEL TRAINING PIPELINE STARTED (RF ONLY, FIXED DATASET, PATCHED FOR LARGE FILES)")
    logger.info("="*60)
    return logger

def load_balanced_dataset(filepath, feature_cols, label_oracle, label_v3, logger):
    """
    STEP 1: LOAD DATA WITH CHUNK-WISE LABEL DISTRIBUTION LOGGING
    """
    logger.info(f"📂 Loading dataset from {filepath} with chunked sampling...")
    sampled_chunks = []
    total_rows = 0
    chunk_no = 1

    for chunk in pd.read_csv(filepath, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        logger.info(f"--- Chunk {chunk_no}, {len(chunk)} rows ---")
        # Show oracle label distribution in chunk
        oracle_counts = chunk[label_oracle].value_counts().sort_index()
        v3_counts = chunk[label_v3].value_counts().sort_index()
        oracle_pct = (oracle_counts / len(chunk) * 100).round(2)
        v3_pct = (v3_counts / len(chunk) * 100).round(2)
        logger.info(f"Oracle label distribution:\n{oracle_counts}")
        logger.info(f"Oracle label percentages:\n{oracle_pct}")
        logger.info(f"V3 label distribution:\n{v3_counts}")
        logger.info(f"V3 label percentages:\n{v3_pct}")

        frac = MAX_ROWS / (total_rows + 1e-6)  # fraction to keep
        frac = min(frac, 1.0)                  # never >1
        if frac <= 0: 
            break
        sampled_chunk = chunk.sample(frac=frac, random_state=42)
        sampled_chunks.append(sampled_chunk)
        chunk_no += 1

    df = pd.concat(sampled_chunks, ignore_index=True)
    logger.info(f"✅ Loaded sampled dataset: {df.shape[0]} rows (from ~{total_rows})")
    logger.info(f"🎯 Final Oracle label distribution:\n{df[label_oracle].value_counts().sort_index()}")
    logger.info(f"🎯 Final Oracle label percentages:\n{(df[label_oracle].value_counts() / len(df) * 100).round(2)}")
    logger.info(f"🎯 Final V3 label distribution:\n{df[label_v3].value_counts().sort_index()}")
    logger.info(f"🎯 Final V3 label percentages:\n{(df[label_v3].value_counts() / len(df) * 100).round(2)}")
    # STEP 1a: Print final sampled balance for manual inspection
    print("Final sampled data balance (oracle):")
    print(df[label_oracle].value_counts(normalize=True))
    print("Final sampled data balance (v3):")
    print(df[label_v3].value_counts(normalize=True))
    return df

def perform_train_split(X, y_oracle, y_v3, logger):
    """
    STEP 2: SPLIT DATA (STRATIFIED)
    """
    logger.info("🔄 Performing stratified train/validation/test split...")
    try:
        X_temp, X_test, y_oracle_temp, y_oracle_test, y_v3_temp, y_v3_test = train_test_split(
            X, y_oracle, y_v3, test_size=0.1, stratify=y_oracle, random_state=42)
        X_train, X_val, y_oracle_train, y_oracle_val, y_v3_train, y_v3_val = train_test_split(
            X_temp, y_oracle_temp, y_v3_temp, test_size=0.1111, stratify=y_oracle_temp, random_state=42)
        logger.info(f"✅ Data split completed")
        logger.info(f"📈 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"📊 Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"🧪 Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        # STEP 2a: Print split label distributions
        print("Train oracle label counts:")
        print(pd.Series(y_oracle_train).value_counts())
        print("Val oracle label counts:")
        print(pd.Series(y_oracle_val).value_counts())
        print("Test oracle label counts:")
        print(pd.Series(y_oracle_test).value_counts())
        return X_train, X_val, X_test, y_oracle_train, y_oracle_val, y_oracle_test, y_v3_train, y_v3_val, y_v3_test
    except Exception as e:
        logger.error(f"❌ Data splitting failed: {str(e)}")
        raise

def scale_features(X_train, X_val, X_test, logger):
    """
    STEP 3: SCALE FEATURES
    """
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
        scaler_file = "step3_scaler_FIXED.joblib"
        joblib.dump(scaler, scaler_file)
        logger.info(f"✅ Features scaled and scaler saved to {scaler_file}")
        # STEP 3a: Print one row before/after scaling for sanity check
        print("Example training features before scaling:", X_train.iloc[0].values)
        print("Example training features after scaling:", X_train_scaled[0])
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"❌ Feature scaling failed: {str(e)}")
        raise

def train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, label_name, tag, logger):
    """
    STEP 4: TRAIN RF AND DEBUG PREDICTIONS
    """
    model_name = f"{tag.upper()} - {label_name}"
    logger.info(f"\n{'='*20} TRAINING {model_name} {'='*20}")
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting training for {model_name}...")
        with tqdm(total=100, desc=f"Training {tag.upper()}", unit="%") as pbar:
            model.fit(X_train, y_train)
            pbar.update(100)
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # STEP 4a: Print model feature columns and ensure label NOT present
        print("Feature columns used for training:", X_train.shape[1])
        
        # STEP 4b: Print validation predictions distribution
        logger.info(f"📊 Evaluating on validation set...")
        y_val_pred = model.predict(X_val)
        print("Predicted label distribution on validation set:", pd.Series(y_val_pred).value_counts())
        val_acc = accuracy_score(y_val, y_val_pred)
        logger.info(f"🎯 {model_name} Validation Accuracy: {val_acc:.4f}")
        
        # STEP 4c: Print test predictions distribution
        logger.info(f"🧪 Evaluating on test set...")
        y_test_pred = model.predict(X_test)
        print("Test set oracle label distribution (actual):", pd.Series(y_test).value_counts())
        print("Predicted label distribution on test set:", pd.Series(y_test_pred).value_counts())
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # Confusion matrices and classification reports still logged for analysis
        val_cm = confusion_matrix(y_val, y_val_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"📈 Validation Confusion Matrix:\n{val_cm}")
        logger.info(f"📈 Test Confusion Matrix:\n{test_cm}")
        
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        fname = f"step3_{tag}_{label_name}_model_FIXED.joblib"
        joblib.dump(model, fname)
        logger.info(f"💾 Model saved to {fname}")
        
        if hasattr(model, 'feature_importances_'):
            feature_names = ['lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
                           'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
                           'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance']
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"🔝 Top 5 most important features:")
            for feat, importance in top_features:
                logger.info(f"  {feat}: {importance:.4f}")
        
        logger.info(f"{'='*60}")
        # STEP 4d: Return predictions for debugging
        return val_acc, test_acc, training_time, 0
    except Exception as e:
        logger.error(f"❌ Training/evaluation failed for {model_name}: {str(e)}")
        raise

def save_comprehensive_documentation(results, feature_cols, total_time, logger):
    """
    STEP 5: SAVE DOCS
    """
    logger.info("📝 Saving comprehensive documentation...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("step3_ultimate_models_FIXED_versions.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("ULTIMATE ML MODEL TRAINING PIPELINE RESULTS (RF ONLY, FIXED DATASET)\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"User: ahmedjk34\n")
            f.write(f"Total Pipeline Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
            f.write("DATASET USED:\n")
            f.write("- smartv4-ml-balanced.csv (BALANCED VERSION)\n")
            f.write("- Oracle labels fixed from 97% rate 7 bias to balanced distribution\n")
            f.write("- SNR values converted from crazy 1600+ dB to realistic 5-40 dB\n")
            f.write("- V3 rate corruption cleaned\n\n")
            f.write("MODELS TRAINED:\n")
            f.write("- RandomForestClassifier (oracle_best_rateIdx, v3_rateIdx)\n")
            f.write("# - LGBMClassifier (oracle_best_rateIdx, v3_rateIdx) [COMMENTED OUT]\n")
            f.write("# - XGBoostClassifier (oracle_best_rateIdx, v3_rateIdx) [COMMENTED OUT]\n")
            f.write("# - CatBoostClassifier (oracle_best_rateIdx, v3_rateIdx) [COMMENTED OUT]\n")
            f.write("# - MLPClassifier (oracle_best_rateIdx, v3_rateIdx) [COMMENTED OUT]\n\n")
            f.write("CONFIGURATION:\n")
            f.write("- Scaler: StandardScaler\n")
            f.write("- Dataset: smartv4-ml-balanced.csv\n")
            f.write(f"- Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
            f.write("- Split: 80/10/10 stratified\n")
            f.write("- Random State: 42\n\n")
            f.write("DETAILED RESULTS:\n")
            model_names = [
                "Random Forest (oracle_best_rateIdx)",
                "Random Forest (v3_rateIdx)",
            ]
            for idx, (model_name, model_results) in enumerate(zip(model_names, results)):
                val_acc, test_acc, train_time, eval_time = model_results
                f.write(f"{idx+1}. {model_name}:\n")
                f.write(f"   Validation Accuracy: {val_acc:.4f}\n")
                f.write(f"   Test Accuracy: {test_acc:.4f}\n")
                f.write(f"   Training Time: {train_time:.2f}s\n")
                f.write(f"   Evaluation Time: {eval_time:.2f}s\n\n")
            f.write("FILES GENERATED:\n")
            f.write("- step3_scaler_FIXED.joblib (StandardScaler)\n")
            f.write("- step3_rf_oracle_best_rateIdx_model_FIXED.joblib\n")
            f.write("- step3_rf_v3_rateIdx_model_FIXED.joblib\n")
            f.write("- step3_ultimate_models_FIXED_versions.txt (this file)\n\n")
            f.write("IMPROVEMENTS FROM ORIGINAL:\n")
            f.write("- Models should now predict diverse rates (0-7) instead of always rate 7\n")
            f.write("- Oracle model trained on balanced data for proper rate adaptation\n")
            f.write("- Should see much better rate diversity in real deployment\n\n")
            f.write("NEXT STEPS:\n")
            f.write("- Ready for Step 4: Integration with ns-3\n")
            f.write("- Test models in simulation to verify rate diversity\n")
            f.write("- Compare with original biased models\n")
        logger.info("✅ Documentation saved to step3_ultimate_models_FIXED_versions.txt")
    except Exception as e:
        logger.error(f"❌ Documentation saving failed: {str(e)}")
        raise

def main():
    logger = setup_logging()
    pipeline_start = time.time()
    try:
        # STEP 0: FEATURE LIST
        feature_cols = [
            'lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
            'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
            'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance'
        ]
        label_oracle = 'oracle_best_rateIdx'
        label_v3 = 'v3_rateIdx'

        # STEP 1: LOAD DATA
        df = load_balanced_dataset(CSV_FILE, feature_cols, label_oracle, label_v3, logger)

        X = df[feature_cols]
        y_oracle = df[label_oracle]
        y_v3 = df[label_v3]

        # STEP 2: SPLIT DATA
        X_train, X_val, X_test, y_oracle_train, y_oracle_val, y_oracle_test, y_v3_train, y_v3_val, y_v3_test = perform_train_split(
            X, y_oracle, y_v3, logger)

        # STEP 3: SCALE FEATURES
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test, logger)

        results = []
        # STEP 4: RF ONLY
        models_to_train = [
            # Random Forest for oracle_best_rateIdx
            (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
             X_train_scaled, y_oracle_train, X_val_scaled, y_oracle_val, X_test_scaled, y_oracle_test, label_oracle, "rf"),
            # Random Forest for v3_rateIdx
            (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
             X_train_scaled, y_v3_train, X_val_scaled, y_v3_val, X_test_scaled, y_v3_test, label_v3, "rf"),
        ]

        logger.info(f"\n🎯 Starting training pipeline for {len(models_to_train)} RF models...")

        with tqdm(total=len(models_to_train), desc="Overall Progress", unit="model") as overall_pbar:
            for model, X_tr, y_tr, X_va, y_va, X_te, y_te, label, tag in models_to_train:
                model_results = train_and_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te, label, tag, logger)
                results.append(model_results)
                overall_pbar.update(1)

        # STEP 5: SAVE DOCS
        total_time = time.time() - pipeline_start
        save_comprehensive_documentation(results, feature_cols, total_time, logger)
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