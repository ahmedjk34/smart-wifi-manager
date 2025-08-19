"""
Phase 4 Step 3: Ultimate ML Model Training Pipeline (with all benchmark models)

Models:
- Random Forest (oracle_best_rateIdx, v3_rateIdx)
- LightGBM (oracle_best_rateIdx, v3_rateIdx)
- XGBoost (oracle_best_rateIdx, v3_rateIdx)
- CatBoost (oracle_best_rateIdx, v3_rateIdx)
- MLP Neural Net (oracle_best_rateIdx, v3_rateIdx)

Author: github.com/ahmedjk34
Enhanced with logging, progress tracking, and error handling
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ml_training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("ULTIMATE ML MODEL TRAINING PIPELINE STARTED")
    logger.info("="*60)
    return logger

def validate_data(df, feature_cols, label_oracle, label_v3, logger):
    logger.info("🔍 Validating input data...")
    try:
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features: raise ValueError(f"Missing feature columns: {missing_features}")
        missing_labels = []
        if label_oracle not in df.columns: missing_labels.append(label_oracle)
        if label_v3 not in df.columns: missing_labels.append(label_v3)
        if missing_labels: raise ValueError(f"Missing label columns: {missing_labels}")
        missing_counts = df[feature_cols + [label_oracle, label_v3]].isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        logger.info(f"✅ Data validation passed")
        logger.info(f"📊 Dataset shape: {df.shape}")
        logger.info(f"🎯 Oracle label distribution:\n{df[label_oracle].value_counts().sort_index()}")
        logger.info(f"🎯 V3 label distribution:\n{df[label_v3].value_counts().sort_index()}")
        return True
    except Exception as e:
        logger.error(f"❌ Data validation failed: {str(e)}")
        raise

def perform_train_split(X, y_oracle, y_v3, logger):
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
        return X_train, X_val, X_test, y_oracle_train, y_oracle_val, y_oracle_test, y_v3_train, y_v3_val, y_v3_test
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
        scaler_file = "step3_scaler.joblib"
        joblib.dump(scaler, scaler_file)
        logger.info(f"✅ Features scaled and scaler saved to {scaler_file}")
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"❌ Feature scaling failed: {str(e)}")
        raise

def train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, label_name, tag, logger):
    model_name = f"{tag.upper()} - {label_name}"
    logger.info(f"\n{'='*20} TRAINING {model_name} {'='*20}")
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting training for {model_name}...")
        with tqdm(total=100, desc=f"Training {tag.upper()}", unit="%") as pbar:
            if hasattr(model, 'fit'):
                if 'lgb' in tag.lower():
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
                elif 'cat' in tag.lower():
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                elif 'xgb' in tag.lower():
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)
                pbar.update(100)
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"📊 Evaluating on validation set...")
        eval_start = time.time()
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        logger.info(f"🎯 {model_name} Validation Accuracy: {val_acc:.4f}")
        logger.info(f"📈 Validation Confusion Matrix:\n{confusion_matrix(y_val, y_val_pred)}")
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        logger.info(f"📋 Validation Classification Report:")
        for class_name, metrics in val_report.items():
            if isinstance(metrics, dict):
                logger.info(f"  Class {class_name}: Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}")
        logger.info(f"🧪 Evaluating on test set...")
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        eval_time = time.time() - eval_start
        logger.info(f"✅ Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"🎯 {model_name} Test Accuracy: {test_acc:.4f}")
        logger.info(f"📈 Test Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        logger.info(f"📋 Test Classification Report:")
        for class_name, metrics in test_report.items():
            if isinstance(metrics, dict):
                logger.info(f"  Class {class_name}: Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}")
        fname = f"step3_{tag}_{label_name}_model.joblib"
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
        return val_acc, test_acc, training_time, eval_time
    except Exception as e:
        logger.error(f"❌ Training/evaluation failed for {model_name}: {str(e)}")
        raise

def save_comprehensive_documentation(results, feature_cols, total_time, logger):
    logger.info("📝 Saving comprehensive documentation...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("step3_ultimate_models_versions.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("ULTIMATE ML MODEL TRAINING PIPELINE RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Pipeline Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
            f.write("MODELS TRAINED:\n")
            f.write("- RandomForestClassifier (oracle_best_rateIdx, v3_rateIdx)\n")
            f.write("- LGBMClassifier (oracle_best_rateIdx, v3_rateIdx)\n")
            f.write("- XGBoostClassifier (oracle_best_rateIdx, v3_rateIdx)\n")
            f.write("- CatBoostClassifier (oracle_best_rateIdx, v3_rateIdx)\n")
            f.write("- MLPClassifier (oracle_best_rateIdx, v3_rateIdx)\n\n")
            f.write("CONFIGURATION:\n")
            f.write("- Scaler: StandardScaler\n")
            f.write("- Dataset: smartv4-ml-ready-cleaned.csv\n")
            f.write(f"- Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
            f.write("- Split: 80/10/10 stratified\n")
            f.write("- Random State: 42\n\n")
            f.write("DETAILED RESULTS:\n")
            model_names = [
                "Random Forest (oracle_best_rateIdx)",
                "Random Forest (v3_rateIdx)",
                "LightGBM (oracle_best_rateIdx)", 
                "LightGBM (v3_rateIdx)",
                "XGBoost (oracle_best_rateIdx)",
                "XGBoost (v3_rateIdx)",
                "CatBoost (oracle_best_rateIdx)",
                "CatBoost (v3_rateIdx)",
                "MLP (oracle_best_rateIdx)",
                "MLP (v3_rateIdx)"
            ]
            for idx, (model_name, model_results) in enumerate(zip(model_names, results)):
                val_acc, test_acc, train_time, eval_time = model_results
                f.write(f"{idx+1}. {model_name}:\n")
                f.write(f"   Validation Accuracy: {val_acc:.4f}\n")
                f.write(f"   Test Accuracy: {test_acc:.4f}\n")
                f.write(f"   Training Time: {train_time:.2f}s\n")
                f.write(f"   Evaluation Time: {eval_time:.2f}s\n\n")
            f.write("FILES GENERATED:\n")
            f.write("- step3_scaler.joblib (StandardScaler)\n")
            f.write("- All model .joblib files\n")
            f.write("- step3_ultimate_models_versions.txt (this file)\n\n")
            f.write("NEXT STEPS:\n")
            f.write("- Ready for Step 4: Integration with ns-3 or ensemble selection\n")
            f.write("- All models are trained and saved for deployment\n")
        logger.info("✅ Documentation saved to step3_ultimate_models_versions.txt")
    except Exception as e:
        logger.error(f"❌ Documentation saving failed: {str(e)}")
        raise

def main():
    logger = setup_logging()
    pipeline_start = time.time()
    try:
        feature_cols = [
            'lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
            'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
            'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance'
        ]
        label_oracle = 'oracle_best_rateIdx'
        label_v3 = 'v3_rateIdx'
        logger.info("📂 Loading dataset...")
        df = pd.read_csv("smartv4-ml-ready-cleaned.csv")
        logger.info(f"✅ Dataset loaded: {df.shape}")
        validate_data(df, feature_cols, label_oracle, label_v3, logger)
        X = df[feature_cols]
        y_oracle = df[label_oracle]
        y_v3 = df[label_v3]
        X_train, X_val, X_test, y_oracle_train, y_oracle_val, y_oracle_test, y_v3_train, y_v3_val, y_v3_test = perform_train_split(
            X, y_oracle, y_v3, logger)
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test, logger)
        results = []
        models_to_train = [
            # Random Forest
            (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
             X_train_scaled, y_oracle_train, X_val_scaled, y_oracle_val, X_test_scaled, y_oracle_test, label_oracle, "rf"),
            (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
             X_train_scaled, y_v3_train, X_val_scaled, y_v3_val, X_test_scaled, y_v3_test, label_v3, "rf"),
            # LightGBM
            (lgb.LGBMClassifier(n_estimators=100, max_depth=12, random_state=42, verbose=-1),
             X_train_scaled, y_oracle_train, X_val_scaled, y_oracle_val, X_test_scaled, y_oracle_test, label_oracle, "lgb"),
            (lgb.LGBMClassifier(n_estimators=100, max_depth=12, random_state=42, verbose=-1),
             X_train_scaled, y_v3_train, X_val_scaled, y_v3_val, X_test_scaled, y_v3_test, label_v3, "lgb"),
            # XGBoost
            (xgb.XGBClassifier(n_estimators=100, max_depth=12, random_state=42, tree_method='hist', verbosity=0),
             X_train_scaled, y_oracle_train, X_val_scaled, y_oracle_val, X_test_scaled, y_oracle_test, label_oracle, "xgb"),
            (xgb.XGBClassifier(n_estimators=100, max_depth=12, random_state=42, tree_method='hist', verbosity=0),
             X_train_scaled, y_v3_train, X_val_scaled, y_v3_val, X_test_scaled, y_v3_test, label_v3, "xgb"),
            # CatBoost
            (CatBoostClassifier(iterations=100, depth=12, random_seed=42, verbose=False),
             X_train_scaled, y_oracle_train, X_val_scaled, y_oracle_val, X_test_scaled, y_oracle_test, label_oracle, "cat"),
            (CatBoostClassifier(iterations=100, depth=12, random_seed=42, verbose=False),
             X_train_scaled, y_v3_train, X_val_scaled, y_v3_val, X_test_scaled, y_v3_test, label_v3, "cat"),
            # MLPClassifier (Neural Net)
            (MLPClassifier(hidden_layer_sizes=(64,64), max_iter=30, random_state=42, verbose=False),
             X_train_scaled, y_oracle_train, X_val_scaled, y_oracle_val, X_test_scaled, y_oracle_test, label_oracle, "mlp"),
            (MLPClassifier(hidden_layer_sizes=(64,64), max_iter=30, random_state=42, verbose=False),
             X_train_scaled, y_v3_train, X_val_scaled, y_v3_val, X_test_scaled, y_v3_test, label_v3, "mlp"),
        ]
        logger.info(f"\n🎯 Starting training pipeline for {len(models_to_train)} models...")
        with tqdm(total=len(models_to_train), desc="Overall Progress", unit="model") as overall_pbar:
            for i, (model, X_tr, y_tr, X_va, y_va, X_te, y_te, label, tag) in enumerate(models_to_train):
                overall_pbar.set_postfix({"current": f"{tag.upper()}-{label}"})
                model_results = train_and_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te, label, tag, logger)
                results.append(model_results)
                overall_pbar.update(1)
        total_time = time.time() - pipeline_start
        save_comprehensive_documentation(results, feature_cols, total_time, logger)
        logger.info("\n" + "="*60)
        logger.info("🎉 STEP 3 ULTIMATE MODELS PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"⏱️  Total Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"📊 Models Trained: {len(results)}")
        logger.info(f"💾 Files Generated: {len(results) + 2} (models + scaler + documentation)")
        logger.info("✅ All models, scaler, and documentation files saved.")
        logger.info("🚀 Ready for Step 4: Integration with ns-3 or ensemble selection.")
        logger.info("\n📈 PERFORMANCE SUMMARY:")
        model_names = [
            "RF-Oracle", "RF-V3", "LGB-Oracle", "LGB-V3",
            "XGB-Oracle", "XGB-V3", "CAT-Oracle", "CAT-V3",
            "MLP-Oracle", "MLP-V3"
        ]
        for name, (val_acc, test_acc, _, _) in zip(model_names, results):
            logger.info(f"   {name}: Val={val_acc:.4f}, Test={test_acc:.4f}")
        return True
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {str(e)}")
        logger.error("Please ensure 'smartv4-ml-ready-cleaned.csv' exists in the current directory")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        logger.error("Check the log file for detailed error information")
        return False
    finally:
        if 'pipeline_start' in locals():
            final_time = time.time() - pipeline_start
            logger.info(f"\n⏱️  Total execution time: {final_time:.2f} seconds")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)