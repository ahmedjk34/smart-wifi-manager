"""
Step 5: Enhanced Debugging Model Evaluation for WiFi Rate Adaptation ML Pipeline

This script provides exhaustive evaluation WITH COMPREHENSIVE DEBUGGING to identify issues like:
- Data leakage detection
- Class distribution verification
- Feature leakage analysis
- Cross-validation testing
- Train/test contamination checks
- Realistic performance benchmarking

Author: ahmedjk34 (github.com/ahmedjk34)
Date: 2025-09-22
Pipeline Stage: Step 5 - Enhanced Debugging Evaluation
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
CSV_FILE = "smart-v3-ml-enriched.csv"
MODEL_FILE = "step3_rf_rateIdx_model_FIXED.joblib"
SCALER_FILE = "step3_scaler_FIXED.joblib"
CLASS_WEIGHTS_FILE = "python_files/model_artifacts/class_weights.json"

# ALL POSSIBLE FEATURES (will validate which exist)
ALL_FEATURE_COLS = [
    "phyRate", "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
    "snrStabilityIndex", "snrPredictionConfidence", "shortSuccRatio", "medSuccRatio", 
    "consecSuccess", "consecFailure", "recentThroughputTrend", "packetLossRate",
    "retrySuccessRatio", "recentRateChanges", "timeSinceLastRateChange", 
    "rateStabilityScore", "optimalRateDistance", "aggressiveFactor", 
    "conservativeFactor", "recommendedSafeRate", "severity", "confidence",
    "T1", "T2", "T3", "decisionReason", "packetSuccess", "offeredLoad", 
    "queueLen", "retryCount", "channelWidth", "mobilityMetric", "snrVariance"
]

# POTENTIALLY LEAKY FEATURES (that might contain future information)
SUSPICIOUS_FEATURES = [
    "phyRate",  # Current rate - might be giving away the answer
    "T1", "T2", "T3",  # Unknown transformation features
    "decisionReason",  # Might encode the decision
    "recommendedSafeRate",  # Might be the answer itself
    "optimalRateDistance"  # Distance from optimal might reveal answer
]

# SAFE FEATURES (definitely no leakage)
SAFE_FEATURES = [
    "lastSnr", "snrFast", "snrSlow", "snrVariance", "snrTrendShort",
    "snrStabilityIndex", "snrPredictionConfidence", "shortSuccRatio", "medSuccRatio", 
    "consecSuccess", "consecFailure", "packetLossRate", "retrySuccessRatio",
    "recentRateChanges", "timeSinceLastRateChange", "rateStabilityScore",
    "severity", "confidence", "packetSuccess", "offeredLoad", 
    "queueLen", "retryCount", "channelWidth", "mobilityMetric"
]

TARGET_LABEL = "rateIdx"
CONTEXT_LABEL = "network_context"
ORACLE_LABELS = ["oracle_conservative", "oracle_balanced", "oracle_aggressive"]

# WiFi Rate Information
WIFI_RATES = {
    0: {"rate": "1 Mbps", "modulation": "BPSK", "coding": "1/2"},
    1: {"rate": "2 Mbps", "modulation": "QPSK", "coding": "1/2"},
    2: {"rate": "5.5 Mbps", "modulation": "CCK", "coding": "N/A"},
    3: {"rate": "6 Mbps", "modulation": "BPSK", "coding": "1/2"},
    4: {"rate": "9 Mbps", "modulation": "BPSK", "coding": "3/4"},
    5: {"rate": "11 Mbps", "modulation": "CCK", "coding": "N/A"},
    6: {"rate": "12 Mbps", "modulation": "QPSK", "coding": "1/2"},
    7: {"rate": "18 Mbps", "modulation": "QPSK", "coding": "3/4"}
}

RANDOM_STATE = 42
OUTPUT_DIR = Path("debug_evaluation_results")

# ================== SETUP AND UTILITIES ==================
def setup_logging_and_output():
    """Setup comprehensive logging and output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"debug_evaluation_log_{timestamp}.log"
    
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
    logger.info("STEP 5: ENHANCED DEBUGGING WiFi RATE ADAPTATION MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"🔍 DEBUGGING MODE: Will identify 100% accuracy causes")
    
    return logger

class DebugProgress:
    """Enhanced progress tracker with debugging capabilities."""
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.issues_found = []
        self.warning_flags = []
        self.current_stage = None
        
    def start_stage(self, stage_name):
        self.current_stage = stage_name
        self.logger.info(f"\n🔄 DEBUG STAGE: {stage_name}")
        
    def add_issue(self, issue_type, description, severity="WARNING"):
        issue = {
            'type': issue_type,
            'description': description,
            'severity': severity,
            'stage': self.current_stage,
            'timestamp': time.time() - self.start_time
        }
        self.issues_found.append(issue)
        
        emoji = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "WARNING" else "ℹ️"
        self.logger.error(f"{emoji} {severity}: {description}")
        
    def log_success(self, task_name, details=None):
        elapsed = time.time() - self.start_time
        self.logger.info(f"✅ {task_name} - {elapsed:.1f}s")
        if details:
            self.logger.info(f"   {details}")
    
    def get_debug_summary(self):
        critical_issues = [i for i in self.issues_found if i['severity'] == 'CRITICAL']
        warnings = [i for i in self.issues_found if i['severity'] == 'WARNING']
        return {
            'total_issues': len(self.issues_found),
            'critical_issues': len(critical_issues),
            'warnings': len(warnings),
            'issues': self.issues_found
        }

# ================== DEBUGGING FUNCTIONS ==================
def debug_data_integrity(df, logger, progress):
    """Comprehensive data integrity and leakage detection."""
    progress.start_stage("Data Integrity and Leakage Detection")
    
    try:
        # 1. Check class distribution in full dataset
        if TARGET_LABEL in df.columns:
            full_class_dist = df[TARGET_LABEL].value_counts().sort_index()
            logger.info("🔍 FULL DATASET Class Distribution:")
            
            missing_classes = []
            for class_id in range(8):
                count = full_class_dist.get(class_id, 0)
                pct = (count / len(df)) * 100
                logger.info(f"  Class {class_id}: {count:,} samples ({pct:.1f}%)")
                
                if count == 0:
                    missing_classes.append(class_id)
                elif count < 10:
                    progress.add_issue("LOW_SAMPLE_COUNT", 
                                     f"Class {class_id} has only {count} samples", "WARNING")
            
            if missing_classes:
                progress.add_issue("MISSING_CLASSES", 
                                 f"Classes {missing_classes} completely missing from dataset", "CRITICAL")
        
        # 2. Check for perfect correlations (data leakage indicators)
        logger.info("\n🔍 Checking for potential data leakage...")
        available_features = [col for col in ALL_FEATURE_COLS if col in df.columns]
        
        if TARGET_LABEL in df.columns and available_features:
            # Check correlation between features and target
            feature_target_corr = []
            for feature in available_features:
                if df[feature].dtype in ['int64', 'float64']:
                    try:
                        corr = df[feature].corr(df[TARGET_LABEL])
                        if abs(corr) > 0.9:
                            progress.add_issue("HIGH_CORRELATION", 
                                             f"Feature '{feature}' has correlation {corr:.3f} with target", "CRITICAL")
                        feature_target_corr.append((feature, corr))
                    except:
                        continue
            
            # Sort by absolute correlation
            feature_target_corr.sort(key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0, reverse=True)
            logger.info("📊 Feature-Target Correlations (top 10):")
            for feature, corr in feature_target_corr[:10]:
                status = "🚨 SUSPICIOUS" if abs(corr) > 0.8 else "✅ OK"
                logger.info(f"  {feature}: {corr:.3f} {status}")
        
        # 3. Check for suspicious feature values
        logger.info("\n🔍 Checking for suspicious feature patterns...")
        for feature in SUSPICIOUS_FEATURES:
            if feature in df.columns:
                unique_vals = df[feature].nunique()
                if unique_vals == len(df[TARGET_LABEL].unique()):
                    progress.add_issue("PERFECT_FEATURE_MAPPING", 
                                     f"Feature '{feature}' has exactly {unique_vals} unique values matching target classes", "CRITICAL")
        
        # 4. Check for data duplication
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            progress.add_issue("DATA_DUPLICATION", 
                             f"Found {duplicates} duplicate rows", "WARNING")
        
        progress.log_success("Data integrity check")
        return available_features
        
    except Exception as e:
        progress.add_issue("DEBUG_ERROR", f"Data integrity check failed: {str(e)}", "CRITICAL")
        return []

def debug_train_test_split_integrity(X, y, logger, progress):
    """Debug train/test split to ensure all classes are preserved."""
    progress.start_stage("Train/Test Split Integrity Check")
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Check original distribution
        original_dist = pd.Series(y).value_counts().sort_index()
        logger.info("🔍 Original Class Distribution:")
        for class_id, count in original_dist.items():
            pct = (count / len(y)) * 100
            logger.info(f"  Class {class_id}: {count:,} samples ({pct:.1f}%)")
        
        # Test multiple split strategies
        strategies = [
            ("30% Test Split", 0.3),
            ("20% Test Split", 0.2),
            ("10% Test Split", 0.1)
        ]
        
        for strategy_name, test_size in strategies:
            logger.info(f"\n📊 Testing {strategy_name}:")
            try:
                _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
                test_dist = pd.Series(y_test).value_counts().sort_index()
                
                missing_in_test = []
                for class_id in range(8):
                    if class_id in original_dist.index and class_id not in test_dist.index:
                        missing_in_test.append(class_id)
                
                if missing_in_test:
                    progress.add_issue("SPLIT_MISSING_CLASSES", 
                                     f"{strategy_name}: Missing classes {missing_in_test} in test set", "CRITICAL")
                else:
                    logger.info(f"  ✅ All classes preserved in {strategy_name}")
                    
            except Exception as e:
                progress.add_issue("SPLIT_ERROR", 
                                 f"{strategy_name}: Split failed - {str(e)}", "CRITICAL")
        
        progress.log_success("Train/test split integrity check")
        
    except Exception as e:
        progress.add_issue("SPLIT_DEBUG_ERROR", f"Split debugging failed: {str(e)}", "CRITICAL")

def debug_feature_leakage_analysis(df, available_features, logger, progress):
    """Analyze features for potential leakage by testing prediction capability."""
    progress.start_stage("Feature Leakage Analysis")
    
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        
        if TARGET_LABEL not in df.columns:
            progress.add_issue("NO_TARGET", "Target label not found for leakage analysis", "CRITICAL")
            return
        
        X_full = df[available_features]
        y_full = df[TARGET_LABEL].astype(int)
        
        # Remove rows with missing target
        valid_mask = y_full.notna()
        X_full = X_full[valid_mask]
        y_full = y_full[valid_mask]
        
        logger.info("🔍 Testing individual features for leakage potential:")
        
        leaky_features = []
        
        for feature in available_features:
            if feature in SUSPICIOUS_FEATURES:
                try:
                    # Test if single feature can predict target with high accuracy
                    X_single = X_full[[feature]].fillna(0)
                    
                    # Quick decision tree test
                    dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
                    scores = cross_val_score(dt, X_single, y_full, cv=3, scoring='accuracy')
                    avg_score = scores.mean()
                    
                    status = "🚨 LEAKY" if avg_score > 0.9 else "⚠️ SUSPICIOUS" if avg_score > 0.7 else "✅ OK"
                    logger.info(f"  {feature}: {avg_score:.3f} accuracy {status}")
                    
                    if avg_score > 0.9:
                        leaky_features.append((feature, avg_score))
                        progress.add_issue("FEATURE_LEAKAGE", 
                                         f"Feature '{feature}' achieves {avg_score:.3f} accuracy alone", "CRITICAL")
                    elif avg_score > 0.7:
                        progress.add_issue("SUSPICIOUS_FEATURE", 
                                         f"Feature '{feature}' achieves {avg_score:.3f} accuracy alone", "WARNING")
                        
                except Exception as e:
                    logger.info(f"  {feature}: Error testing - {str(e)}")
        
        if leaky_features:
            logger.info(f"\n🚨 IDENTIFIED LEAKY FEATURES:")
            for feature, score in leaky_features:
                logger.info(f"  {feature}: {score:.3f} accuracy")
        
        progress.log_success("Feature leakage analysis", f"Tested {len(available_features)} features")
        
    except Exception as e:
        progress.add_issue("LEAKAGE_ANALYSIS_ERROR", f"Feature leakage analysis failed: {str(e)}", "CRITICAL")

def debug_cross_validation_reality_check(X, y, artifacts, logger, progress):
    """Perform cross-validation to check if 100% accuracy is realistic."""
    progress.start_stage("Cross-Validation Reality Check")
    
    try:
        # Remove any potentially leaky features for clean test
        safe_features_available = [f for f in SAFE_FEATURES if f in X.columns]
        X_safe = X[safe_features_available]
        
        logger.info(f"🔍 Cross-validation with {len(safe_features_available)} SAFE features:")
        logger.info(f"Safe features: {safe_features_available}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_safe.fillna(0))
        
        # Test with simple model first
        simple_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE)
        
        # 5-fold cross-validation
        cv_scores = cross_val_score(simple_rf, X_scaled, y, cv=5, scoring='accuracy')
        
        logger.info("📊 5-Fold Cross-Validation Results (Safe Features Only):")
        for i, score in enumerate(cv_scores):
            logger.info(f"  Fold {i+1}: {score:.4f} ({score*100:.1f}%)")
        
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        
        logger.info(f"📈 CV Mean: {mean_cv:.4f} ± {std_cv:.4f}")
        
        # Reality check
        if mean_cv > 0.99:
            progress.add_issue("UNREALISTIC_CV", 
                             f"Cross-validation shows {mean_cv:.1%} accuracy - likely data leakage", "CRITICAL")
        elif mean_cv > 0.95:
            progress.add_issue("HIGH_CV", 
                             f"Cross-validation shows {mean_cv:.1%} accuracy - check for overfitting", "WARNING")
        else:
            logger.info(f"✅ Realistic CV performance: {mean_cv:.1%}")
        
        # Test original model if available
        if artifacts and 'model' in artifacts and 'scaler' in artifacts:
            logger.info("\n🔍 Testing ORIGINAL model with cross-validation:")
            original_model = artifacts['model']
            
            # Use all available features that model was trained on
            model_features = [f for f in ALL_FEATURE_COLS if f in X.columns]
            X_model = X[model_features]
            X_model_scaled = artifacts['scaler'].transform(X_model.fillna(0))
            
            # Manual cross-validation to avoid retraining
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            original_cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_model_scaled, y)):
                X_val_fold = X_model_scaled[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                y_pred_fold = original_model.predict(X_val_fold)
                fold_acc = accuracy_score(y_val_fold, y_pred_fold)
                original_cv_scores.append(fold_acc)
                
                logger.info(f"  Original Model Fold {fold+1}: {fold_acc:.4f} ({fold_acc*100:.1f}%)")
            
            original_mean = np.mean(original_cv_scores)
            logger.info(f"📈 Original Model CV Mean: {original_mean:.4f}")
            
            if original_mean > 0.99:
                progress.add_issue("ORIGINAL_MODEL_UNREALISTIC", 
                                 f"Original model CV: {original_mean:.1%} - confirms data leakage", "CRITICAL")
        
        progress.log_success("Cross-validation reality check", f"CV accuracy: {mean_cv:.3f}")
        
    except Exception as e:
        progress.add_issue("CV_ERROR", f"Cross-validation failed: {str(e)}", "CRITICAL")

def debug_model_prediction_patterns(model, scaler, X, y, logger, progress):
    """Analyze model prediction patterns to detect anomalies."""
    progress.start_stage("Model Prediction Pattern Analysis")
    
    try:
        # Make predictions
        X_scaled = scaler.transform(X.fillna(0))
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        # Analyze prediction confidence
        max_probabilities = np.max(y_pred_proba, axis=1)
        
        logger.info("🔍 Prediction Confidence Analysis:")
        logger.info(f"  Mean confidence: {max_probabilities.mean():.3f}")
        logger.info(f"  Min confidence: {max_probabilities.min():.3f}")
        logger.info(f"  % predictions with >99% confidence: {(max_probabilities > 0.99).mean()*100:.1f}%")
        
        if (max_probabilities > 0.99).mean() > 0.8:
            progress.add_issue("OVERCONFIDENT_PREDICTIONS", 
                             f"{(max_probabilities > 0.99).mean()*100:.1f}% of predictions have >99% confidence", "WARNING")
        
        # Check prediction distribution vs actual
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        actual_dist = pd.Series(y).value_counts().sort_index()
        
        logger.info("\n📊 Prediction vs Actual Distribution:")
        logger.info(f"{'Class':<8} {'Actual':<10} {'Predicted':<10} {'Diff':<10}")
        logger.info("-" * 40)
        
        for class_id in range(8):
            actual_count = actual_dist.get(class_id, 0)
            pred_count = pred_dist.get(class_id, 0)
            diff = pred_count - actual_count
            
            logger.info(f"{class_id:<8} {actual_count:<10} {pred_count:<10} {diff:<10}")
            
            if actual_count == 0 and pred_count > 0:
                progress.add_issue("PREDICTING_MISSING_CLASS", 
                                 f"Model predicts class {class_id} but it doesn't exist in data", "CRITICAL")
        
        progress.log_success("Prediction pattern analysis")
        
    except Exception as e:
        progress.add_issue("PREDICTION_ANALYSIS_ERROR", f"Prediction analysis failed: {str(e)}", "CRITICAL")

def perform_clean_model_test(X, y, logger, progress):
    """Train a new model with only safe features to test realistic performance."""
    progress.start_stage("Clean Model Test (Safe Features Only)")
    
    try:
        # Use only safe features
        safe_features = [f for f in SAFE_FEATURES if f in X.columns]
        X_safe = X[safe_features].fillna(0)
        
        logger.info(f"🧪 Training clean model with {len(safe_features)} safe features:")
        logger.info(f"Features: {safe_features}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_safe, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train simple model
        clean_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        clean_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clean_model.predict(X_test_scaled)
        clean_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"🎯 Clean Model Accuracy: {clean_accuracy:.4f} ({clean_accuracy*100:.1f}%)")
        
        # Per-class performance
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        logger.info("📊 Clean Model Per-Class Performance:")
        for class_id in range(8):
            if str(class_id) in class_report:
                metrics = class_report[str(class_id)]
                logger.info(f"  Class {class_id}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Reality assessment
        if clean_accuracy > 0.95:
            progress.add_issue("CLEAN_MODEL_HIGH", 
                             f"Even clean model achieves {clean_accuracy:.1%} - may indicate inherent leakage", "WARNING")
        elif clean_accuracy < 0.7:
            logger.info(f"✅ Realistic clean model performance: {clean_accuracy:.1%}")
        else:
            logger.info(f"📊 Good clean model performance: {clean_accuracy:.1%}")
        
        progress.log_success("Clean model test", f"Clean accuracy: {clean_accuracy:.3f}")
        
    except Exception as e:
        progress.add_issue("CLEAN_MODEL_ERROR", f"Clean model test failed: {str(e)}", "CRITICAL")

# ================== MAIN EXECUTION ==================
def main():
    """Enhanced debugging evaluation to identify 100% accuracy causes."""
    logger = setup_logging_and_output()
    progress = DebugProgress(logger)
    
    try:
        logger.info("🚀 Starting ENHANCED DEBUGGING evaluation to identify 100% accuracy issues...")
        logger.info("🔍 This will help determine if results are realistic or indicate problems")
        
        # STAGE 1: Load and analyze raw data
        progress.start_stage("Data Loading and Initial Analysis")
        
        try:
            df = pd.read_csv(CSV_FILE, low_memory=False)
            progress.log_success("Load dataset", f"{len(df):,} samples loaded")
        except Exception as e:
            progress.add_issue("DATA_LOAD_ERROR", f"Failed to load dataset: {str(e)}", "CRITICAL")
            return False
        
        # STAGE 2: Comprehensive data integrity check
        available_features = debug_data_integrity(df, logger, progress)
        
        # STAGE 3: Prepare clean data for testing
        progress.start_stage("Data Preparation for Testing")
        
        # Clean data minimally
        initial_rows = len(df)
        df_clean = df.dropna(subset=[TARGET_LABEL])
        df_clean = df_clean.dropna(subset=available_features, thresh=int(len(available_features) * 0.5))
        
        logger.info(f"📊 Data after cleaning: {len(df_clean):,} rows ({len(df_clean)/initial_rows*100:.1f}% retained)")
        
        X = df_clean[available_features].fillna(0)
        y = df_clean[TARGET_LABEL].astype(int)
        
        # STAGE 4: Debug train/test split integrity
        debug_train_test_split_integrity(X, y, logger, progress)
        
        # STAGE 5: Feature leakage analysis
        debug_feature_leakage_analysis(df_clean, available_features, logger, progress)
        
        # STAGE 6: Load trained model for analysis
        artifacts = {}
        try:
            artifacts['model'] = joblib.load(MODEL_FILE)
            artifacts['scaler'] = joblib.load(SCALER_FILE)
            progress.log_success("Load model artifacts")
        except Exception as e:
            progress.add_issue("MODEL_LOAD_ERROR", f"Failed to load model: {str(e)}", "WARNING")
        
        # STAGE 7: Cross-validation reality check
        debug_cross_validation_reality_check(X, y, artifacts, logger, progress)
        
        # STAGE 8: Model prediction pattern analysis
        if artifacts:
            debug_model_prediction_patterns(artifacts['model'], artifacts['scaler'], X, y, logger, progress)
        
        # STAGE 9: Clean model test
        perform_clean_model_test(X, y, logger, progress)
        
        # FINAL ANALYSIS AND RECOMMENDATIONS
        progress.start_stage("Final Analysis and Recommendations")
        
        debug_summary = progress.get_debug_summary()
        
        logger.info("\n" + "="*80)
        logger.info("🔍 DEBUGGING ANALYSIS COMPLETE")
        logger.info("="*80)
        
        logger.info(f"📊 Issues Found: {debug_summary['total_issues']}")
        logger.info(f"🚨 Critical Issues: {debug_summary['critical_issues']}")
        logger.info(f"⚠️ Warnings: {debug_summary['warnings']}")
        
        if debug_summary['critical_issues'] > 0:
            logger.info("\n🚨 CRITICAL ISSUES IDENTIFIED:")
            for issue in debug_summary['issues']:
                if issue['severity'] == 'CRITICAL':
                    logger.info(f"  - {issue['type']}: {issue['description']}")
            
            logger.info("\n💡 RECOMMENDATIONS:")
            logger.info("1. 🔧 Remove potentially leaky features (especially phyRate)")
            logger.info("2. 🔄 Retrain model with safe features only")
            logger.info("3. 📊 Expect realistic accuracy of 85-95%")
            logger.info("4. ✅ Validate with proper cross-validation")
            
        elif debug_summary['warnings'] > 0:
            logger.info("\n⚠️ WARNINGS FOUND - INVESTIGATE FURTHER:")
            for issue in debug_summary['issues']:
                if issue['severity'] == 'WARNING':
                    logger.info(f"  - {issue['type']}: {issue['description']}")
        else:
            logger.info("\n✅ NO MAJOR ISSUES FOUND")
            logger.info("If you're still getting 100% accuracy, it might be legitimate!")
        
        # Save detailed debug report
        debug_report_file = OUTPUT_DIR / "debug_analysis_report.md"
        with open(debug_report_file, 'w') as f:
            f.write("# WiFi ML Model Debug Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Issues Found\n\n")
            for issue in debug_summary['issues']:
                f.write(f"### {issue['severity']}: {issue['type']}\n")
                f.write(f"- **Description:** {issue['description']}\n")
                f.write(f"- **Stage:** {issue['stage']}\n")
                f.write(f"- **Time:** {issue['timestamp']:.1f}s\n\n")
            
            if debug_summary['critical_issues'] > 0:
                f.write("## Recommendations\n\n")
                f.write("1. Remove leaky features (especially `phyRate`)\n")
                f.write("2. Retrain with safe features only\n")
                f.write("3. Expect 85-95% realistic accuracy\n")
                f.write("4. Use proper cross-validation\n")
        
        logger.info(f"\n📄 Detailed debug report saved: {debug_report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)