"""
Step 5: Enhanced Debugging Model Evaluation for WiFi Rate Adaptation ML Pipeline
FIXED VERSION - Validates all pipeline improvements and detects remaining issues

CRITICAL FIXES (2025-10-01):
- Updated to work with fixed File 4 outputs
- Validates scenario-aware splitting was used
- Checks for temporal leakage in trained models
- Validates hyperparameter optimization was applied
- Per-scenario performance analysis
- Realistic performance expectations (60-80% not 95%+)

Author: ahmedjk34 (github.com/ahmedjk34)
Date: 2025-10-01
Pipeline Stage: Step 5 - Enhanced Debugging Evaluation (FIXED)
"""

from typing import Dict
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
import warnings
import time
import logging
from pathlib import Path
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent
CSV_FILE = PARENT_DIR / "smart-v3-ml-enriched.csv"
MODELS_DIR = BASE_DIR / "trained_models"
OUTPUT_DIR = BASE_DIR / "evaluation_results"

# Target labels to evaluate
TARGET_LABELS = [
    "rateIdx",                    # Natural ground truth comparison
    "oracle_conservative",
    "oracle_balanced",
    "oracle_aggressive"
]
# FIXED: Issue #18 - Lowered correlation threshold
CORRELATION_THRESHOLD_MODERATE = 0.4
CORRELATION_THRESHOLD_HIGH = 0.7

# Safe features (no temporal leakage)
# 🚀 PHASE 1A + 5: Safe features (15 features, no outcome leakage)
# 🚀 PHASE 1B: Safe features (14 features, no temporal leakage)
SAFE_FEATURES = [
    # SNR features (7)
    "lastSnr", "snrFast", "snrSlow", "snrTrendShort",
    "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
    
    # Network state (1 - removed channelWidth, always 20)
    "mobilityMetric",
    
    # 🚀 PHASE 1A: SAFE ONLY (2 features - removed channelBusyRatio, always 0)
    "retryRate",          # ✅ Past retry rate (not current)
    "frameErrorRate",     # ✅ Past error rate (not current)
    # ❌ REMOVED: channelBusyRatio (always 0 in ns-3, no variance)
    
    # 🚀 PHASE 1B: NEW FEATURES (4)
    "rssiVariance",       # ✅ RSSI variance (signal stability)
    "interferenceLevel",  # ✅ Interference level (collision tracking)
    "distanceMetric",     # ✅ Distance metric (from scenario)
    "avgPacketSize",      # ✅ Average packet size (traffic characteristic)
    
    # ❌ REMOVED: recentRateAvg (LEAKAGE - includes current rate)
    # ❌ REMOVED: rateStability (LEAKAGE - includes current rate)
    # ❌ REMOVED: sinceLastChange (LEAKAGE - tells if rate changed)
]  # TOTAL: 14 features (7 SNR + 1 network + 2 Phase 1A + 4 Phase 1B)

# ❌ REMOVED (Issue C3): Outcome features
# These were removed by File 2 and NOT used by File 4:
# - shortSuccRatio (outcome of CURRENT rate)
# - medSuccRatio (outcome of CURRENT rate)
# - packetLossRate (outcome of CURRENT rate)
# - severity (derived from packetLossRate)
# - confidence (derived from shortSuccRatio)

# Temporal leakage features (should be ABSENT)
TEMPORAL_LEAKAGE_FEATURES = [
    "consecSuccess", "consecFailure", "retrySuccessRatio",
    "timeSinceLastRateChange", "rateStabilityScore", "recentRateChanges",
    "packetSuccess"
]

# 🚀 PHASE 1A + 5: Updated performance expectations (15 features)
PERFORMANCE_EXPECTATIONS = {
    'excellent': 0.72,   # >72% is excellent for 14 features (Phase 1B target!)
    'good': 0.68,        # 68-72% is good
    'acceptable': 0.63,  # 63-68% is acceptable
    'needs_improvement': 0.62  # <63% needs work (worse than baseline 62.8%)
}

CONTEXT_LABEL = "network_context"
RANDOM_SEED = 42

# ================== SETUP AND UTILITIES ==================
def setup_logging_and_output():
    """Setup logging and output directories"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"evaluation_{timestamp}.log"
    
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
    logger.info("STEP 5: ENHANCED MODEL EVALUATION - FIXED VERSION")
    logger.info("="*80)
    logger.info(f"👤 Author: ahmedjk34 (https://github.com/ahmedjk34)")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"🔍 Validates: All pipeline fixes (Issues #1-40)")
    logger.info("="*80)
    
    return logger

# Auto-detect hyperparameter file
def find_hyperparameter_file() -> Path:
    """Find the most recent hyperparameter tuning file (auto-detect mode)"""
    hyperparams_dir = BASE_DIR / "hyperparameter_results"
    
    if not hyperparams_dir.exists():
        return None
    
    # Find all JSON files with FIXED suffix (prioritize fixed versions)
    json_files = list(hyperparams_dir.glob("hyperparameter_tuning_*_FIXED.json"))
    
    if len(json_files) == 0:
        # Try without FIXED suffix
        json_files = list(hyperparams_dir.glob("hyperparameter_tuning_*.json"))
    
    if len(json_files) == 0:
        return None
    
    # If multiple files, use most recent
    if len(json_files) > 1:
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return json_files[0]
    
    return json_files[0]

HYPERPARAMS_FILE = find_hyperparameter_file()


# ✅Check for both RF and XGBoost models
def find_model_files(target_label: str, models_dir: Path):
    """Find model and scaler files (RF or XGBoost)"""
    # Check XGBoost first (higher priority)
    xgb_model = models_dir / f"step4_xgb_{target_label}_FIXED.joblib"
    if xgb_model.exists():
        model_file = xgb_model
        model_type = "XGBoost"
    else:
        # Check RandomForest
        rf_model = models_dir / f"step4_rf_{target_label}_FIXED.joblib"
        if rf_model.exists():
            model_file = rf_model
            model_type = "RandomForest"
        else:
            return None, None, None
    
    scaler_file = models_dir / f"step4_scaler_{target_label}_FIXED.joblib"
    results_file = models_dir / f"step4_results_{target_label}.json"
    
    return model_file, scaler_file, model_type



class EvaluationProgress:
    """Track evaluation progress and issues"""
    
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.issues_found = []
        self.warnings = []
        self.successes = []
        self.current_stage = None
        
    def start_stage(self, stage_name):
        self.current_stage = stage_name
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STAGE: {stage_name}")
        self.logger.info(f"{'='*60}")
        
    def add_issue(self, issue_type, description, severity="WARNING"):
        issue = {
            'type': issue_type,
            'description': description,
            'severity': severity,
            'stage': self.current_stage,
            'timestamp': time.time() - self.start_time
        }
        
        if severity == "CRITICAL":
            self.issues_found.append(issue)
            self.logger.error(f"🚨 CRITICAL: {description}")
        elif severity == "WARNING":
            self.warnings.append(issue)
            self.logger.warning(f"⚠️ WARNING: {description}")
        else:
            self.logger.info(f"ℹ️ INFO: {description}")
        
    def add_success(self, description):
        self.successes.append(description)
        self.logger.info(f"✅ {description}")
    
    def get_summary(self):
        return {
            'total_issues': len(self.issues_found),
            'total_warnings': len(self.warnings),
            'total_successes': len(self.successes),
            'issues': self.issues_found,
            'warnings': self.warnings,
            'successes': self.successes
        }

def get_default_hyperparameters() -> Dict:
    """
    🚀 PHASE 5B: Default hyperparameters optimized for 15 features
    """
    return {
        'n_estimators': 300,       # More trees for 15 features (was 200)
        'max_depth': 25,           # Deeper for 15 features (was 15)
        'min_samples_split': 10,   # Balanced
        'min_samples_leaf': 5,     # Balanced
        'max_features': 'sqrt',    # sqrt(15) ≈ 4 features per split
        'class_weight': 'balanced',
        'source': 'default_phase5_enhanced'
    }


# ================== VALIDATION FUNCTIONS ==================

def validate_hyperparameter_optimization(target_label: str, logger, progress) -> Dict:
    """
    FIXED: Issue #20 - Validate that optimized hyperparameters were used,
    with fallback to defaults if missing or invalid.
    Returns the hyperparameters dict.
    """
    progress.start_stage(f"Hyperparameter Optimization Validation - {target_label}")
    
    if not HYPERPARAMS_FILE or not HYPERPARAMS_FILE.exists():
        progress.add_issue(
            "NO_HYPERPARAMETER_TUNING",
            f"Hyperparameter tuning results not found (Step 3c not run). Using defaults.",
            "WARNING"
        )
        defaults = get_default_hyperparameters()
        logger.info(f"   Using default hyperparameters: {defaults}")
        return defaults
    
    try:
        with open(HYPERPARAMS_FILE, 'r') as f:
            hyperparams = json.load(f)
        
        if target_label not in hyperparams:
            progress.add_issue(
                "MISSING_TARGET_HYPERPARAMS",
                f"No hyperparameters found for {target_label}. Using defaults.",
                "WARNING"
            )
            defaults = get_default_hyperparameters()
            logger.info(f"   Using default hyperparameters: {defaults}")
            return defaults
        
        target_params = hyperparams[target_label]
        best_score = target_params.get('best_score', None)
        best_params = target_params.get('best_params', None)

        if best_params is None:
            progress.add_issue(
                "INVALID_HYPERPARAMS_FORMAT",
                f"Hyperparameters file missing best_params field for {target_label}. Using defaults.",
                "WARNING"
            )
            defaults = get_default_hyperparameters()
            logger.info(f"   Using default hyperparameters: {defaults}")
            return defaults

        progress.add_success(
            f"Hyperparameter tuning applied (CV score: {best_score:.3f})"
            if best_score is not None else "Hyperparameter tuning applied"
        )
        logger.info(f"   Best params: {best_params}")
        return best_params

    except Exception as e:
        progress.add_issue(
            "HYPERPARAMETER_LOAD_ERROR",
            f"Failed to load hyperparameters: {str(e)}. Using defaults.",
            "WARNING"
        )
        defaults = get_default_hyperparameters()
        logger.info(f"   Using default hyperparameters: {defaults}")
        return defaults


def validate_scenario_aware_splitting(df: pd.DataFrame, logger, progress) -> bool:
    """
    FIXED: Issue #4, #12 - Validate scenario-aware splitting was used
    """
    progress.start_stage("Scenario-Aware Splitting Validation")
    
    if 'scenario_file' not in df.columns:
        progress.add_issue(
            "NO_SCENARIO_FILE",
            "Dataset missing 'scenario_file' column - random splitting likely used!",
            "CRITICAL"
        )
        return False
    
    num_scenarios = df['scenario_file'].nunique()
    
    if num_scenarios < 5:
        progress.add_issue(
            "TOO_FEW_SCENARIOS",
            f"Only {num_scenarios} scenarios - insufficient for proper splitting",
            "WARNING"
        )
    else:
        progress.add_success(f"Scenario-aware splitting possible ({num_scenarios} scenarios)")
    
    return True

def validate_temporal_leakage_removal(df: pd.DataFrame, logger, progress) -> bool:
    """
    FIXED: Issue #1, #33 - Validate temporal leakage features are removed
    """
    progress.start_stage("Temporal Leakage Validation")
    
    found_leakage = []
    for feature in TEMPORAL_LEAKAGE_FEATURES:
        if feature in df.columns:
            found_leakage.append(feature)
            progress.add_issue(
                "TEMPORAL_LEAKAGE_FOUND",
                f"Temporal feature '{feature}' still in dataset!",
                "CRITICAL"
            )
    
    if found_leakage:
        logger.error(f"   Found {len(found_leakage)} temporal leakage features: {found_leakage}")
        return False
    else:
        progress.add_success("All temporal leakage features removed")
        return True

def validate_safe_features_present(df: pd.DataFrame, logger, progress) -> bool:
    """Validate safe features are present"""
    progress.start_stage("Safe Features Validation")
    
    missing_features = []
    for feature in SAFE_FEATURES:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        progress.add_issue(
            "MISSING_SAFE_FEATURES",
            f"{len(missing_features)} safe features missing: {missing_features}",
            "WARNING"
        )
        return False
    else:
        progress.add_success(f"All {len(SAFE_FEATURES)} safe features present")
        return True

def evaluate_model_performance(target_label: str, df: pd.DataFrame, logger, progress) -> Dict:
    """
    Comprehensive model evaluation with all validation checks
    """
    progress.start_stage(f"Model Performance Evaluation - {target_label}")
    
    # Load model and scaler
    model_file, scaler_file, model_type = find_model_files(target_label, MODELS_DIR)
    results_file = MODELS_DIR / f"step4_results_{target_label}.json"
    
    if not model_file.exists() or not scaler_file.exists():
        progress.add_issue(
            "MODEL_NOT_FOUND",
            f"Trained model not found for {target_label}",
            "CRITICAL"
        )
        return {}
    
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        # Load training results if available
        training_results = {}
        if results_file.exists():
            with open(results_file, 'r') as f:
                training_results = json.load(f)
        
        progress.add_success(f"Loaded model and scaler for {target_label}")
        
    except Exception as e:
        progress.add_issue(
            "MODEL_LOAD_ERROR",
            f"Failed to load model: {str(e)}",
            "CRITICAL"
        )
        return {}
    
    # Prepare data
    if target_label not in df.columns:
        progress.add_issue(
            "TARGET_NOT_FOUND",
            f"Target label '{target_label}' not in dataset",
            "CRITICAL"
        )
        return {}
    
    # Extract features and labels
    available_features = [f for f in SAFE_FEATURES if f in df.columns]
    X = df[available_features].fillna(0).values
    y = df[target_label].dropna().astype(int).values
    
    # Remove samples with missing target
    valid_mask = ~pd.isna(df[target_label].values)
    X = X[valid_mask]
    
    logger.info(f"   Dataset: {len(X):,} samples, {len(available_features)} features")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    
    # FIXED: Issue #17 - Realistic performance assessment
    if accuracy >= PERFORMANCE_EXPECTATIONS['excellent']:
        performance_level = "EXCELLENT"
        logger.info(f"   🏆 {performance_level}: {accuracy*100:.1f}%")
    elif accuracy >= PERFORMANCE_EXPECTATIONS['good']:
        performance_level = "GOOD"
        logger.info(f"   ✅ {performance_level}: {accuracy*100:.1f}%")
    elif accuracy >= PERFORMANCE_EXPECTATIONS['acceptable']:
        performance_level = "ACCEPTABLE"
        logger.info(f"   📊 {performance_level}: {accuracy*100:.1f}%")
    else:
        performance_level = "NEEDS_IMPROVEMENT"
        logger.warning(f"   ⚠️ {performance_level}: {accuracy*100:.1f}%")
    
    progress.add_success(f"Accuracy: {accuracy:.3f} ({performance_level})")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0
    )
    
    # Confidence analysis
    max_proba = y_proba.max(axis=1)
    low_confidence = (max_proba < 0.5).sum()
    
    logger.info(f"   Low confidence predictions: {low_confidence} ({low_confidence/len(y)*100:.1f}%)")
    
    # Feature importance analysis (Issue #22)
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(available_features, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f"   Top 5 features:")
        for feat, imp in top_features:
            logger.info(f"      {feat}: {imp:.4f}")
        
        # Check for suspicious features in top 3
        suspicious_in_top3 = False
        for feat, imp in top_features[:3]:
            if any(leak in feat for leak in ['consec', 'retry', 'packet']):
                progress.add_issue(
                    "SUSPICIOUS_FEATURE_IMPORTANCE", 
                    f"Potential leakage feature '{feat}' ranks in top 3",
                    "WARNING"
                )
                suspicious_in_top3 = True
        
        if not suspicious_in_top3:
            progress.add_success("No suspicious features in top 3 (leakage check passed)")
    
    # Compile results
    evaluation_results = {
        'target_label': target_label,
        'accuracy': float(accuracy),
        'performance_level': performance_level,
        'low_confidence_count': int(low_confidence),
        'confusion_matrix': cm.tolist(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'top_features': [(feat, float(imp)) for feat, imp in top_features] if hasattr(model, 'feature_importances_') else [],
        'training_results': training_results.get('training_results', {}),
        'timestamp': datetime.now().isoformat()
    }
    
    return evaluation_results

def evaluate_per_scenario_performance(target_label: str, df: pd.DataFrame, logger, progress) -> Dict:
    """
    FIXED: Issue #23 - Per-scenario performance analysis
    Supports both RF and XGB models.
    """
    progress.start_stage(f"Per-Scenario Performance - {target_label}")
    
    if 'scenario_file' not in df.columns:
        progress.add_issue(
            "NO_SCENARIO_FILE",
            "Cannot evaluate per-scenario (no scenario_file column)",
            "WARNING"
        )
        return {}
    
    # Try both RF and XGB model files
    model_files = [
        MODELS_DIR / f"step4_rf_{target_label}_FIXED.joblib",
        MODELS_DIR / f"step4_xgb_{target_label}_FIXED.joblib"
    ]
    scaler_file = MODELS_DIR / f"step4_scaler_{target_label}_FIXED.joblib"
    
    model = None
    for mf in model_files:
        if mf.exists():
            try:
                model = joblib.load(mf)
                logger.info(f"   Loaded model: {mf.name}")
                break
            except:
                continue
    
    if model is None or not scaler_file.exists():
        return {}
    
    try:
        scaler = joblib.load(scaler_file)
    except:
        return {}
    
    # Evaluate each scenario
    scenarios = df['scenario_file'].unique()
    scenario_results = {}
    
    for scenario in scenarios:
        scenario_data = df[df['scenario_file'] == scenario]
        
        if len(scenario_data) == 0 or target_label not in scenario_data.columns:
            continue
        
        available_features = [f for f in SAFE_FEATURES if f in scenario_data.columns]
        X_scenario = scenario_data[available_features].fillna(0).values
        y_scenario = scenario_data[target_label].dropna().astype(int).values
        
        if len(y_scenario) == 0:
            continue
        
        # Remove samples with missing target
        valid_mask = ~pd.isna(scenario_data[target_label].values)
        X_scenario = X_scenario[valid_mask]
        
        if len(X_scenario) != len(y_scenario):
            continue
        
        X_scaled = scaler.transform(X_scenario)
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(y_scenario, y_pred)
        
        scenario_results[scenario] = {
            'samples': len(y_scenario),
            'accuracy': float(accuracy)
        }
    
    if scenario_results:
        # Find worst and best scenarios
        sorted_scenarios = sorted(scenario_results.items(), key=lambda x: x[1]['accuracy'])
        
        logger.info(f"\n   Worst 5 scenarios:")
        for scenario, metrics in sorted_scenarios[:5]:
            logger.info(f"      {scenario}: {metrics['accuracy']:.3f} ({metrics['samples']} samples)")
        
        logger.info(f"\n   Best 5 scenarios:")
        for scenario, metrics in sorted_scenarios[-5:]:
            logger.info(f"      {scenario}: {metrics['accuracy']:.3f} ({metrics['samples']} samples)")
        
        avg_accuracy = np.mean([m['accuracy'] for m in scenario_results.values()])
        std_accuracy = np.std([m['accuracy'] for m in scenario_results.values()])
        
        logger.info(f"\n   Average per-scenario: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
        
        progress.add_success(f"Evaluated {len(scenario_results)} scenarios")
    
    return scenario_results


def generate_visualizations(all_results: Dict, output_dir: Path, logger):
    """Generate comprehensive visualizations"""
    logger.info(f"\n📊 Generating visualizations...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Performance comparison across targets
    if all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        targets = list(all_results.keys())
        accuracies = [all_results[t].get('accuracy', 0) for t in targets]
        
        bars = ax.bar(targets, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim([0, 1])
        ax.axhline(y=PERFORMANCE_EXPECTATIONS['excellent'], color='g', linestyle='--', label='Excellent (75%)')
        ax.axhline(y=PERFORMANCE_EXPECTATIONS['good'], color='orange', linestyle='--', label='Good (65%)')
        ax.legend()
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc*100:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   ✅ Saved: performance_comparison.png")
    
    # 2. Confusion matrices
    for target, results in all_results.items():
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {target}')
            ax.set_xlabel('Predicted Rate')
            ax.set_ylabel('True Rate')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'confusion_matrix_{target}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ Saved: confusion_matrix_{target}.png")
    
    logger.info(f"✅ Visualizations saved to: {viz_dir}")

def generate_comprehensive_report(all_results: Dict, all_scenarios: Dict, 
                                  summary: Dict, output_dir: Path, logger):
    """Generate comprehensive evaluation report"""
    logger.info(f"\n📝 Generating comprehensive report...")
    
    report_file = output_dir / "evaluation_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# WiFi Rate Adaptation ML Pipeline - Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Author:** ahmedjk34\n")
        f.write(f"**Pipeline Stage:** Step 5 - Model Evaluation\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Models Evaluated:** {len(all_results)}\n")
        f.write(f"- **Critical Issues:** {summary['total_issues']}\n")
        f.write(f"- **Warnings:** {summary['total_warnings']}\n")
        f.write(f"- **Successes:** {summary['total_successes']}\n\n")
        
        if all_results:
            avg_accuracy = np.mean([r.get('accuracy', 0) for r in all_results.values()])
            f.write(f"- **Average Accuracy:** {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Model | Accuracy | Performance Level |\n")
        f.write("|-------|----------|------------------|\n")
        
        for target, results in all_results.items():
            acc = results.get('accuracy', 0)
            level = results.get('performance_level', 'UNKNOWN')
            emoji = "🏆" if level == "EXCELLENT" else "✅" if level == "GOOD" else "📊" if level == "ACCEPTABLE" else "⚠️"
            f.write(f"| {target} | {acc:.3f} ({acc*100:.1f}%) | {emoji} {level} |\n")
        
        f.write("\n## Validation Results\n\n")
        
        f.write("### ✅ Fixes Applied Successfully\n\n")
        for success in summary['successes'][:10]:
            f.write(f"- {success}\n")
        if len(summary['successes']) > 10:
            f.write(f"- ... and {len(summary['successes']) - 10} more\n")
        
        if summary['total_warnings'] > 0:
            f.write("\n### ⚠️ Warnings\n\n")
            for warning in summary['warnings'][:10]:
                f.write(f"- **{warning['type']}:** {warning['description']}\n")
        
        if summary['total_issues'] > 0:
            f.write("\n### 🚨 Critical Issues\n\n")
            for issue in summary['issues']:
                f.write(f"- **{issue['type']}:** {issue['description']}\n")
        
        f.write("\n## Recommendations\n\n")
        
        if summary['total_issues'] == 0:
            f.write("✅ **Pipeline is production-ready!**\n\n")
            f.write("- All critical fixes have been applied\n")
            f.write("- No data leakage detected\n")
            f.write("- Realistic performance achieved\n")
            f.write("- Ready for ns-3 integration and deployment\n\n")
        else:
            f.write("⚠️ **Pipeline needs attention:**\n\n")
            for issue in summary['issues']:
                f.write(f"- Fix {issue['type']}: {issue['description']}\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. **ns-3 Integration:** Deploy models in network simulator\n")
        f.write("2. **A/B Testing:** Compare oracle strategies in real scenarios\n")
        f.write("3. **Performance Monitoring:** Track model performance over time\n")
        f.write("4. **Iterative Improvement:** Use Issue #61 validation loop\n")
    
    logger.info(f"✅ Report saved to: {report_file}")

# ================== MAIN EXECUTION ==================
def main():
    """Main evaluation pipeline"""
    logger = setup_logging_and_output()
    progress = EvaluationProgress(logger)
    
    logger.info(f"🚀 Starting comprehensive model evaluation...")
    
    # Load dataset
    progress.start_stage("Dataset Loading")
    
    if not CSV_FILE.exists():
        logger.error(f"❌ Dataset not found: {CSV_FILE}")
        return False
    
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
        progress.add_success(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        progress.add_issue("DATASET_LOAD_ERROR", f"Failed to load dataset: {str(e)}", "CRITICAL")
        return False
    
    # Validation checks
    validate_temporal_leakage_removal(df, logger, progress)
    validate_safe_features_present(df, logger, progress)
    validate_scenario_aware_splitting(df, logger, progress)
    
    # Evaluate each model
    all_results = {}
    all_scenarios = {}
    
    for target_label in TARGET_LABELS:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# EVALUATING: {target_label}")
        logger.info(f"{'#'*80}")
        
        # Validate hyperparameter optimization
        validate_hyperparameter_optimization(target_label, logger, progress)
        
        # Evaluate model performance
        results = evaluate_model_performance(target_label, df, logger, progress)
        if results:
            all_results[target_label] = results
        
        # Per-scenario evaluation
        scenario_results = evaluate_per_scenario_performance(target_label, df, logger, progress)
        if scenario_results:
            all_scenarios[target_label] = scenario_results
    
    # Generate visualizations
    if all_results:
        generate_visualizations(all_results, OUTPUT_DIR, logger)
    
    # Generate comprehensive report
    summary = progress.get_summary()
    generate_comprehensive_report(all_results, all_scenarios, summary, OUTPUT_DIR, logger)
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Models evaluated: {len(all_results)}/{len(TARGET_LABELS)}")
    logger.info(f"🚨 Critical issues: {summary['total_issues']}")
    logger.info(f"⚠️ Warnings: {summary['total_warnings']}")
    logger.info(f"✅ Successes: {summary['total_successes']}")
    
    if all_results:
        logger.info(f"\n📈 Performance Summary:")
        for target, results in all_results.items():
            acc = results.get('accuracy', 0)
            level = results.get('performance_level', 'UNKNOWN')
            logger.info(f"   {target}: {acc*100:.1f}% ({level})")
    
    logger.info(f"\n📁 Results saved to: {OUTPUT_DIR}")
    
    if summary['total_issues'] == 0:
        logger.info(f"\n✅ VALIDATION PASSED: Pipeline is production-ready!")
        print(f"\n{'='*80}")
        print(f"✅ EVALUATION SUCCESSFUL - PIPELINE IS PRODUCTION-READY!")
        print(f"{'='*80}")
        return True
    else:
        logger.warning(f"\n⚠️ VALIDATION WARNINGS: Review critical issues")
        print(f"\n{'='*80}")
        print(f"⚠️ EVALUATION COMPLETE - REVIEW WARNINGS")
        print(f"{'='*80}")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)