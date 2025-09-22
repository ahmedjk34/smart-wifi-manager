"""
Step 5: Comprehensive and Detailed Model Evaluation for WiFi Rate Adaptation ML Pipeline

This script provides exhaustive evaluation of the trained Random Forest model with:
- Multi-dimensional performance analysis (overall, per-class, per-network-context)
- Class weight effectiveness assessment
- Feature importance analysis with WiFi domain interpretation
- Confusion matrix analysis with actionable insights
- Edge case performance evaluation
- Real-world deployment readiness assessment
- Comparative analysis against baseline approaches
- Comprehensive visualizations and reporting

Author: ahmedjk34 (github.com/ahmedjk34)
Date: 2025-09-22
Pipeline Stage: Step 5 - Model Evaluation
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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
import warnings
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
# Data and Model Files
CSV_FILE = "smart-v3-ml-enriched.csv"
MODEL_FILE = "step3_rf_rateIdx_model_FIXED.joblib"
SCALER_FILE = "step3_scaler_FIXED.joblib"
CLASS_WEIGHTS_FILE = "python_files/model_artifacts/class_weights.json"

# Feature Configuration (matches training)
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

# Evaluation Configuration
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = Path("evaluation_results")

# ================== SETUP AND UTILITIES ==================
def setup_logging_and_output():
    """Setup comprehensive logging and output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"evaluation_log_{timestamp}.log"
    
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
    logger.info("STEP 5: COMPREHENSIVE WiFi RATE ADAPTATION MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    return logger

class EvaluationProgress:
    """Track evaluation progress with detailed reporting."""
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_stage = None
        
    def start_stage(self, stage_name):
        self.current_stage = stage_name
        self.logger.info(f"\n🔄 Starting: {stage_name}")
        
    def complete_task(self, task_name, success=True, details=None):
        elapsed = time.time() - self.start_time
        status = "✅" if success else "❌"
        
        if success:
            self.completed_tasks.append((task_name, elapsed, details))
            self.logger.info(f"{status} {task_name} - {elapsed:.1f}s")
            if details:
                self.logger.info(f"   {details}")
        else:
            self.failed_tasks.append((task_name, elapsed, details))
            self.logger.error(f"{status} {task_name} FAILED - {details}")
    
    def get_summary(self):
        total_time = time.time() - self.start_time
        success_rate = len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) * 100
        return {
            'total_time': total_time,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': success_rate
        }

# ================== DATA LOADING AND PREPARATION ==================
def load_evaluation_data(logger, progress):
    """Load and prepare data for comprehensive evaluation."""
    progress.start_stage("Data Loading and Preparation")
    
    try:
        # Load dataset
        df = pd.read_csv(CSV_FILE, low_memory=False)
        progress.complete_task("Load dataset", True, f"{len(df):,} samples loaded")
        
        # Validate required columns
        missing_cols = [col for col in FEATURE_COLS + [TARGET_LABEL] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        progress.complete_task("Validate columns", True, "All required columns present")
        
        # Clean data
        initial_rows = len(df)
        df = df.dropna(subset=[TARGET_LABEL])
        df = df.dropna(subset=FEATURE_COLS, thresh=int(len(FEATURE_COLS) * 0.7))
        progress.complete_task("Clean data", True, f"Retained {len(df):,}/{initial_rows:,} samples")
        
        # Prepare features and target
        X = df[FEATURE_COLS]
        y = df[TARGET_LABEL].astype(int)
        
        # Additional context data if available
        context_data = None
        oracle_data = None
        
        if CONTEXT_LABEL in df.columns:
            context_data = df[CONTEXT_LABEL]
            progress.complete_task("Load context data", True, f"{len(context_data.unique())} unique contexts")
        
        available_oracles = [col for col in ORACLE_LABELS if col in df.columns]
        if available_oracles:
            oracle_data = df[available_oracles]
            progress.complete_task("Load oracle data", True, f"{len(available_oracles)} oracle strategies")
        
        # Data distribution analysis
        class_dist = y.value_counts().sort_index()
        logger.info("📊 Class distribution:")
        for class_id, count in class_dist.items():
            pct = (count / len(y)) * 100
            rate_info = WIFI_RATES[class_id]["rate"]
            logger.info(f"  Class {class_id} ({rate_info}): {count:,} samples ({pct:.1f}%)")
        
        return {
            'X': X,
            'y': y,
            'df': df,
            'context_data': context_data,
            'oracle_data': oracle_data,
            'class_distribution': class_dist
        }
        
    except Exception as e:
        progress.complete_task("Data loading", False, str(e))
        raise

def load_model_artifacts(logger, progress):
    """Load trained model, scaler, and class weights."""
    progress.start_stage("Model Artifacts Loading")
    
    artifacts = {}
    
    try:
        # Load model
        model = joblib.load(MODEL_FILE)
        artifacts['model'] = model
        progress.complete_task("Load model", True, f"Random Forest with {model.n_estimators} estimators")
        
        # Load scaler
        scaler = joblib.load(SCALER_FILE)
        artifacts['scaler'] = scaler
        progress.complete_task("Load scaler", True, "StandardScaler loaded")
        
        # Load class weights
        try:
            with open(CLASS_WEIGHTS_FILE, 'r') as f:
                all_weights = json.load(f)
            class_weights = all_weights.get(TARGET_LABEL, {})
            class_weights = {int(k): v for k, v in class_weights.items()}
            artifacts['class_weights'] = class_weights
            progress.complete_task("Load class weights", True, f"{len(class_weights)} class weights loaded")
        except Exception as e:
            progress.complete_task("Load class weights", False, str(e))
            artifacts['class_weights'] = None
        
        return artifacts
        
    except Exception as e:
        progress.complete_task("Model artifacts loading", False, str(e))
        raise

# ================== COMPREHENSIVE EVALUATION FUNCTIONS ==================
def perform_overall_evaluation(model, scaler, X, y, logger, progress):
    """Perform comprehensive overall model evaluation."""
    progress.start_stage("Overall Model Performance Evaluation")
    
    results = {}
    
    try:
        # Scale features
        X_scaled = scaler.transform(X)
        progress.complete_task("Feature scaling", True)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        progress.complete_task("Generate predictions", True)
        
        # Overall accuracy
        accuracy = accuracy_score(y, y_pred)
        results['accuracy'] = accuracy
        logger.info(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
        
        # Detailed classification report
        class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        results['classification_report'] = class_report
        
        # Log per-class performance
        logger.info("📊 Per-Class Performance:")
        logger.info(f"{'Class':<8} {'Rate':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Support':<8}")
        logger.info("-" * 60)
        
        for i in range(8):
            if str(i) in class_report:
                metrics = class_report[str(i)]
                rate = WIFI_RATES[i]["rate"]
                logger.info(f"{i:<8} {rate:<12} {metrics['precision']:<10.3f} {metrics['recall']:<8.3f} "
                           f"{metrics['f1-score']:<8.3f} {int(metrics['support']):<8}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        results['confusion_matrix'] = cm
        
        # Macro and weighted averages
        results['macro_avg'] = class_report['macro avg']
        results['weighted_avg'] = class_report['weighted avg']
        
        progress.complete_task("Overall evaluation", True, f"Accuracy: {accuracy:.1%}")
        return results
        
    except Exception as e:
        progress.complete_task("Overall evaluation", False, str(e))
        raise

def evaluate_class_weight_effectiveness(model, scaler, X, y, class_weights, class_dist, logger, progress):
    """Evaluate how effectively class weights handle imbalance."""
    progress.start_stage("Class Weight Effectiveness Analysis")
    
    results = {}
    
    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        # Analyze prediction distribution vs actual distribution
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        actual_dist = pd.Series(y).value_counts().sort_index()
        
        logger.info("⚖️ Class Weight Effectiveness Analysis:")
        logger.info(f"{'Class':<8} {'Weight':<10} {'Actual%':<10} {'Pred%':<10} {'Recall':<10} {'Status':<12}")
        logger.info("-" * 70)
        
        effectiveness_scores = {}
        
        for class_id in range(8):
            weight = class_weights.get(class_id, 1.0) if class_weights else 1.0
            actual_pct = (actual_dist.get(class_id, 0) / len(y)) * 100
            pred_pct = (pred_dist.get(class_id, 0) / len(y_pred)) * 100
            
            # Calculate recall for this class
            class_mask = (y == class_id)
            if class_mask.sum() > 0:
                recall = (y_pred[class_mask] == class_id).sum() / class_mask.sum()
            else:
                recall = 0
            
            # Effectiveness assessment
            if actual_pct < 1.0:  # Rare class
                if recall > 0.3:
                    status = "EFFECTIVE"
                elif recall > 0.1:
                    status = "MODERATE"
                else:
                    status = "POOR"
            else:  # Common class
                if recall > 0.9:
                    status = "EXCELLENT"
                elif recall > 0.8:
                    status = "GOOD"
                else:
                    status = "NEEDS_WORK"
            
            effectiveness_scores[class_id] = {
                'weight': weight,
                'actual_pct': actual_pct,
                'pred_pct': pred_pct,
                'recall': recall,
                'status': status
            }
            
            logger.info(f"{class_id:<8} {weight:<10.1f} {actual_pct:<10.1f} {pred_pct:<10.1f} "
                       f"{recall:<10.3f} {status:<12}")
        
        results['effectiveness_scores'] = effectiveness_scores
        
        # Overall effectiveness score
        rare_classes = [i for i in range(8) if actual_dist.get(i, 0) / len(y) < 0.01]
        rare_recall_avg = np.mean([effectiveness_scores[i]['recall'] for i in rare_classes])
        
        common_classes = [i for i in range(8) if actual_dist.get(i, 0) / len(y) >= 0.01]
        common_recall_avg = np.mean([effectiveness_scores[i]['recall'] for i in common_classes])
        
        overall_effectiveness = (rare_recall_avg * 0.4 + common_recall_avg * 0.6)
        results['overall_effectiveness'] = overall_effectiveness
        
        logger.info(f"\n📈 Class Weight Effectiveness Summary:")
        logger.info(f"  Rare classes average recall: {rare_recall_avg:.3f}")
        logger.info(f"  Common classes average recall: {common_recall_avg:.3f}")
        logger.info(f"  Overall effectiveness score: {overall_effectiveness:.3f}")
        
        progress.complete_task("Class weight analysis", True, f"Effectiveness: {overall_effectiveness:.3f}")
        return results
        
    except Exception as e:
        progress.complete_task("Class weight analysis", False, str(e))
        raise

def evaluate_per_context_performance(model, scaler, X, y, context_data, logger, progress):
    """Evaluate model performance across different network contexts."""
    if context_data is None:
        progress.complete_task("Context evaluation", False, "No context data available")
        return None
        
    progress.start_stage("Per-Context Performance Analysis")
    
    results = {}
    
    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        contexts = context_data.unique()
        logger.info(f"🌐 Network Context Performance Analysis ({len(contexts)} contexts):")
        
        context_results = {}
        
        for context in contexts:
            context_mask = (context_data == context)
            if context_mask.sum() < 10:  # Skip contexts with too few samples
                continue
                
            y_context = y[context_mask]
            y_pred_context = y_pred[context_mask]
            
            # Calculate metrics for this context
            accuracy = accuracy_score(y_context, y_pred_context)
            
            # Class distribution in this context
            context_class_dist = y_context.value_counts().sort_index()
            
            context_results[context] = {
                'accuracy': accuracy,
                'sample_count': len(y_context),
                'class_distribution': context_class_dist.to_dict()
            }
            
            logger.info(f"  {context}: {accuracy:.3f} accuracy ({len(y_context):,} samples)")
        
        results['context_results'] = context_results
        
        # Identify best and worst performing contexts
        if context_results:
            best_context = max(context_results.keys(), key=lambda x: context_results[x]['accuracy'])
            worst_context = min(context_results.keys(), key=lambda x: context_results[x]['accuracy'])
            
            logger.info(f"\n🏆 Best performing context: {best_context} ({context_results[best_context]['accuracy']:.3f})")
            logger.info(f"⚠️  Worst performing context: {worst_context} ({context_results[worst_context]['accuracy']:.3f})")
            
            results['best_context'] = best_context
            results['worst_context'] = worst_context
        
        progress.complete_task("Context evaluation", True, f"{len(context_results)} contexts analyzed")
        return results
        
    except Exception as e:
        progress.complete_task("Context evaluation", False, str(e))
        raise

def analyze_feature_importance(model, feature_names, logger, progress):
    """Analyze and interpret feature importance with WiFi domain knowledge."""
    progress.start_stage("Feature Importance Analysis")
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create feature importance analysis
        feature_analysis = []
        
        logger.info("🔝 Top 15 Most Important Features:")
        logger.info(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'WiFi Relevance':<30}")
        logger.info("-" * 80)
        
        # WiFi domain interpretations
        wifi_interpretations = {
            'lastSnr': 'Signal quality - critical for rate selection',
            'shortSuccRatio': 'Recent transmission success - key performance indicator',
            'snrVariance': 'Signal stability - affects rate adaptation decisions',
            'severity': 'Network condition severity - emergency vs normal operation',
            'phyRate': 'Current rate - influences next rate decision',
            'recentThroughputTrend': 'Throughput direction - increasing or decreasing',
            'conservativeFactor': 'Risk assessment - conservative vs aggressive adaptation',
            'consecSuccess': 'Success streak - indicates stable good conditions',
            'optimalRateDistance': 'Distance from theoretical optimum',
            'consecFailure': 'Failure streak - indicates poor conditions',
            'snrFast': 'Fast SNR estimate - immediate signal assessment',
            'medSuccRatio': 'Medium-term success rate - stability indicator',
            'aggressiveFactor': 'Aggressiveness in rate increases',
            'rateStabilityScore': 'How stable the current rate selection is',
            'snrSlow': 'Slow SNR estimate - long-term signal trend'
        }
        
        for i, idx in enumerate(indices[:15]):
            feature_name = feature_names[idx]
            importance = importances[idx]
            interpretation = wifi_interpretations.get(feature_name, 'WiFi protocol feature')
            
            feature_analysis.append({
                'rank': i + 1,
                'feature': feature_name,
                'importance': importance,
                'interpretation': interpretation
            })
            
            logger.info(f"{i+1:<6} {feature_name:<25} {importance:<12.4f} {interpretation:<30}")
        
        # Analyze feature categories
        snr_features = [f for f in feature_names if 'snr' in f.lower()]
        success_features = [f for f in feature_names if 'success' in f.lower() or 'ratio' in f.lower()]
        stability_features = [f for f in feature_names if 'stability' in f.lower() or 'variance' in f.lower()]
        
        category_importance = {
            'SNR/Signal Quality': sum(importances[feature_names.index(f)] for f in snr_features if f in feature_names),
            'Success/Performance': sum(importances[feature_names.index(f)] for f in success_features if f in feature_names),
            'Stability/Variance': sum(importances[feature_names.index(f)] for f in stability_features if f in feature_names)
        }
        
        logger.info(f"\n📊 Feature Category Importance:")
        for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {importance:.3f}")
        
        results = {
            'feature_analysis': feature_analysis,
            'category_importance': category_importance,
            'all_importances': dict(zip(feature_names, importances))
        }
        
        progress.complete_task("Feature importance analysis", True, f"Top feature: {feature_names[indices[0]]}")
        return results
        
    except Exception as e:
        progress.complete_task("Feature importance analysis", False, str(e))
        raise

def evaluate_edge_cases(model, scaler, X, y, logger, progress):
    """Evaluate model performance on edge cases and challenging scenarios."""
    progress.start_stage("Edge Case Performance Evaluation")
    
    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        edge_case_results = {}
        
        # Edge Case 1: Very low SNR scenarios
        low_snr_mask = X['lastSnr'] < 10
        if low_snr_mask.sum() > 0:
            edge_acc = accuracy_score(y[low_snr_mask], y_pred[low_snr_mask])
            edge_case_results['low_snr'] = {
                'accuracy': edge_acc,
                'sample_count': low_snr_mask.sum(),
                'description': 'Very low SNR (< 10 dB) scenarios'
            }
            logger.info(f"📉 Low SNR edge cases: {edge_acc:.3f} accuracy ({low_snr_mask.sum():,} samples)")
        
        # Edge Case 2: High variance/unstable conditions
        if 'snrVariance' in X.columns:
            high_variance_mask = X['snrVariance'] > X['snrVariance'].quantile(0.95)
            if high_variance_mask.sum() > 0:
                edge_acc = accuracy_score(y[high_variance_mask], y_pred[high_variance_mask])
                edge_case_results['high_variance'] = {
                    'accuracy': edge_acc,
                    'sample_count': high_variance_mask.sum(),
                    'description': 'High variance/unstable signal conditions'
                }
                logger.info(f"📊 High variance edge cases: {edge_acc:.3f} accuracy ({high_variance_mask.sum():,} samples)")
        
        # Edge Case 3: Consecutive failure scenarios
        if 'consecFailure' in X.columns:
            high_failure_mask = X['consecFailure'] >= 3
            if high_failure_mask.sum() > 0:
                edge_acc = accuracy_score(y[high_failure_mask], y_pred[high_failure_mask])
                edge_case_results['consecutive_failures'] = {
                    'accuracy': edge_acc,
                    'sample_count': high_failure_mask.sum(),
                    'description': 'Consecutive failure scenarios (≥3 failures)'
                }
                logger.info(f"❌ Consecutive failure edge cases: {edge_acc:.3f} accuracy ({high_failure_mask.sum():,} samples)")
        
        # Edge Case 4: Very low success ratio
        if 'shortSuccRatio' in X.columns:
            low_success_mask = X['shortSuccRatio'] < 0.5
            if low_success_mask.sum() > 0:
                edge_acc = accuracy_score(y[low_success_mask], y_pred[low_success_mask])
                edge_case_results['low_success'] = {
                    'accuracy': edge_acc,
                    'sample_count': low_success_mask.sum(),
                    'description': 'Low success ratio (< 50%) scenarios'
                }
                logger.info(f"📉 Low success ratio edge cases: {edge_acc:.3f} accuracy ({low_success_mask.sum():,} samples)")
        
        # Overall edge case assessment
        if edge_case_results:
            avg_edge_accuracy = np.mean([result['accuracy'] for result in edge_case_results.values()])
            total_edge_samples = sum([result['sample_count'] for result in edge_case_results.values()])
            
            logger.info(f"\n🔍 Edge Case Summary:")
            logger.info(f"  Average edge case accuracy: {avg_edge_accuracy:.3f}")
            logger.info(f"  Total edge case samples: {total_edge_samples:,}")
            
            edge_case_results['summary'] = {
                'average_accuracy': avg_edge_accuracy,
                'total_samples': total_edge_samples
            }
        
        progress.complete_task("Edge case evaluation", True, f"{len(edge_case_results)} edge case types analyzed")
        return edge_case_results
        
    except Exception as e:
        progress.complete_task("Edge case evaluation", False, str(e))
        raise

def compare_with_oracle_strategies(model, scaler, X, y, oracle_data, logger, progress):
    """Compare model performance with oracle strategies."""
    if oracle_data is None:
        progress.complete_task("Oracle comparison", False, "No oracle data available")
        return None
        
    progress.start_stage("Oracle Strategy Comparison")
    
    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        model_accuracy = accuracy_score(y, y_pred)
        
        oracle_comparisons = {}
        
        logger.info("🎯 Model vs Oracle Strategy Comparison:")
        logger.info(f"Model accuracy: {model_accuracy:.4f}")
        
        for oracle_col in oracle_data.columns:
            if oracle_col in oracle_data.columns:
                oracle_y = oracle_data[oracle_col].astype(int)
                
                # Remove samples where oracle has NaN
                valid_mask = oracle_y.notna()
                if valid_mask.sum() == 0:
                    continue
                
                oracle_accuracy = accuracy_score(y[valid_mask], oracle_y[valid_mask])
                accuracy_diff = model_accuracy - oracle_accuracy
                
                oracle_comparisons[oracle_col] = {
                    'oracle_accuracy': oracle_accuracy,
                    'model_accuracy': model_accuracy,
                    'accuracy_difference': accuracy_diff,
                    'sample_count': valid_mask.sum()
                }
                
                comparison_status = "BETTER" if accuracy_diff > 0 else "WORSE" if accuracy_diff < -0.01 else "SIMILAR"
                logger.info(f"  vs {oracle_col}: {oracle_accuracy:.4f} "
                           f"(diff: {accuracy_diff:+.4f}) - {comparison_status}")
        
        progress.complete_task("Oracle comparison", True, f"{len(oracle_comparisons)} oracle strategies compared")
        return oracle_comparisons
        
    except Exception as e:
        progress.complete_task("Oracle comparison", False, str(e))
        raise

# ================== VISUALIZATION AND REPORTING ==================
def create_visualizations(evaluation_results, logger, progress):
    """Create comprehensive visualizations of evaluation results."""
    progress.start_stage("Visualization Generation")
    
    try:
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig_size = (15, 12)
        
        # Create a comprehensive figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle('WiFi Rate Adaptation Model - Comprehensive Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        if 'overall_results' in evaluation_results and 'confusion_matrix' in evaluation_results['overall_results']:
            cm = evaluation_results['overall_results']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_xlabel('Predicted Rate Class')
            axes[0,0].set_ylabel('Actual Rate Class')
        
        # 2. Per-Class Performance
        if 'overall_results' in evaluation_results and 'classification_report' in evaluation_results['overall_results']:
            class_report = evaluation_results['overall_results']['classification_report']
            classes = [str(i) for i in range(8) if str(i) in class_report]
            f1_scores = [class_report[cls]['f1-score'] for cls in classes]
            
            bars = axes[0,1].bar(classes, f1_scores, color='skyblue', alpha=0.7)
            axes[0,1].set_title('F1-Score by Rate Class')
            axes[0,1].set_xlabel('Rate Class')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Feature Importance
        if 'feature_importance' in evaluation_results:
            feature_data = evaluation_results['feature_importance']['feature_analysis'][:10]
            features = [item['feature'] for item in feature_data]
            importances = [item['importance'] for item in feature_data]
            
            y_pos = np.arange(len(features))
            axes[0,2].barh(y_pos, importances, color='lightcoral', alpha=0.7)
            axes[0,2].set_yticks(y_pos)
            axes[0,2].set_yticklabels(features)
            axes[0,2].set_xlabel('Importance')
            axes[0,2].set_title('Top 10 Feature Importance')
            axes[0,2].invert_yaxis()
        
        # 4. Class Weight Effectiveness
        if 'class_weight_effectiveness' in evaluation_results:
            effectiveness = evaluation_results['class_weight_effectiveness']['effectiveness_scores']
            classes = list(effectiveness.keys())
            recalls = [effectiveness[cls]['recall'] for cls in classes]
            weights = [effectiveness[cls]['weight'] for cls in classes]
            
            axes[1,0].scatter(weights, recalls, alpha=0.7, s=100)
            axes[1,0].set_xlabel('Class Weight')
            axes[1,0].set_ylabel('Recall')
            axes[1,0].set_title('Class Weight vs Recall')
            axes[1,0].set_xscale('log')
            
            # Add class labels
            for cls, weight, recall in zip(classes, weights, recalls):
                axes[1,0].annotate(f'C{cls}', (weight, recall), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8)
        
        # 5. Context Performance (if available)
        if 'context_results' in evaluation_results and evaluation_results['context_results']:
            context_data = evaluation_results['context_results']['context_results']
            contexts = list(context_data.keys())
            accuracies = [context_data[ctx]['accuracy'] for ctx in contexts]
            
            axes[1,1].bar(range(len(contexts)), accuracies, color='lightgreen', alpha=0.7)
            axes[1,1].set_xticks(range(len(contexts)))
            axes[1,1].set_xticklabels([ctx.replace('_', '\n') for ctx in contexts], rotation=45, fontsize=8)
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_title('Performance by Network Context')
            axes[1,1].set_ylim(0, 1)
        
        # 6. Edge Case Performance
        if 'edge_case_results' in evaluation_results and evaluation_results['edge_case_results']:
            edge_data = evaluation_results['edge_case_results']
            edge_types = [key for key in edge_data.keys() if key != 'summary']
            if edge_types:
                edge_accuracies = [edge_data[edge_type]['accuracy'] for edge_type in edge_types]
                
                axes[1,2].bar(range(len(edge_types)), edge_accuracies, color='orange', alpha=0.7)
                axes[1,2].set_xticks(range(len(edge_types)))
                axes[1,2].set_xticklabels([edge_type.replace('_', '\n') for edge_type in edge_types], 
                                         rotation=45, fontsize=8)
                axes[1,2].set_ylabel('Accuracy')
                axes[1,2].set_title('Edge Case Performance')
                axes[1,2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save the visualization
        viz_file = OUTPUT_DIR / "comprehensive_evaluation_results.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        progress.complete_task("Create visualizations", True, f"Saved to {viz_file}")
        
    except Exception as e:
        progress.complete_task("Create visualizations", False, str(e))

def generate_comprehensive_report(evaluation_results, artifacts, data_info, logger, progress):
    """Generate a comprehensive evaluation report."""
    progress.start_stage("Report Generation")
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file = OUTPUT_DIR / "comprehensive_evaluation_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# WiFi Rate Adaptation Model - Comprehensive Evaluation Report\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Author:** ahmedjk34\n")
            f.write(f"**Pipeline Stage:** Step 5 - Model Evaluation\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            if 'overall_results' in evaluation_results:
                overall_acc = evaluation_results['overall_results']['accuracy']
                f.write(f"- **Overall Model Accuracy:** {overall_acc:.4f} ({overall_acc*100:.1f}%)\n")
            
            if 'class_weight_effectiveness' in evaluation_results:
                effectiveness = evaluation_results['class_weight_effectiveness']['overall_effectiveness']
                f.write(f"- **Class Weight Effectiveness:** {effectiveness:.3f}\n")
            
            f.write(f"- **Dataset Size:** {len(data_info['y']):,} samples\n")
            f.write(f"- **Feature Count:** {len(data_info['X'].columns)} features\n")
            f.write(f"- **Rate Classes:** 8 (IEEE 802.11g rates)\n\n")
            
            # Model Architecture
            f.write("## 🏗️ Model Architecture\n\n")
            model = artifacts['model']
            f.write(f"- **Algorithm:** Random Forest Classifier\n")
            f.write(f"- **Estimators:** {model.n_estimators}\n")
            f.write(f"- **Max Depth:** {model.max_depth}\n")
            f.write(f"- **Class Weights:** {'Applied' if artifacts['class_weights'] else 'Not Applied'}\n\n")
            
            # Dataset Characteristics
            f.write("## 📊 Dataset Characteristics\n\n")
            f.write("### Class Distribution\n\n")
            f.write("| Class | Rate | Samples | Percentage |\n")
            f.write("|-------|------|---------|------------|\n")
            
            class_dist = data_info['class_distribution']
            total_samples = len(data_info['y'])
            
            for class_id in range(8):
                count = class_dist.get(class_id, 0)
                pct = (count / total_samples) * 100
                rate = WIFI_RATES[class_id]["rate"]
                f.write(f"| {class_id} | {rate} | {count:,} | {pct:.1f}% |\n")
            
            f.write("\n")
            
            # Overall Performance
            if 'overall_results' in evaluation_results:
                f.write("## 🎯 Overall Performance\n\n")
                results = evaluation_results['overall_results']
                
                f.write("### Key Metrics\n\n")
                f.write(f"- **Accuracy:** {results['accuracy']:.4f}\n")
                if 'macro_avg' in results:
                    f.write(f"- **Macro Average F1:** {results['macro_avg']['f1-score']:.4f}\n")
                    f.write(f"- **Weighted Average F1:** {results['weighted_avg']['f1-score']:.4f}\n")
                
                f.write("\n### Per-Class Performance\n\n")
                f.write("| Class | Rate | Precision | Recall | F1-Score | Support |\n")
                f.write("|-------|------|-----------|--------|----------|--------|\n")
                
                class_report = results['classification_report']
                for i in range(8):
                    if str(i) in class_report:
                        metrics = class_report[str(i)]
                        rate = WIFI_RATES[i]["rate"]
                        f.write(f"| {i} | {rate} | {metrics['precision']:.3f} | "
                               f"{metrics['recall']:.3f} | {metrics['f1-score']:.3f} | "
                               f"{int(metrics['support'])} |\n")
                
                f.write("\n")
            
            # Class Weight Effectiveness
            if 'class_weight_effectiveness' in evaluation_results:
                f.write("## ⚖️ Class Weight Effectiveness\n\n")
                effectiveness_data = evaluation_results['class_weight_effectiveness']['effectiveness_scores']
                
                f.write("| Class | Weight | Actual% | Recall | Status |\n")
                f.write("|-------|--------|---------|--------|--------|\n")
                
                for class_id in range(8):
                    if class_id in effectiveness_data:
                        data = effectiveness_data[class_id]
                        f.write(f"| {class_id} | {data['weight']:.1f} | {data['actual_pct']:.1f}% | "
                               f"{data['recall']:.3f} | {data['status']} |\n")
                
                f.write("\n")
            
            # Feature Importance
            if 'feature_importance' in evaluation_results:
                f.write("## 🔝 Feature Importance Analysis\n\n")
                feature_data = evaluation_results['feature_importance']['feature_analysis'][:15]
                
                f.write("| Rank | Feature | Importance | WiFi Relevance |\n")
                f.write("|------|---------|------------|----------------|\n")
                
                for item in feature_data:
                    f.write(f"| {item['rank']} | {item['feature']} | {item['importance']:.4f} | "
                           f"{item['interpretation']} |\n")
                
                f.write("\n")
            
            # Context Performance
            if 'context_results' in evaluation_results and evaluation_results['context_results']:
                f.write("## 🌐 Network Context Performance\n\n")
                context_data = evaluation_results['context_results']['context_results']
                
                f.write("| Context | Accuracy | Samples |\n")
                f.write("|---------|----------|--------|\n")
                
                for context, data in context_data.items():
                    f.write(f"| {context} | {data['accuracy']:.3f} | {data['sample_count']:,} |\n")
                
                f.write("\n")
            
            # Edge Case Performance
            if 'edge_case_results' in evaluation_results:
                f.write("## 🔍 Edge Case Performance\n\n")
                edge_data = evaluation_results['edge_case_results']
                
                for edge_type, data in edge_data.items():
                    if edge_type != 'summary':
                        f.write(f"### {edge_type.replace('_', ' ').title()}\n")
                        f.write(f"- **Description:** {data['description']}\n")
                        f.write(f"- **Accuracy:** {data['accuracy']:.3f}\n")
                        f.write(f"- **Sample Count:** {data['sample_count']:,}\n\n")
            
            # Oracle Comparison
            if 'oracle_comparison' in evaluation_results and evaluation_results['oracle_comparison']:
                f.write("## 🎯 Oracle Strategy Comparison\n\n")
                oracle_data = evaluation_results['oracle_comparison']
                
                f.write("| Strategy | Oracle Accuracy | Model Accuracy | Difference | Status |\n")
                f.write("|----------|----------------|----------------|------------|--------|\n")
                
                for strategy, data in oracle_data.items():
                    diff = data['accuracy_difference']
                    status = "BETTER" if diff > 0 else "WORSE" if diff < -0.01 else "SIMILAR"
                    f.write(f"| {strategy} | {data['oracle_accuracy']:.4f} | "
                           f"{data['model_accuracy']:.4f} | {diff:+.4f} | {status} |\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations and Next Steps\n\n")
            f.write("### Model Performance\n")
            
            if 'overall_results' in evaluation_results:
                accuracy = evaluation_results['overall_results']['accuracy']
                if accuracy > 0.95:
                    f.write("- ✅ **Excellent Performance:** Model achieves production-ready accuracy\n")
                elif accuracy > 0.90:
                    f.write("- ✅ **Good Performance:** Model suitable for deployment with monitoring\n")
                else:
                    f.write("- ⚠️ **Moderate Performance:** Consider model improvements before deployment\n")
            
            f.write("\n### Class Imbalance Handling\n")
            if 'class_weight_effectiveness' in evaluation_results:
                effectiveness = evaluation_results['class_weight_effectiveness']['overall_effectiveness']
                if effectiveness > 0.7:
                    f.write("- ✅ **Effective Class Weights:** Successfully handling imbalanced data\n")
                else:
                    f.write("- ⚠️ **Improve Class Weights:** Consider adjusting weights for better balance\n")
            
            f.write("\n### Deployment Readiness\n")
            f.write("- 🔄 **Real-time Testing:** Deploy in controlled ns-3 simulation environment\n")
            f.write("- 📊 **Performance Monitoring:** Implement comprehensive logging and metrics collection\n")
            f.write("- 🔧 **Continuous Learning:** Plan for model updates based on real-world performance\n")
            f.write("- 🛡️ **Edge Case Handling:** Implement fallback strategies for challenging scenarios\n\n")
            
            # Technical Details
            f.write("## 🔧 Technical Details\n\n")
            f.write("### Files Generated\n")
            f.write("- `comprehensive_evaluation_report.md` - This detailed report\n")
            f.write("- `comprehensive_evaluation_results.png` - Visualization dashboard\n")
            f.write("- `evaluation_log_[timestamp].log` - Detailed execution log\n\n")
            
            f.write("### Model Artifacts\n")
            f.write(f"- **Model File:** `{MODEL_FILE}`\n")
            f.write(f"- **Scaler File:** `{SCALER_FILE}`\n")
            f.write(f"- **Class Weights:** `{CLASS_WEIGHTS_FILE}`\n\n")
            
        progress.complete_task("Generate report", True, f"Report saved to {report_file}")
        logger.info(f"📄 Comprehensive report generated: {report_file}")
        
    except Exception as e:
        progress.complete_task("Generate report", False, str(e))
        raise

# ================== MAIN EXECUTION ==================
def main():
    """Main execution function for comprehensive model evaluation."""
    logger = setup_logging_and_output()
    progress = EvaluationProgress(logger)
    
    try:
        logger.info("🚀 Starting comprehensive WiFi rate adaptation model evaluation...")
        
        # Load data and artifacts
        data_info = load_evaluation_data(logger, progress)
        artifacts = load_model_artifacts(logger, progress)
        
        # Prepare data splits for evaluation
        X, y = data_info['X'], data_info['y']
        
        # Use a portion of data for evaluation (to simulate real test set)
        from sklearn.model_selection import train_test_split
        _, X_eval, _, y_eval = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
        
        # Get corresponding context and oracle data
        eval_indices = X_eval.index
        context_eval = data_info['context_data'].loc[eval_indices] if data_info['context_data'] is not None else None
        oracle_eval = data_info['oracle_data'].loc[eval_indices] if data_info['oracle_data'] is not None else None
        
        # Perform comprehensive evaluation
        evaluation_results = {}
        
        # 1. Overall Performance Evaluation
        overall_results = perform_overall_evaluation(
            artifacts['model'], artifacts['scaler'], X_eval, y_eval, logger, progress
        )
        evaluation_results['overall_results'] = overall_results
        
        # 2. Class Weight Effectiveness Analysis
        class_weight_results = evaluate_class_weight_effectiveness(
            artifacts['model'], artifacts['scaler'], X_eval, y_eval, 
            artifacts['class_weights'], data_info['class_distribution'], logger, progress
        )
        evaluation_results['class_weight_effectiveness'] = class_weight_results
        
        # 3. Feature Importance Analysis
        feature_importance_results = analyze_feature_importance(
            artifacts['model'], FEATURE_COLS, logger, progress
        )
        evaluation_results['feature_importance'] = feature_importance_results
        
        # 4. Per-Context Performance (if available)
        if context_eval is not None:
            context_results = evaluate_per_context_performance(
                artifacts['model'], artifacts['scaler'], X_eval, y_eval, context_eval, logger, progress
            )
            evaluation_results['context_results'] = context_results
        
        # 5. Edge Case Performance
        edge_case_results = evaluate_edge_cases(
            artifacts['model'], artifacts['scaler'], X_eval, y_eval, logger, progress
        )
        evaluation_results['edge_case_results'] = edge_case_results
        
        # 6. Oracle Strategy Comparison (if available)
        if oracle_eval is not None:
            oracle_comparison = compare_with_oracle_strategies(
                artifacts['model'], artifacts['scaler'], X_eval, y_eval, oracle_eval, logger, progress
            )
            evaluation_results['oracle_comparison'] = oracle_comparison
        
        # 7. Create Visualizations
        create_visualizations(evaluation_results, logger, progress)
        
        # 8. Generate Comprehensive Report
        generate_comprehensive_report(evaluation_results, artifacts, data_info, logger, progress)
        
        # Final Summary
        summary = progress.get_summary()
        logger.info("\n" + "="*80)
        logger.info("🎉 COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"⏱️  Total execution time: {summary['total_time']:.1f} seconds")
        logger.info(f"✅ Completed tasks: {summary['completed_tasks']}")
        logger.info(f"❌ Failed tasks: {summary['failed_tasks']}")
        logger.info(f"📊 Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"📁 Results directory: {OUTPUT_DIR.absolute()}")
        
        if 'overall_results' in evaluation_results:
            final_accuracy = evaluation_results['overall_results']['accuracy']
            logger.info(f"🎯 Final Model Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")
        
        logger.info("📄 Generated outputs:")
        logger.info("  - comprehensive_evaluation_report.md")
        logger.info("  - comprehensive_evaluation_results.png") 
        logger.info("  - evaluation_log_[timestamp].log")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)