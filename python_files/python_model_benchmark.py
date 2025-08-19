"""
Step 4A: Python Benchmarking Script for WiFi Rate Adaptation ML Models

Loads trained models (.joblib), scaler, and test data.
Performs inference for RF, LGB, XGB models on both targets (oracle_best_rateIdx, v3_rateIdx).
Outputs accuracy, confusion matrix, classification report, and per-class metrics for each model.
Optionally supports custom test cases (feature vectors + expected label).

Author: github.com/ahmedjk34
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to test data (should match features used in training) - looking in parent directory
PARENT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PARENT_DIR / "smartv4-ml-ready-cleaned.csv"
TEST_SPLIT_RATIO = 0.1                             # Use final 10% as test
FEATURE_COLS = [
    'lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
    'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
    'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance'
]
LABELS = {
    "oracle": "oracle_best_rateIdx",
    "v3": "v3_rateIdx"
}
MODEL_FILES = {
    "rf_oracle": PARENT_DIR / "step3_rf_oracle_best_rateldx_model.joblib",
    "rf_v3": PARENT_DIR / "step3_rf_v3_rateldx_model.joblib",
    "lgb_oracle": PARENT_DIR / "step3_lgb_oracle_best_rateldx_model.joblib",
    "lgb_v3": PARENT_DIR / "step3_lgb_v3_rateldx_model.joblib",
    "xgb_oracle": PARENT_DIR / "step3_xgb_oracle_best_rateldx_model.joblib",
    "xgb_v3": PARENT_DIR / "step3_xgb_v3_rateldx_model.joblib"
}

SCALER_FILE = PARENT_DIR / "step3_scaler.joblib"
CUSTOM_TEST_CASES_PATH = PARENT_DIR / "custom_test_cases.csv"   # Optional: user-supplied test cases

# --- PROGRESS & ERROR HANDLING UTILS ---
class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.steps_completed = 0
        self.total_steps = 0
        self.errors = []
        
    def set_total_steps(self, total):
        self.total_steps = total
        
    def step_complete(self, step_name, success=True, error_msg=None):
        self.steps_completed += 1
        elapsed = time.time() - self.start_time
        progress = (self.steps_completed / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        status = "✓" if success else "✗"
        print(f"[{status}] Step {self.steps_completed}/{self.total_steps} ({progress:.1f}%) - {step_name} - {elapsed:.1f}s elapsed")
        
        if not success and error_msg:
            self.errors.append(f"{step_name}: {error_msg}")
            print(f"    Error: {error_msg}")
    
    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            'total_time': elapsed,
            'steps_completed': self.steps_completed,
            'total_steps': self.total_steps,
            'success_rate': (self.steps_completed - len(self.errors)) / max(self.steps_completed, 1) * 100,
            'errors': self.errors
        }

# Global progress tracker
progress = ProgressTracker()

def safe_file_check(file_path, file_type="file"):
    """Safely check if file exists with error handling"""
    try:
        if Path(file_path).exists():
            return True, None
        else:
            return False, f"{file_type} not found: {file_path}"
    except Exception as e:
        return False, f"Error checking {file_type}: {str(e)}"

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

# --- LOAD DATA ---
def load_test_data():
    print_header("Loading Test Data")
    try:
        exists, error = safe_file_check(TEST_DATA_PATH, "Test data file")
        if not exists:
            raise FileNotFoundError(error)
            
        df = pd.read_csv(TEST_DATA_PATH, low_memory=False)
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate required columns
        missing_cols = [col for col in FEATURE_COLS + list(LABELS.values()) if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Stratified test split
        test_df = df.sample(frac=TEST_SPLIT_RATIO, random_state=42)
        print(f"Test set size: {test_df.shape[0]} samples")
        
        # Check for missing values in test set
        missing_counts = test_df[FEATURE_COLS].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Warning: Found {missing_counts.sum()} missing values in features")
            
        progress.step_complete("Load test data", True)
        return test_df
        
    except Exception as e:
        progress.step_complete("Load test data", False, str(e))
        raise

def load_custom_test_cases():
    try:
        exists, error = safe_file_check(CUSTOM_TEST_CASES_PATH, "Custom test cases file")
        if not exists:
            print(f"No custom test cases found at {CUSTOM_TEST_CASES_PATH}")
            progress.step_complete("Load custom test cases", True, "No custom cases file found (optional)")
            return None
            
        print_header("Loading Custom Test Cases")
        df = pd.read_csv(CUSTOM_TEST_CASES_PATH)
        
        # Validate columns
        missing_features = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in custom cases: {missing_features}")
            
        print(f"Custom test cases loaded: {df.shape[0]} samples")
        progress.step_complete("Load custom test cases", True)
        return df
        
    except Exception as e:
        progress.step_complete("Load custom test cases", False, str(e))
        print(f"Error loading custom test cases: {str(e)}")
        return None

def load_scaler():
    print_header("Loading Feature Scaler")
    try:
        exists, error = safe_file_check(SCALER_FILE, "Scaler file")
        if not exists:
            raise FileNotFoundError(error)
            
        scaler = joblib.load(SCALER_FILE)
        print(f"Scaler loaded from {SCALER_FILE}")
        progress.step_complete("Load scaler", True)
        return scaler
        
    except Exception as e:
        progress.step_complete("Load scaler", False, str(e))
        raise

def load_models():
    print_header("Loading Trained Models")
    models = {}
    models_loaded = 0
    
    for key, path in MODEL_FILES.items():
        try:
            exists, error = safe_file_check(path, f"Model file {key}")
            if not exists:
                print(f"Warning: {error}. Skipping.")
                continue
                
            models[key] = joblib.load(path)
            print(f"Loaded model: {key} from {path}")
            models_loaded += 1
            
        except Exception as e:
            print(f"Error loading model {key}: {str(e)}")
            continue
    
    if models_loaded == 0:
        progress.step_complete("Load models", False, "No models could be loaded")
        raise RuntimeError("No models could be loaded successfully")
    else:
        progress.step_complete("Load models", True, f"Loaded {models_loaded}/{len(MODEL_FILES)} models")
    
    return models

# --- EVALUATION ---
def evaluate_model(model, scaler, X, y, label_name, model_name):
    print_header(f"Evaluating {model_name} ({label_name})")
    try:
        # Validate input data
        if X.empty or y.empty:
            raise ValueError("Empty input data")
            
        if len(X) != len(y):
            raise ValueError(f"Feature-label length mismatch: {len(X)} vs {len(y)}")
        
        # Check for missing values
        if X.isnull().sum().sum() > 0:
            print(f"Warning: Found {X.isnull().sum().sum()} missing values in features, filling with 0")
            X = X.fillna(0)
        
        print(f"Evaluating on {len(X)} samples...")
        
        # Transform features with progress indication
        print("Scaling features...")
        X_scaled = scaler.transform(X)
        
        # Make predictions with progress indication
        print("Making predictions...")
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        cm = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        print("Classification Report:")
        report = classification_report(y, y_pred, digits=3, zero_division=0)
        print(report)
        
        # Per-class metrics for summary
        class_metrics = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        return acc, class_metrics
        
    except Exception as e:
        print(f"Error evaluating model {model_name}: {str(e)}")
        raise

def benchmark_all_models(test_df, scaler, models):
    results = []
    total_combinations = len(LABELS) * 3  # 3 model types (rf, lgb, xgb)
    current_combination = 0
    
    print_header("Starting Model Benchmarking")
    print(f"Total model-label combinations to evaluate: {total_combinations}")
    
    for label_key, label_col in LABELS.items():
        try:
            X = test_df[FEATURE_COLS]
            y = test_df[label_col]
            
            # Check label distribution
            label_counts = y.value_counts()
            print(f"\nLabel distribution for {label_col}:")
            print(label_counts)
            
            for model_tag in ["rf", "lgb", "xgb"]:
                current_combination += 1
                model_key = f"{model_tag}_{label_key}"
                
                print(f"\n[{current_combination}/{total_combinations}] Processing {model_key}...")
                
                if model_key not in models:
                    print(f"Model {model_key} not loaded, skipping.")
                    progress.step_complete(f"Evaluate {model_key}", False, "Model not loaded")
                    continue
                
                try:
                    acc, metrics = evaluate_model(models[model_key], scaler, X, y, label_col, model_key)
                    results.append({
                        "model": model_key,
                        "label": label_col,
                        "accuracy": acc,
                        "metrics": metrics
                    })
                    progress.step_complete(f"Evaluate {model_key}", True)
                    
                except Exception as e:
                    progress.step_complete(f"Evaluate {model_key}", False, str(e))
                    print(f"Failed to evaluate {model_key}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing label {label_key}: {str(e)}")
            continue
    
    return results

def benchmark_custom_cases(df_cases, scaler, models):
    print_header("Benchmarking Custom Test Cases")
    
    if df_cases is None or df_cases.empty:
        print("No custom test cases to process")
        return []
    
    X = df_cases[FEATURE_COLS]
    results = []
    
    for label_key, label_col in LABELS.items():
        if label_col not in df_cases.columns:
            print(f"Label column {label_col} not found in custom cases, skipping")
            continue
            
        y = df_cases[label_col]
        
        for model_tag in ["rf", "lgb", "xgb"]:
            model_key = f"{model_tag}_{label_key}"
            if model_key not in models:
                continue
                
            try:
                print_header(f"Custom Cases: {model_key}")
                X_scaled = scaler.transform(X)
                y_pred = models[model_key].predict(X_scaled)
                print(f"Predictions: {y_pred}")
                
                acc = accuracy_score(y, y_pred)
                print(f"Accuracy: {acc:.4f}")
                
                cm = confusion_matrix(y, y_pred)
                print("Confusion Matrix:")
                print(cm)
                
                report = classification_report(y, y_pred, digits=3, zero_division=0)
                print("Classification Report:")
                print(report)
                
                results.append({
                    "model": model_key,
                    "label": label_col,
                    "accuracy": acc
                })
                
            except Exception as e:
                print(f"Error evaluating custom cases for {model_key}: {str(e)}")
                continue
    
    return results

def save_benchmark_results(results, out_path=None):
    print_header("Saving Benchmark Results")
    try:
        # Save in the same directory as the script by default
        if out_path is None:
            out_path = Path(__file__).resolve().parent / "step4a_benchmark_results.md"
        if not results:
            print("No results to save")
            return
            
        with open(out_path, "w") as f:
            f.write("# Step 4A Model Benchmarking Results\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total models evaluated: {len(results)}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Model | Label | Accuracy |\n")
            f.write("|-------|-------|----------|\n")
            for res in results:
                f.write(f"| {res['model']} | {res['label']} | {res['accuracy']:.4f} |\n")
            f.write("\n")
            
            # Detailed results
            for res in results:
                f.write(f"## Model: {res['model']} | Label: {res['label']}\n")
                f.write(f"- Accuracy: {res['accuracy']:.4f}\n")
                f.write("### Per-class metrics:\n")
                for k, v in res['metrics'].items():
                    if isinstance(v, dict):
                        precision = v.get('precision', 0)
                        recall = v.get('recall', 0)
                        f1 = v.get('f1-score', 0)
                        f.write(f"  - Class {k}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n")
                f.write("\n")
        
        print(f"Results saved to {out_path}")
        progress.step_complete("Save results", True)
        
    except Exception as e:
        progress.step_complete("Save results", False, str(e))
        print(f"Error saving results: {str(e)}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print_header("WiFi Rate Adaptation ML Model Benchmarking")
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate total steps for progress tracking
    total_steps = 4  # load_test_data, load_scaler, load_models, save_results
    total_steps += 1  # load_custom_test_cases
    progress.set_total_steps(total_steps)
    
    try:
        # Load data and models
        test_df = load_test_data()
        scaler = load_scaler()
        models = load_models()
        
        # Benchmark full test set
        results = benchmark_all_models(test_df, scaler, models)
        save_benchmark_results(results)
        
        # Benchmark custom test cases if available
        custom_cases_df = load_custom_test_cases()
        if custom_cases_df is not None:
            custom_results = benchmark_custom_cases(custom_cases_df, scaler, models)
        
        # Final summary
        summary = progress.get_summary()
        print_header("Benchmarking Complete - Summary")
        print(f"Total execution time: {summary['total_time']:.1f} seconds")
        print(f"Steps completed: {summary['steps_completed']}/{summary['total_steps']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        if summary['errors']:
            print(f"Errors encountered: {len(summary['errors'])}")
            for error in summary['errors']:
                print(f"  - {error}")
        else:
            print("All steps completed successfully!")
            
        print(f"Results processed: {len(results)} model evaluations")
        
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("Benchmarking terminated.")
        sys.exit(1)