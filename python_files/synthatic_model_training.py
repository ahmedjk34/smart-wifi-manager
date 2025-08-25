#!/usr/bin/env python3
"""
Enhanced ML Training Pipeline for WiFi Rate Adaptation
Addresses poor performance with advanced techniques
"""

import pandas as pd
import numpy as np
import joblib
import logging
import time
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class EnhancedWiFiMLTrainer:
    """Advanced ML training for WiFi rate adaptation"""
    
    def __init__(self, dataset_path, output_dir="models_enhanced"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Enhanced feature engineering
        self.feature_engineering_enabled = True
        self.hyperparameter_tuning = True
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"enhanced_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("🚀 Enhanced WiFi ML Training Pipeline Started")
        return logger
    
    def load_and_analyze_data(self):
        """Load data with comprehensive analysis"""
        self.logger.info(f"📂 Loading dataset from {self.dataset_path}")
        
        # Load in chunks if large
        try:
            df = pd.read_csv(self.dataset_path)
            self.logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        except MemoryError:
            self.logger.warning("🔄 Large dataset detected, loading in chunks...")
            chunks = []
            for chunk in pd.read_csv(self.dataset_path, chunksize=100000):
                chunks.append(chunk.sample(frac=0.5))  # Sample 50% to fit memory
            df = pd.concat(chunks, ignore_index=True)
            self.logger.info(f"✅ Loaded sampled dataset: {len(df)} samples")
        
        # Data quality analysis
        self.logger.info("🔍 Analyzing data quality...")
        self.logger.info(f"Missing values: {df.isnull().sum().sum()}")
        self.logger.info(f"Duplicate rows: {df.duplicated().sum()}")
        
        # Feature analysis
        feature_cols = [col for col in df.columns if col not in 
                       ['oracle_best_rateIdx', 'v3_rateIdx', 'distance', 'speed', 
                        'environment', 'scenario', 'num_interferers']]
        
        self.logger.info(f"📊 Feature columns ({len(feature_cols)}): {feature_cols}")
        
        # Label distribution analysis
        oracle_dist = df['oracle_best_rateIdx'].value_counts().sort_index()
        v3_dist = df['v3_rateIdx'].value_counts().sort_index()
        
        self.logger.info(f"📈 Oracle rate distribution:\n{oracle_dist}")
        self.logger.info(f"📈 V3 rate distribution:\n{v3_dist}")
        
        # Check for single-rate bias (main issue with previous dataset)
        oracle_entropy = -sum(p * np.log2(p) for p in (oracle_dist / oracle_dist.sum()) if p > 0)
        v3_entropy = -sum(p * np.log2(p) for p in (v3_dist / v3_dist.sum()) if p > 0)
        
        self.logger.info(f"📊 Oracle label entropy: {oracle_entropy:.3f} (max=3.0)")
        self.logger.info(f"📊 V3 label entropy: {v3_entropy:.3f} (max=3.0)")
        
        if oracle_entropy < 2.0:
            self.logger.warning("⚠️ Low oracle entropy detected - possible rate bias")
        if v3_entropy < 2.0:
            self.logger.warning("⚠️ Low v3 entropy detected - possible rate bias")
        
        return df, feature_cols
    
    def engineer_features(self, df, feature_cols):
        """Advanced feature engineering"""
        if not self.feature_engineering_enabled:
            return df[feature_cols]
        
        self.logger.info("🔧 Engineering advanced features...")
        
        # Create enhanced feature set
        enhanced_df = df[feature_cols].copy()
        
        # SNR-based features
        enhanced_df['snr_margin'] = df['lastSnr'] - 10  # Margin above minimum
        enhanced_df['snr_stability'] = df['snrFast'] / (df['snrVariance'] + 1e-6)
        enhanced_df['snr_trend'] = df['snrFast'] - df['snrSlow']
        
        # Performance-based features
        enhanced_df['success_difference'] = df['shortSuccRatio'] - df['medSuccRatio']
        enhanced_df['reliability_score'] = (df['shortSuccRatio'] * df['confidence']) - df['severity']
        enhanced_df['adaptation_pressure'] = df['consecFailure'] / (df['consecSuccess'] + 1)
        
        # Temporal features
        enhanced_df['timing_efficiency'] = df['T1'] / (df['T2'] + 1e-6)
        enhanced_df['response_delay'] = np.log1p(df['T3'])
        
        # Network load features
        enhanced_df['load_pressure'] = df['offeredLoad'] * (df['queueLen'] + 1) / 100
        enhanced_df['congestion_indicator'] = df['retryCount'] * df['queueLen']
        
        # Mobility features
        enhanced_df['mobility_impact'] = df['mobilityMetric'] * df['snrVariance']
        enhanced_df['channel_dynamics'] = np.sqrt(df['snrVariance'] * df['mobilityMetric'])
        
        # Rate capacity features (physics-based)
        enhanced_df['theoretical_capacity'] = enhanced_df['channelWidth'] * np.log2(1 + 10**(df['lastSnr']/10))
        enhanced_df['efficiency_ratio'] = df['offeredLoad'] / (enhanced_df['theoretical_capacity'] + 1e-6)
        
        self.logger.info(f"✅ Enhanced features: {len(enhanced_df.columns)} (added {len(enhanced_df.columns) - len(feature_cols)})")
        
        return enhanced_df
    
    def prepare_data(self, df, feature_cols, target_col):
        """Prepare data with advanced preprocessing"""
        X = self.engineer_features(df, feature_cols)
        y = df[target_col]
        
        # Handle outliers
        self.logger.info("🔧 Handling outliers...")
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.01)
            Q3 = X[col].quantile(0.99)
            X[col] = X[col].clip(Q1, Q3)
        
        # Time-aware split (important for time-series data)
        self.logger.info("📊 Performing time-aware train/validation/test split...")
        
        # Sort by a proxy for time (using combination of features)
        time_proxy = df['T1'] + df['T2'] + df['T3']
        sort_idx = time_proxy.argsort()
        
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        
        # 70-15-15 split, time-ordered
        n_train = int(0.7 * len(X_sorted))
        n_val = int(0.15 * len(X_sorted))
        
        X_train = X_sorted.iloc[:n_train]
        X_val = X_sorted.iloc[n_train:n_train+n_val]
        X_test = X_sorted.iloc[n_train+n_val:]
        
        y_train = y_sorted.iloc[:n_train]
        y_val = y_sorted.iloc[n_train:n_train+n_val]
        y_test = y_sorted.iloc[n_train+n_val:]
        
        self.logger.info(f"📈 Training: {len(X_train)} samples")
        self.logger.info(f"📊 Validation: {len(X_val)} samples")
        self.logger.info(f"🧪 Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Advanced feature scaling"""
        self.logger.info("⚖️ Scaling features with RobustScaler...")
        
        # Use RobustScaler for better outlier handling
        scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='rf'):
        """Hyperparameter optimization with Optuna"""
        if not self.hyperparameter_tuning:
            return self._get_default_model(model_type)
        
        self.logger.info(f"🎯 Optimizing hyperparameters for {model_type}...")
        
        def objective(trial):
            if model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 6, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 6, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBClassifier(**params)
            
            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        self.logger.info(f"🏆 Best hyperparameters: {study.best_params}")
        self.logger.info(f"🎯 Best CV score: {study.best_value:.4f}")
        
        # Return best model
        if model_type == 'rf':
            return RandomForestClassifier(**study.best_params)
        elif model_type == 'lgb':
            return lgb.LGBMClassifier(**study.best_params)
        elif model_type == 'xgb':
            return xgb.XGBClassifier(**study.best_params)
    
    def _get_default_model(self, model_type):
        """Get default model configurations"""
        if model_type == 'rf':
            return RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        elif model_type == 'lgb':
            return lgb.LGBMClassifier(n_estimators=200, max_depth=15, random_state=42, verbose=-1)
        elif model_type == 'xgb':
            return xgb.XGBClassifier(n_estimators=200, max_depth=15, random_state=42, verbosity=0)
        elif model_type == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, random_state=42)
    
    def train_and_evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
        """Train and comprehensively evaluate model"""
        self.logger.info(f"🚀 Training {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Accuracy scores
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        self.logger.info(f"✅ {model_name} - Validation Accuracy: {val_acc:.4f}")
        self.logger.info(f"✅ {model_name} - Test Accuracy: {test_acc:.4f}")
        self.logger.info(f"⏱️ Training time: {train_time:.2f} seconds")
        
        # Check rate diversity
        pred_diversity = len(np.unique(y_test_pred))
        self.logger.info(f"🔍 {model_name} predicts {pred_diversity}/8 different rates")
        
        if pred_diversity < 4:
            self.logger.warning(f"⚠️ {model_name} shows low rate diversity - possible bias!")
        
        # Detailed classification report
        self.logger.info(f"📋 {model_name} Classification Report:")
        report = classification_report(y_test, y_test_pred, target_names=[f'Rate_{i}' for i in range(8)])
        self.logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        self.logger.info(f"📊 {model_name} Confusion Matrix:\n{cm}")
        
        # Save model
        # model_file = self.output_dir / f"enhanced_{model_name.lower().replace(' ', '_')}_model.joblib"
        model_file = self.output_dir / model_file = self.output_dir / f"enhanced_{model_name.lower().replace(' ', '_')}_model.joblib"


        joblib.dump(model, model_file)
        self.logger.info(f"💾 Model saved to {model_file}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            
            self.logger.info(f"🔝 Top 10 features for {model_name}:")
            for feat, imp in top_features:
                self.logger.info(f"  {feat}: {imp:.4f}")
        
        return {
            'model': model,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'train_time': train_time,
            'rate_diversity': pred_diversity
        }
    
    def train_ensemble_models(self):
        """Train multiple models and create ensemble"""
        self.logger.info("🎯 Starting comprehensive model training...")
        
        # Load and prepare data
        df, feature_cols = self.load_and_analyze_data()
        
        results = {}
        
        # Train for both oracle and v3 targets
        for target in ['oracle_best_rateIdx', 'v3_rateIdx']:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"🎯 Training models for {target}")
            self.logger.info(f"{'='*50}")
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df, feature_cols, target)
            X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_val, X_test)
            
            # Save scaler
            scaler_file = self.output_dir / f"enhanced_scaler_{target}.joblib"
            joblib.dump(scaler, scaler_file)
            
            target_results = {}
            
            # Train multiple model types
            model_configs = [
                ('Random Forest', 'rf'),
                ('LightGBM', 'lgb'),
                ('XGBoost', 'xgb'),
                ('MLP', 'mlp')
            ]
            
            for model_name, model_type in model_configs:
                try:
                    # Optimize and train
                    model = self.optimize_hyperparameters(X_train_scaled, y_train, model_type)
                    
                    # Train and evaluate
                    result = self.train_and_evaluate_model(
                        model, X_train_scaled, y_train, X_val_scaled, y_val, 
                        X_test_scaled, y_test, f"{model_name} ({target})"
                    )
                    
                    target_results[model_name] = result
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model_name} for {target}: {e}")
            
            results[target] = target_results
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 ENHANCED TRAINING SUMMARY REPORT")
        self.logger.info("="*60)
        
        summary_file = self.output_dir / "enhanced_training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Enhanced WiFi ML Training Summary\n")
            f.write("="*50 + "\n\n")
            
            for target, target_results in results.items():
                f.write(f"Target: {target}\n")
                f.write("-" * 30 + "\n")
                
                for model_name, result in target_results.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Validation Accuracy: {result['val_accuracy']:.4f}\n")
                    f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
                    f.write(f"  Training Time: {result['train_time']:.2f}s\n")
                    f.write(f"  Rate Diversity: {result['rate_diversity']}/8\n\n")
                
                # Find best model
                best_model = max(target_results.items(), key=lambda x: x[1]['test_accuracy'])
                f.write(f"🏆 Best model: {best_model[0]} ({best_model[1]['test_accuracy']:.4f})\n\n")
                
                self.logger.info(f"🏆 Best {target} model: {best_model[0]} ({best_model[1]['test_accuracy']:.4f})")
        
        self.logger.info(f"📝 Summary report saved to {summary_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced WiFi ML Training')
    parser.add_argument('--dataset', type=str, default='realistic_wifi_dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, default='models_enhanced',
                       help='Output directory for models')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Disable hyperparameter tuning')
    parser.add_argument('--no-features', action='store_true',
                       help='Disable feature engineering')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EnhancedWiFiMLTrainer(args.dataset, args.output)
    trainer.hyperparameter_tuning = not args.no_tuning
    trainer.feature_engineering_enabled = not args.no_features
    
    # Train models
    results = trainer.train_ensemble_models()
    
    print("\n🎉 Enhanced training completed successfully!")
    print(f"📁 Models saved in: {args.output}")

if __name__ == "__main__":
    main()