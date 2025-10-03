#!/usr/bin/env python3
"""
Enhanced ML Inference Server - Production-ready WiFi rate adaptation inference
🚀 FULLY UPDATED FOR PHASE 1A (15 safe features, oracle_aggressive default)

CRITICAL UPDATES (2025-10-02 20:18:13 UTC):
- PHASE 1A: Now expects 15 features (was 9)
- Added 6 new features: retryRate, frameErrorRate, channelBusyRatio, 
  recentRateAvg, rateStability, sinceLastChange
- Updated feature validation and clamping for 15 features
- Auto-discovery still works (backward compatible with 9-feature models)

Author: ahmedjk34 (https://github.com/ahmedjk34)
Date: 2025-10-02 20:18:13 UTC
Usage: python3 python_files/6a_enhanced_ml_inference_server.py
"""

import json
import time
import argparse
import warnings
import joblib
import numpy as np
import socket
import threading
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import traceback

warnings.filterwarnings("ignore")

# ================== CONFIGURATION CLASSES ==================
@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_path: str
    scaler_path: str
    description: str = ""
    features_count: int = 14  # 🚀 FIXED: 14 safe features (Phase 1A)
    rate_classes: int = 8

@dataclass
class ServerConfig:
    """Main server configuration."""
    port: int = 8765
    host: str = "localhost"
    max_connections: int = 100
    socket_timeout: float = 1.0
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_monitoring: bool = True
    monitoring_window: int = 1000


    
# ================== FEATURE DEFINITIONS ==================
class WiFiFeatures:
    """WiFi feature definitions and validation - 14 SAFE FEATURES (PHASE 1B)."""
    
    # 🚀 PHASE 1B: 14 safe features matching File 4 training pipeline
    FEATURE_NAMES = [
        # SNR features (7)
        "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
        "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
        
        # Network state (1 - removed channelWidth)
        "mobilityMetric",
        
        # Phase 1A features (2 - removed channelBusyRatio)
        "retryRate",          # Retry rate (past performance)
        "frameErrorRate",     # Error rate (PHY feedback)
        
        # 🚀 PHASE 1B: NEW FEATURES (4)
        "rssiVariance",       # RSSI variance (signal stability)
        "interferenceLevel",  # Interference level (collision tracking)
        "distanceMetric",     # Distance metric (from scenario)
        "avgPacketSize"       # Average packet size (traffic characteristic)
    ]  # TOTAL: 14 features (7 SNR + 1 network + 2 Phase 1A + 4 Phase 1B)
    
    # 🚀 PHASE 1B: Ranges for 14 safe features
    FEATURE_RANGES = {
        # SNR features (7)
        0: (-5.0, 40.0, "lastSnr (dB)"),
        1: (-5.0, 40.0, "snrFast (dB)"),
        2: (-5.0, 40.0, "snrSlow (dB)"),
        3: (-10.0, 10.0, "snrTrendShort"),
        4: (0.0, 10.0, "snrStabilityIndex"),
        5: (0.0, 1.0, "snrPredictionConfidence"),
        6: (0.0, 100.0, "snrVariance"),
        
        # Network state (1 - removed channelWidth)
        7: (0.0, 50.0, "mobilityMetric"),
        
        # Phase 1A features (2 - removed channelBusyRatio)
        8: (0.0, 1.0, "retryRate"),
        9: (0.0, 1.0, "frameErrorRate"),
        
        # 🚀 PHASE 1B: NEW FEATURES (4)
        10: (0.0, 100.0, "rssiVariance (dB²)"),
        11: (0.0, 1.0, "interferenceLevel"),
        12: (0.0, 200.0, "distanceMetric (m)"),
        13: (64.0, 1500.0, "avgPacketSize (bytes)")
    }

    @classmethod
    def clamp_features_inplace(cls, arr: np.ndarray) -> List[str]:
        """Clamp features to realistic ranges IN-PLACE. Returns list of warnings."""
        warnings = []
        
        for i, (min_val, max_val, name) in cls.FEATURE_RANGES.items():
            if i < len(arr):
                original = arr[i]
                arr[i] = np.clip(arr[i], min_val, max_val)
                
                if abs(original - arr[i]) > 1e-6:
                    warnings.append(f"{name}: {original:.3f} → {arr[i]:.3f}")
        
        return warnings
    
    @classmethod
    def validate_features(cls, features: List[float]) -> Tuple[bool, List[str]]:
        """Validate feature count and basic sanity. Returns (is_valid, errors)."""
        errors = []
        
        if len(features) != len(cls.FEATURE_NAMES):
            errors.append(f"Expected {len(cls.FEATURE_NAMES)} features, got {len(features)}")
            return False, errors
        
        # Check for NaN/inf
        for i, val in enumerate(features):
            if not np.isfinite(val):
                errors.append(f"Feature {i} ({cls.FEATURE_NAMES[i]}): invalid value {val}")
        
        return len(errors) == 0, errors

# ================== MONITORING & STATISTICS ==================
class ServerMonitor:
    """Comprehensive server monitoring and statistics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Performance tracking
        self.latencies = deque(maxlen=window_size)
        self.error_counts = defaultdict(int)
        self.model_usage = defaultdict(int)
        self.recent_errors = deque(maxlen=100)
        
        # Rate tracking
        self.request_times = deque(maxlen=window_size)
        
        self._lock = threading.Lock()
    
    def record_request(self, model_name: str, latency_ms: float, success: bool, error: str = None):
        """Record a request for monitoring."""
        with self._lock:
            self.total_requests += 1
            self.request_times.append(time.time())
            
            if success:
                self.successful_requests += 1
                self.latencies.append(latency_ms)
            else:
                self.failed_requests += 1
                if error:
                    self.error_counts[error] += 1
                    self.recent_errors.append({
                        'timestamp': datetime.now().isoformat(),
                        'error': error,
                        'model': model_name
                    })
            
            self.model_usage[model_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            # Calculate request rate
            now = time.time()
            recent_requests = sum(1 for t in self.request_times if now - t <= 60)
            
            # Calculate latency stats
            latency_stats = {}
            if self.latencies:
                latencies = list(self.latencies)
                latency_stats = {
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'min': np.min(latencies),
                    'max': np.max(latencies)
                }
            
            return {
                'uptime_seconds': uptime,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'requests_per_minute': recent_requests,
                'latency_ms': latency_stats,
                'model_usage': dict(self.model_usage),
                'error_counts': dict(self.error_counts),
                'recent_errors': list(self.recent_errors)[-10:]
            }

# ================== ENHANCED ML INFERENCE SERVER ==================
class EnhancedMLInferenceServer:
    """Production-ready ML inference server with monitoring and extensibility."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.default_model = None  # Will be set to oracle_aggressive
        
        self.monitor = ServerMonitor(config.monitoring_window)
        self._stop_event = threading.Event()
        self._setup_logging()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers = [logging.StreamHandler(sys.stdout)]
        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger('MLInferenceServer')
        self.logger.info("="*80)
        self.logger.info("🚀 Enhanced ML Inference Server v4.0 (PHASE 1A - 15 FEATURES)")
        self.logger.info("="*80)
        self.logger.info(f"👤 Author: ahmedjk34 (https://github.com/ahmedjk34)")
        self.logger.info(f"📅 Pipeline Date: 2025-10-02 20:18:13 UTC")
        self.logger.info(f"🔢 Expected features: {len(WiFiFeatures.FEATURE_NAMES)} (PHASE 1B)")
        self.logger.info(f"✅ PHASE 1B: 14 features (7 SNR + 1 network + 2 Phase 1A + 4 Phase 1B)")
        self.logger.info("="*80)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self._stop_event.set()
    
    def add_model(self, model_config: ModelConfig):
        """Add a model to the server."""
        try:
            self.logger.info(f"📦 Loading model '{model_config.name}': {model_config.model_path}")
            start_time = time.time()
            
            # Load model and scaler
            model = joblib.load(model_config.model_path)
            scaler = joblib.load(model_config.scaler_path)
            
            # Store in dictionaries
            self.models[model_config.name] = model
            self.scalers[model_config.name] = scaler
            self.model_configs[model_config.name] = model_config
            
            # Set default model to oracle_aggressive
            if model_config.name == "oracle_aggressive" or self.default_model is None:
                self.default_model = model_config.name
                self.logger.info(f"✨ Set '{model_config.name}' as default model")
            
            load_time = (time.time() - start_time) * 1000
            self.logger.info(f"✅ Model '{model_config.name}' loaded in {load_time:.1f} ms")
            self.logger.info(f"📝 Description: {model_config.description}")
            self.logger.info(f"🔢 Features: {model_config.features_count}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model '{model_config.name}': {str(e)}")
            raise
    
    def predict(self, features: List[float], model_name: str = None) -> Dict[str, Any]:
        """Make prediction using specified model (or default)."""
        start_time = time.time()
        
        # Select model (default to oracle_aggressive)
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        config = self.model_configs[model_name]
        
        try:
            # Validate features
            is_valid, errors = WiFiFeatures.validate_features(features)
            if not is_valid:
                raise ValueError(f"Feature validation failed: {'; '.join(errors)}")
            
            # Convert to numpy array
            features_array = np.array(features, dtype=float).reshape(1, -1)
            
            # Clamp features and log warnings
            clamp_warnings = WiFiFeatures.clamp_features_inplace(features_array[0])
            if clamp_warnings:
                self.logger.debug(f"🔧 Feature clamping: {'; '.join(clamp_warnings)}")
            
            # Log features (compact format)
            if self.logger.isEnabledFor(logging.DEBUG):
                features_str = " ".join([f"{x:.6g}" for x in features_array[0]])
                self.logger.debug(f"[{model_name}] 🔢 Features: {features_str}")
            
            # Scale and predict
            scaled = scaler.transform(features_array)
            pred = model.predict(scaled)[0]
            
            # Clamp to valid rate index range
            rate_idx = int(np.clip(pred, 0, config.rate_classes - 1))
            
            # Compute confidence if available
            confidence = 1.0
            class_probabilities = None
            
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled)
                    if proba is not None and len(proba.shape) == 2:
                        class_probabilities = proba[0].tolist()
                        confidence = float(np.max(proba[0]))
            except Exception as e:
                self.logger.debug(f"Could not compute confidence: {str(e)}")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000.0
            
            result = {
                "rateIdx": rate_idx,
                "latencyMs": latency_ms,
                "success": True,
                "confidence": confidence,
                "model": model_name,
                "clampWarnings": clamp_warnings,
                "classProbabilities": class_probabilities,
                "featuresCount": len(features)
            }
            
            self.logger.info(f"[{model_name}] 🎯 rateIdx={rate_idx} latency={latency_ms:.2f}ms conf={confidence:.3f}")
            
            # Record for monitoring
            if self.config.enable_monitoring:
                self.monitor.record_request(model_name, latency_ms, True)
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            error_msg = str(e)
            
            self.logger.error(f"[{model_name}] ❌ Prediction failed: {error_msg}")
            
            # Record error for monitoring
            if self.config.enable_monitoring:
                self.monitor.record_request(model_name, latency_ms, False, error_msg)
            
            return {
                "rateIdx": 3,  # Safe fallback (18 Mbps)
                "latencyMs": latency_ms,
                "success": False,
                "error": error_msg,
                "model": model_name,
                "confidence": 0.0
            }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information."""
        models_info = {}
        for name, config in self.model_configs.items():
            models_info[name] = {
                "description": config.description,
                "features_count": config.features_count,
                "rate_classes": config.rate_classes,
                "model_path": config.model_path,
                "scaler_path": config.scaler_path,
                "is_default": (name == self.default_model)
            }
        
        info = {
            "server": {
                "version": "4.0.0",
                "author": "ahmedjk34",
                "github": "https://github.com/ahmedjk34",
                "pipeline_date": "2025-10-02 20:18:13 UTC",
                "phase": "PHASE 1A (15 features)",
                "uptime": time.time() - self.monitor.start_time,
                "config": asdict(self.config)
            },
            "models": models_info,
            "default_model": self.default_model,
            "features": {
                "count": len(WiFiFeatures.FEATURE_NAMES),
                "names": WiFiFeatures.FEATURE_NAMES,
                "safe_features_only": True,
                "no_outcome_features": True,
                "phase_1a": "9 original + 6 enhanced"
            }
        }
        
        if self.config.enable_monitoring:
            info["stats"] = self.monitor.get_stats()
        
        return info
    
    def _recv_until_newline(self, conn: socket.socket) -> str:
        """Receive bytes from a socket until a newline is seen or the peer closes."""
        chunks = []
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
            except socket.timeout:
                break
        return b"".join(chunks).decode("utf-8").strip()
    
    def _send_all(self, conn: socket.socket, text: str) -> None:
        """Send all bytes for the given text."""
        b = text.encode("utf-8")
        total = 0
        while total < len(b):
            sent = conn.send(b[total:])
            if sent <= 0:
                raise RuntimeError("send() failed")
            total += sent
    
    def handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection with comprehensive command support."""
        client_id = f"{addr[0]}:{addr[1]}"
        
        try:
            # Set socket timeout
            conn.settimeout(5.0)
            
            data = self._recv_until_newline(conn)
            if not data:
                return
            
            self.logger.debug(f"[{client_id}] 📨 Received: {data[:100]}...")

            # Handle special commands
            if data.strip() == "SHUTDOWN":
                self.logger.info(f"[{client_id}] 🛑 Shutdown requested")
                self._send_all(conn, json.dumps({"ok": True, "message": "Server shutting down"}) + "\n")
                self._stop_event.set()
                return
            
            elif data.strip() == "INFO":
                self.logger.info(f"[{client_id}] ℹ️ Info requested")
                info = self.get_server_info()
                self._send_all(conn, json.dumps(info) + "\n")
                return
            
            elif data.strip() == "STATS":
                self.logger.info(f"[{client_id}] 📊 Stats requested")
                if self.config.enable_monitoring:
                    stats = self.monitor.get_stats()
                    self._send_all(conn, json.dumps(stats) + "\n")
                else:
                    self._send_all(conn, json.dumps({"error": "Monitoring disabled"}) + "\n")
                return
            
            elif data.strip().startswith("MODELS"):
                self.logger.info(f"[{client_id}] 📋 Models list requested")
                models = {name: config.description for name, config in self.model_configs.items()}
                self._send_all(conn, json.dumps({"models": models, "default": self.default_model}) + "\n")
                return
            
            # Handle prediction request
            try:
                # Parse features and optional model name
                parts = data.strip().split()
                
                # 🚀 PHASE 1B: Check if last part is a model name (14 features expected)
                model_name = None
                if len(parts) > 14 and parts[-1] in self.models:
                    model_name = parts[-1]
                    features = [float(x) for x in parts[:-1]]
                else:
                    features = [float(x) for x in parts]

                result = self.predict(features, model_name)
                
            except Exception as e:
                self.logger.error(f"[{client_id}] ❌ Prediction error: {str(e)}")
                result = {
                    "rateIdx": 3,
                    "latencyMs": 0.0,
                    "success": False,
                    "error": str(e),
                    "confidence": 0.0
                }
            
            response = json.dumps(result) + "\n"
            self._send_all(conn, response)
            
        except Exception as e:
            self.logger.error(f"[{client_id}] ❌ Connection error: {str(e)}")
            try:
                error_response = json.dumps({
                    "rateIdx": 3,
                    "latencyMs": 0.0,
                    "success": False,
                    "error": str(e),
                    "confidence": 0.0
                }) + "\n"
                self._send_all(conn, error_response)
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
    
    def run(self):
        """Run the enhanced server with monitoring and graceful shutdown."""
        if not self.models:
            self.logger.error("❌ No models loaded! Cannot start server.")
            return
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind((self.config.host, self.config.port))
            sock.listen(self.config.max_connections)
            sock.settimeout(self.config.socket_timeout)
            
            self.logger.info(f"🚀 Enhanced ML Inference Server listening on {self.config.host}:{self.config.port}")
            self.logger.info(f"📊 Loaded models: {list(self.models.keys())}")
            self.logger.info(f"✨ Default model: {self.default_model}")
            self.logger.info(f"📈 Monitoring enabled: {self.config.enable_monitoring}")
            self.logger.info(f"🔧 Max connections: {self.config.max_connections}")
            self.logger.info(f"🔢 Features expected: {len(WiFiFeatures.FEATURE_NAMES)} (PHASE 1A)")
            self.logger.info("📋 Available commands: INFO, STATS, MODELS, SHUTDOWN")
            
            while not self._stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr),
                        daemon=True,
                        name=f"Client-{addr[0]}:{addr[1]}"
                    )
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self._stop_event.is_set():
                        self.logger.error(f"Accept error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"❌ Server error: {str(e)}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
            
            if self.config.enable_monitoring:
                final_stats = self.monitor.get_stats()
                self.logger.info(f"📊 Final stats: {final_stats['total_requests']} requests, "
                               f"{final_stats['success_rate']:.1%} success rate")
            
            self.logger.info("🛑 Enhanced ML Inference Server stopped")

# ================== AUTO-DISCOVERY & CONFIGURATION ==================
def auto_discover_models(base_path: Path) -> List[ModelConfig]:
    """Auto-discover trained models in the trained_models directory."""
    models = []
    
    # Model priority order (oracle_aggressive first)
    model_patterns = [
        ("oracle_aggressive", "Aggressive oracle - prefers higher rates (75-80% test accuracy expected)"),
        ("oracle_balanced", "Balanced oracle - symmetric exploration (68-72% test accuracy expected)"),
        ("oracle_conservative", "Conservative oracle - prefers lower rates (60-65% test accuracy expected)"),
        ("rateIdx", "Minstrel-HT behavior - mimics ns-3 implementation")
    ]
    
    for model_name, description in model_patterns:
        model_file = base_path / f"step4_rf_{model_name}_FIXED.joblib"
        scaler_file = base_path / f"step4_scaler_{model_name}_FIXED.joblib"
        
        if model_file.exists() and scaler_file.exists():
            models.append(ModelConfig(
                name=model_name,
                model_path=str(model_file),
                scaler_path=str(scaler_file),
                description=description,
                features_count=14,  # 🚀 PHASE 1B: 14 features
                rate_classes=8
            ))
    
    return models

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ML Inference Server v4.0 (Phase 1A - 15 Features)")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log level")
    parser.add_argument("--models-dir", default="python_files/trained_models", help="Models directory")
    
    args = parser.parse_args()
    
    try:
        # Setup base path
        base_path = Path(args.models_dir)
        
        if not base_path.exists():
            print(f"❌ Models directory not found: {base_path}")
            print(f"💡 Please run training first (File 4 with 15 features)")
            sys.exit(1)
        
        # Auto-discover models
        print(f"🔍 Auto-discovering models in: {base_path}")
        model_configs = auto_discover_models(base_path)
        
        if not model_configs:
            print(f"❌ No trained models found in {base_path}")
            print(f"💡 Expected files: step4_rf_*_FIXED.joblib, step4_scaler_*_FIXED.joblib")
            print(f"⚠️ Models must be trained with 14 features (Phase 1B)!")
            sys.exit(1)
        
        print(f"✅ Found {len(model_configs)} models: {[m.name for m in model_configs]}")
        print(f"🔢 Expecting 15 features (Phase 1A)")
        
        # Create server config
        server_config = ServerConfig(
            port=args.port,
            host=args.host,
            log_level=args.log_level,
            enable_monitoring=True
        )
        
        # Create and configure server
        server = EnhancedMLInferenceServer(server_config)
        
        # Load all models
        for model_config in model_configs:
            try:
                server.add_model(model_config)
            except Exception as e:
                print(f"❌ Failed to load model '{model_config.name}': {str(e)}")
                continue
        
        if not server.models:
            print("❌ No models loaded successfully! Cannot start server.")
            sys.exit(1)
        
        # Start server
        server.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    except Exception as e:
        print(f"❌ Server failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)