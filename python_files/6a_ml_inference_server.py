#!/usr/bin/env python3
"""
🚀 PRODUCTION ML Inference Server v6.0 - FULLY OPTIMIZED + DETAILED LOGGING
Author: ahmedjk34 | Date: 2025-10-04 09:42:52 UTC
Features: 14 (Phase 1B) | Thread Pool: 20 workers | Timeout: 300ms
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import traceback
import gc

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class ModelConfig:
    name: str
    model_path: str
    scaler_path: str
    description: str = ""
    features_count: int = 14
    rate_classes: int = 8

@dataclass
class ServerConfig:
    port: int = 8765
    host: str = "localhost"
    max_workers: int = 20  # 🚀 DOUBLED: was 10
    max_queue_size: int = 100  # 🚀 DOUBLED: was 50
    socket_timeout: float = 0.3  # 🚀 REDUCED: was 0.5 (faster response)
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_monitoring: bool = True
    monitoring_window: int = 1000
    max_requests_per_minute: int = 10000  # 🚀 DOUBLED: was 5000

# ============================================================================
# FEATURE VALIDATION
# ============================================================================
class WiFiFeatures:
    FEATURE_NAMES = [
        "lastSnr", "snrFast", "snrSlow", "snrTrendShort", 
        "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
        "mobilityMetric", "retryRate", "frameErrorRate",
        "rssiVariance", "interferenceLevel", "distanceMetric", "avgPacketSize"
    ]
    
    FEATURE_RANGES = {
        0: (-5.0, 40.0, "lastSnr"), 1: (-5.0, 40.0, "snrFast"), 2: (-5.0, 40.0, "snrSlow"),
        3: (-10.0, 10.0, "snrTrend"), 4: (0.0, 10.0, "snrStability"), 5: (0.0, 1.0, "snrConf"),
        6: (0.0, 100.0, "snrVar"), 7: (0.0, 50.0, "mobility"), 8: (0.0, 1.0, "retry"),
        9: (0.0, 1.0, "frameErr"), 10: (0.0, 100.0, "rssiVar"), 11: (0.0, 1.0, "intf"),
        12: (0.0, 200.0, "dist"), 13: (64.0, 1500.0, "pktSize")
    }

    @classmethod
    def clamp_features_inplace(cls, arr: np.ndarray) -> List[str]:
        warnings = []
        for i, (min_val, max_val, name) in cls.FEATURE_RANGES.items():
            if i < len(arr):
                original = arr[i]
                arr[i] = np.clip(arr[i], min_val, max_val)
                if abs(original - arr[i]) > 1e-6:
                    warnings.append(f"{name}: {original:.2f}→{arr[i]:.2f}")
        return warnings
    
    @classmethod
    def validate_features(cls, features: List[float]) -> Tuple[bool, List[str]]:
        errors = []
        if len(features) != len(cls.FEATURE_NAMES):
            errors.append(f"Expected {len(cls.FEATURE_NAMES)} features, got {len(features)}")
            return False, errors
        for i, val in enumerate(features):
            if not np.isfinite(val):
                errors.append(f"Feature {i} invalid: {val}")
        return len(errors) == 0, errors

# ============================================================================
# MONITORING
# ============================================================================
class ServerMonitor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies = deque(maxlen=window_size)
        self.error_counts = defaultdict(int)
        self.model_usage = defaultdict(int)
        self.recent_errors = deque(maxlen=20)
        self.request_times = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def record_request(self, model_name: str, latency_ms: float, success: bool, error: str = None):
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
                        'error': error[:100],
                        'model': model_name
                    })
            self.model_usage[model_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self.start_time
            now = time.time()
            recent_requests = sum(1 for t in self.request_times if now - t <= 60)
            latency_stats = {}
            if self.latencies:
                latencies = list(self.latencies)
                latency_stats = {
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'p95': np.percentile(latencies, 95),
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
                'top_errors': dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            }

# ============================================================================
# RATE LIMITER
# ============================================================================
class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.clients: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_per_minute))
        self._lock = threading.Lock()
    
    def check_rate_limit(self, client_ip: str) -> bool:
        with self._lock:
            now = time.time()
            client_times = self.clients[client_ip]
            while client_times and now - client_times[0] > 60:
                client_times.popleft()
            if len(client_times) >= self.max_per_minute:
                return False
            client_times.append(now)
            return True

# ============================================================================
# MAIN SERVER WITH DETAILED LOGGING
# ============================================================================
class EnhancedMLInferenceServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.default_model = None
        self.monitor = ServerMonitor(config.monitoring_window)
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        self._stop_event = threading.Event()
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="Worker")
        self._setup_logging()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        handlers = [logging.StreamHandler(sys.stdout)]
        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper()), format=log_format, handlers=handlers)
        self.logger = logging.getLogger('MLServer')
        self.logger.info("="*80)
        self.logger.info("🚀 PRODUCTION ML SERVER v6.0 - OPTIMIZED + DETAILED LOGGING")
        self.logger.info("="*80)
        self.logger.info(f"👤 Author: ahmedjk34 | 📅 Date: 2025-10-04 09:42:52 UTC")
        self.logger.info(f"🔢 Features: {len(WiFiFeatures.FEATURE_NAMES)} | ⚡ Workers: {self.config.max_workers}")
        self.logger.info(f"📊 Queue: {self.config.max_queue_size} | ⏱️ Timeout: {self.config.socket_timeout}s")
        self.logger.info("="*80)

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._stop_event.set()
    
    def add_model(self, model_config: ModelConfig):
        try:
            self.logger.info(f"📦 Loading '{model_config.name}'...")
            start_time = time.time()
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                model = joblib.load(model_config.model_path)
                scaler = joblib.load(model_config.scaler_path)
            finally:
                if gc_was_enabled:
                    gc.enable()
            self.models[model_config.name] = model
            self.scalers[model_config.name] = scaler
            self.model_configs[model_config.name] = model_config
            if model_config.name == "oracle_aggressive" or self.default_model is None:
                self.default_model = model_config.name
                self.logger.info(f"✨ Set '{model_config.name}' as default")
            load_time = (time.time() - start_time) * 1000
            self.logger.info(f"✅ Loaded in {load_time:.1f}ms - {model_config.description}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load '{model_config.name}': {str(e)}")
            raise
    
    def predict(self, features: List[float], model_name: str = None) -> Dict[str, Any]:
        start_time = time.time()
        if model_name is None:
            model_name = self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        config = self.model_configs[model_name]
        try:
            is_valid, errors = WiFiFeatures.validate_features(features)
            if not is_valid:
                raise ValueError(f"Validation failed: {'; '.join(errors)}")
            features_array = np.array(features, dtype=float).reshape(1, -1)
            clamp_warnings = WiFiFeatures.clamp_features_inplace(features_array[0])
            scaled = scaler.transform(features_array)
            pred = model.predict(scaled)[0]
            rate_idx = int(np.clip(pred, 0, config.rate_classes - 1))
            confidence = 1.0
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled)
                    if proba is not None:
                        confidence = float(np.max(proba[0]))
            except:
                pass
            latency_ms = (time.time() - start_time) * 1000.0
            result = {
                "rateIdx": rate_idx,
                "latencyMs": latency_ms,
                "success": True,
                "confidence": confidence,
                "model": model_name
            }
            
            # 🚀 DETAILED LOGGING (ALWAYS ON!)
            snr = features[0]
            dist = features[12]
            intf = features[11]
            self.logger.info(f"✅ [{model_name}] rate={rate_idx} conf={confidence:.3f} "
                           f"latency={latency_ms:.1f}ms | SNR={snr:.1f}dB dist={dist:.0f}m intf={intf:.2f}")
            
            self.monitor.record_request(model_name, latency_ms, True)
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            error_msg = str(e)[:200]
            self.logger.error(f"❌ [{model_name}] FAILED: {error_msg}")
            self.monitor.record_request(model_name, latency_ms, False, error_msg)
            return {
                "rateIdx": 3,
                "latencyMs": latency_ms,
                "success": False,
                "error": error_msg,
                "model": model_name,
                "confidence": 0.0
            }
    
    def get_server_info(self) -> Dict[str, Any]:
        return {
            "server": {
                "version": "6.0.0-PRODUCTION",
                "author": "ahmedjk34",
                "date": "2025-10-04 09:42:52 UTC",
                "phase": "Phase 1B (14 features)",
                "uptime": time.time() - self.monitor.start_time,
                "thread_pool_size": self.config.max_workers,
                "max_queue": self.config.max_queue_size
            },
            "models": {name: {"is_default": name == self.default_model} for name in self.models.keys()},
            "default_model": self.default_model,
            "features": {"count": len(WiFiFeatures.FEATURE_NAMES), "names": WiFiFeatures.FEATURE_NAMES},
            "stats": self.monitor.get_stats()
        }
    
    def _recv_with_timeout(self, conn: socket.socket, max_bytes: int = 4096) -> str:
        chunks = []
        total_received = 0
        while total_received < max_bytes:
            try:
                data = conn.recv(min(1024, max_bytes - total_received))
                if not data:
                    break
                chunks.append(data)
                total_received += len(data)
                if b"\n" in data:
                    break
            except socket.timeout:
                break
            except Exception:
                break
        return b"".join(chunks).decode("utf-8", errors="ignore").strip()
    
    def _send_response(self, conn: socket.socket, data: Dict[str, Any]) -> bool:
        try:
            response = json.dumps(data) + "\n"
            conn.sendall(response.encode("utf-8"))
            return True
        except Exception as e:
            self.logger.debug(f"Send failed: {str(e)}")
            return False
    
    def handle_client(self, conn: socket.socket, addr):
        client_id = f"{addr[0]}:{addr[1]}"
        client_ip = addr[0]
        try:
            conn.settimeout(self.config.socket_timeout)
            if not self.rate_limiter.check_rate_limit(client_ip):
                self.logger.warning(f"[{client_id}] Rate limited")
                self._send_response(conn, {"rateIdx": 3, "success": False, "error": "Rate limit", "latencyMs": 0.0})
                return
            data = self._recv_with_timeout(conn)
            if not data:
                return
            if data == "SHUTDOWN":
                self.logger.info("Shutdown requested")
                self._send_response(conn, {"ok": True, "message": "Shutting down"})
                self._stop_event.set()
                return
            elif data == "INFO":
                info = self.get_server_info()
                self._send_response(conn, info)
                return
            elif data == "STATS":
                stats = self.monitor.get_stats()
                self._send_response(conn, stats)
                return
            try:
                parts = data.split()
                model_name = None
                if len(parts) > 14 and parts[-1] in self.models:
                    model_name = parts[-1]
                    features = [float(x) for x in parts[:-1]]
                else:
                    features = [float(x) for x in parts]
                result = self.predict(features, model_name)
            except Exception as e:
                result = {"rateIdx": 3, "success": False, "error": str(e)[:200], "latencyMs": 0.0, "confidence": 0.0}
            self._send_response(conn, result)
        except Exception as e:
            self.logger.debug(f"[{client_id}] Error: {str(e)}")
            try:
                self._send_response(conn, {"rateIdx": 3, "success": False, "error": "Server error", "latencyMs": 0.0})
            except:
                pass
        finally:
            try:
                conn.close()
            except:
                pass
    
    def run(self):
        if not self.models:
            self.logger.error("❌ No models loaded!")
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.config.host, self.config.port))
            sock.listen(self.config.max_queue_size)
            sock.settimeout(self.config.socket_timeout)
            self.logger.info(f"🚀 Server listening on {self.config.host}:{self.config.port}")
            self.logger.info(f"📊 Models: {list(self.models.keys())}")
            self.logger.info(f"✨ Default: {self.default_model}")
            self.logger.info("📋 Commands: INFO, STATS, SHUTDOWN")
            self.logger.info("✅ Server ready with DETAILED LOGGING enabled!")
            while not self._stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    try:
                        self.thread_pool.submit(self.handle_client, conn, addr)
                    except Exception as e:
                        self.logger.warning(f"Thread pool full: {str(e)}")
                        try:
                            conn.send(b'{"error": "Server busy", "rateIdx": 3}\n')
                            conn.close()
                        except:
                            pass
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self._stop_event.is_set():
                        self.logger.error(f"Accept error: {str(e)}")
        except Exception as e:
            self.logger.error(f"❌ Server error: {str(e)}")
        finally:
            self.logger.info("Shutting down thread pool...")
            self.thread_pool.shutdown(wait=True, timeout=5.0)
            try:
                sock.close()
            except:
                pass
            final_stats = self.monitor.get_stats()
            self.logger.info(f"📊 Final: {final_stats['total_requests']} requests, {final_stats['success_rate']:.1%} success")
            self.logger.info("🛑 Server stopped")

# ============================================================================
# AUTO-DISCOVERY
# ============================================================================
def auto_discover_models(base_path: Path) -> List[ModelConfig]:
    models = []
    oracle_patterns = [
        ("oracle_aggressive", "Aggressive - prefers higher rates"),
        ("oracle_balanced", "Balanced - symmetric exploration"),
        ("oracle_conservative", "Conservative - prefers lower rates")
    ]
    for model_name, description in oracle_patterns:
        model_file = base_path / f"step4_rf_{model_name}_FIXED.joblib"
        scaler_file = base_path / f"step4_scaler_{model_name}_FIXED.joblib"
        if model_file.exists() and scaler_file.exists():
            models.append(ModelConfig(name=model_name, model_path=str(model_file), scaler_path=str(scaler_file), description=description, features_count=14, rate_classes=8))
    return models

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production ML Server v6.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--models-dir", default="python_files/trained_models")
    parser.add_argument("--workers", type=int, default=20, help="Thread pool workers")
    args = parser.parse_args()
    try:
        base_path = Path(args.models_dir)
        if not base_path.exists():
            print(f"❌ Models directory not found: {base_path}")
            sys.exit(1)
        print("🔧 Disabling GC for fast model loading...")
        gc.disable()
        print(f"🔍 Discovering oracle models...")
        model_configs = auto_discover_models(base_path)
        if not model_configs:
            print(f"❌ No oracle models found in {base_path}")
            sys.exit(1)
        print(f"✅ Found {len(model_configs)} models: {[m.name for m in model_configs]}")
        server_config = ServerConfig(port=args.port, host=args.host, log_level=args.log_level, max_workers=args.workers)
        server = EnhancedMLInferenceServer(server_config)
        for model_config in model_configs:
            try:
                server.add_model(model_config)
            except Exception as e:
                print(f"❌ Failed to load '{model_config.name}': {str(e)}")
        gc.enable()
        gc.collect()
        print("🔧 GC re-enabled, memory optimized")
        if not server.models:
            print("❌ No models loaded successfully!")
            sys.exit(1)
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted")
    except Exception as e:
        print(f"❌ Server failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)