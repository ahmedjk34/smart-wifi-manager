#!/usr/bin/env python3
"""
Enhanced ML Client - Robust, feature-rich client for WiFi rate adaptation inference

Author: ahmedjk34 (https://github.com/ahmedjk34)
Date: 2025-09-22
Usage: python3 python_files/6b_enhanced_ml_client.py --features 25 24 23 ... --model oracle_balanced
"""

import socket
import json
import sys
import argparse
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# ================== ENHANCED ML CLIENT ==================
class EnhancedMLClient:
    """Enhanced ML client with comprehensive features and error handling."""
    
    def __init__(self, host: str = "localhost", port: int = 8765, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup client logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('MLClient')
    
    def _recv_until_newline(self, sock: socket.socket) -> str:
        """Receive until newline or peer close."""
        chunks = []
        while True:
            data = sock.recv(1024)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
        return b"".join(chunks).decode("utf-8").strip()
    
    def _send_all(self, sock: socket.socket, text: str) -> None:
        """Send all bytes for the given text."""
        b = text.encode("utf-8")
        total = 0
        while total < len(b):
            sent = sock.send(b[total:])
            if sent <= 0:
                raise RuntimeError("send() failed")
            total += sent
    
    def _connect_and_send(self, message: str) -> Dict[str, Any]:
        """Connect to server and send message, return parsed response."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            self._send_all(sock, message + "\n")
            response = self._recv_until_newline(sock)
            sock.close()
            
            return json.loads(response) if response else {"error": "Empty response"}
            
        except socket.timeout:
            return {"error": "Connection timeout", "success": False}
        except ConnectionRefusedError:
            return {"error": "Connection refused - is server running?", "success": False}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def predict(self, features: List[float], model: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction with optional model selection."""
        if len(features) != 28:  # FIXED: 28 features
            return {
                "error": f"Expected 28 features, got {len(features)}",
                "success": False,
                "rateIdx": 3
            }
        
        # Build message
        message = " ".join(map(str, features))
        if model:
            message += f" {model}"
        
        self.logger.debug(f"Sending prediction request: {message[:50]}...")
        return self._connect_and_send(message)
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information."""
        return self._connect_and_send("INFO")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self._connect_and_send("STATS")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models."""
        return self._connect_and_send("MODELS")
    
    def shutdown_server(self) -> Dict[str, Any]:
        """Request server shutdown."""
        return self._connect_and_send("SHUTDOWN")
    
    def ping(self) -> Dict[str, Any]:
        """Simple connectivity test."""
        start_time = time.time()
        result = self.get_available_models()
        result['ping_time_ms'] = (time.time() - start_time) * 1000
        return result
    
    def benchmark(self, features: List[float], iterations: int = 100, model: Optional[str] = None) -> Dict[str, Any]:
        """Benchmark prediction performance."""
        if len(features) != 28:  # FIXED: 28 features
            return {"error": f"Expected 28 features, got {len(features)}", "success": False}
        
        latencies = []
        successes = 0
        errors = []
        
        self.logger.info(f"🚀 Starting benchmark: {iterations} iterations")
        
        for i in range(iterations):
            if i % 10 == 0:
                self.logger.info(f"Progress: {i}/{iterations}")
            
            result = self.predict(features, model)
            
            if result.get('success', False):
                successes += 1
                latencies.append(result.get('latencyMs', 0))
            else:
                errors.append(result.get('error', 'Unknown error'))
        
        if latencies:
            import statistics
            stats = {
                'iterations': iterations,
                'successes': successes,
                'success_rate': successes / iterations,
                'latency_ms': {
                    'mean': statistics.mean(latencies),
                    'median': statistics.median(latencies),
                    'min': min(latencies),
                    'max': max(latencies),
                    'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
                },
                'errors': list(set(errors)),
                'error_count': len(errors)
            }
        else:
            stats = {
                'iterations': iterations,
                'successes': 0,
                'success_rate': 0,
                'errors': list(set(errors)),
                'error_count': len(errors)
            }
        
        return stats

# ================== COMMAND LINE INTERFACE ==================
def load_features_from_file(filepath: str) -> List[float]:
    """Load features from a file (one per line or space-separated)."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    # Try space-separated first
    try:
        return [float(x) for x in content.split()]
    except ValueError:
        # Try line-separated
        lines = content.split('\n')
        return [float(line.strip()) for line in lines if line.strip()]

def print_formatted_result(result: Dict[str, Any], format_type: str):
    """Print result in specified format."""
    if format_type == "json":
        print(json.dumps(result, indent=2))
    elif format_type == "compact":
        if result.get('success', False):
            print(f"rateIdx={result.get('rateIdx', 'N/A')} "
                  f"latency={result.get('latencyMs', 0):.1f}ms "
                  f"conf={result.get('confidence', 0):.3f}")
        else:
            print(f"ERROR: {result.get('error', 'Unknown')}")
    elif format_type == "rate":
        print(result.get('rateIdx', 3))
    else:  # detailed
        print("🎯 ML Inference Result:")
        print(f"  Rate Index: {result.get('rateIdx', 'N/A')}")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Latency: {result.get('latencyMs', 0):.2f} ms")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Model: {result.get('model', 'Unknown')}")
        
        if result.get('clampWarnings'):
            print(f"  ⚠️  Clamp Warnings: {len(result['clampWarnings'])}")
        
        if not result.get('success', False):
            print(f"  ❌ Error: {result.get('error', 'Unknown')}")

def main():
    """Enhanced command line interface."""
    parser = argparse.ArgumentParser(description="Enhanced ML Client for WiFi Rate Adaptation")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--timeout", type=float, default=5.0, help="Connection timeout")
    
    # Feature input options - FIXED: 28 features
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument("--features", nargs=28, type=float, help="28 WiFi safe features")
    feature_group.add_argument("--features-file", help="File containing features")
    
    # Model selection
    parser.add_argument("--model", help="Model name to use")
    
    # Output format
    parser.add_argument("--format", choices=["json", "compact", "rate", "detailed"], 
                       default="detailed", help="Output format")
    
    # Commands
    parser.add_argument("--info", action="store_true", help="Get server info")
    parser.add_argument("--stats", action="store_true", help="Get server stats")
    parser.add_argument("--models", action="store_true", help="List available models")
    parser.add_argument("--ping", action="store_true", help="Test server connectivity")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown server")
    parser.add_argument("--benchmark", type=int, metavar="N", help="Benchmark N predictions")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create client
    client = EnhancedMLClient(args.host, args.port, args.timeout)
    
    try:
        # Handle info commands
        if args.info:
            result = client.get_server_info()
            print_formatted_result(result, args.format)
            return
        
        if args.stats:
            result = client.get_server_stats()
            print_formatted_result(result, args.format)
            return
        
        if args.models:
            result = client.get_available_models()
            print_formatted_result(result, args.format)
            return
        
        if args.ping:
            result = client.ping()
            if result.get('models'):
                print(f"✅ Server is responding (ping: {result.get('ping_time_ms', 0):.1f}ms)")
                print(f"📊 Available models: {list(result['models'].keys())}")
            else:
                print(f"❌ Server connection failed: {result.get('error', 'Unknown')}")
            return
        
        if args.shutdown:
            result = client.shutdown_server()
            print(f"🛑 Shutdown request sent: {result}")
            return
        
        # Load features
        features = None
        if args.features:
            features = args.features
        elif args.features_file:
            try:
                features = load_features_from_file(args.features_file)
                print(f"📁 Loaded {len(features)} features from {args.features_file}")
            except Exception as e:
                print(f"❌ Failed to load features from file: {str(e)}")
                sys.exit(1)
        else:
            print("❌ No features provided. Use --features or --features-file")
            parser.print_help()
            sys.exit(1)
        
        # Validate feature count - FIXED: 28 features
        if len(features) != 28:
            print(f"❌ Expected 28 features, got {len(features)}")
            sys.exit(1)
        
        # Handle prediction commands
        if args.benchmark:
            print(f"🚀 Benchmarking {args.benchmark} predictions...")
            result = client.benchmark(features, args.benchmark, args.model)
            print_formatted_result(result, args.format)
        else:
            # Single prediction
            result = client.predict(features, args.model)
            print_formatted_result(result, args.format)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Client error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()