#!/usr/bin/env python3
"""
Enhanced ML Client - Test client for WiFi rate adaptation inference server
FULLY UPDATED FOR NEW PIPELINE (9 safe features, oracle_aggressive default)

Author: ahmedjk34 (https://github.com/ahmedjk34)
Date: 2025-10-02
Usage: 
  python3 python_files/6b_ml_client.py --info
  python3 python_files/6b_ml_client.py --stats
  python3 python_files/6b_ml_client.py --models
  python3 python_files/6b_ml_client.py --predict 25 25 25 0 0.01 0.99 0.5 20 0.5
  python3 python_files/6b_ml_client.py --predict 25 25 25 0 0.01 0.99 0.5 20 0.5 oracle_balanced
  python3 python_files/6b_ml_client.py --batch test_features.txt
"""

import socket
import json
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# ================== ML CLIENT ==================
class MLClient:
    """Client for Enhanced ML Inference Server."""
    
    def __init__(self, host: str = "localhost", port: int = 8765, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
    
    def _send_request(self, message: str) -> Dict[str, Any]:
        """Send a request to the server and return parsed JSON response."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # Send message (with newline)
            sock.sendall((message + "\n").encode("utf-8"))
            
            # Receive response
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
            
            response = b"".join(chunks).decode("utf-8").strip()
            sock.close()
            
            return json.loads(response)
            
        except socket.timeout:
            return {"success": False, "error": "Connection timeout"}
        except ConnectionRefusedError:
            return {"success": False, "error": "Connection refused - is server running?"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def predict(self, features: List[float], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make a prediction with the given features."""
        # Build message: space-separated features, optionally followed by model name
        message = " ".join(str(f) for f in features)
        if model_name:
            message += f" {model_name}"
        
        return self._send_request(message)
    
    def get_info(self) -> Dict[str, Any]:
        """Get server information."""
        return self._send_request("INFO")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self._send_request("STATS")
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models."""
        return self._send_request("MODELS")
    
    def shutdown_server(self) -> Dict[str, Any]:
        """Request server shutdown."""
        return self._send_request("SHUTDOWN")

# ================== PRETTY PRINTING ==================
def print_prediction(result: Dict[str, Any], features: List[float], model_name: Optional[str] = None):
    """Pretty print a prediction result."""
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    
    if result.get("success", False):
        print(f"✅ Status: SUCCESS")
        print(f"🎯 Rate Index: {result['rateIdx']}")
        print(f"⏱️  Latency: {result['latencyMs']:.2f} ms")
        print(f"🔮 Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"🤖 Model: {result.get('model', 'unknown')}")
        
        if result.get('clampWarnings'):
            print(f"⚠️  Clamping: {len(result['clampWarnings'])} features adjusted")
            for warning in result['clampWarnings']:
                print(f"   • {warning}")
        
        if result.get('classProbabilities'):
            print(f"📊 Class Probabilities:")
            probs = result['classProbabilities']
            for i, p in enumerate(probs):
                bar = "█" * int(p * 30)
                print(f"   Rate {i}: {p:.3f} {bar}")
    else:
        print(f"❌ Status: FAILED")
        print(f"⚠️  Error: {result.get('error', 'Unknown error')}")
        print(f"🔄 Fallback Rate Index: {result.get('rateIdx', 3)}")
        print(f"🤖 Model: {result.get('model', 'unknown')}")
    
    print("\n📝 Input Features (9):")
    feature_names = [
        "lastSnr", "snrFast", "snrSlow", "snrTrendShort",
        "snrStabilityIndex", "snrPredictionConfidence", "snrVariance",
        "channelWidth", "mobilityMetric"
    ]
    for i, (name, val) in enumerate(zip(feature_names, features)):
        print(f"   {i}: {name:25s} = {val:10.4f}")
    
    print("="*70 + "\n")

def print_info(info: Dict[str, Any]):
    """Pretty print server info."""
    print("\n" + "="*70)
    print("SERVER INFORMATION")
    print("="*70)
    
    if "error" in info:
        print(f"❌ Error: {info['error']}")
        return
    
    server = info.get("server", {})
    print(f"📦 Version: {server.get('version', 'unknown')}")
    print(f"👤 Author: {server.get('author', 'unknown')}")
    print(f"🔗 GitHub: {server.get('github', 'unknown')}")
    print(f"📅 Pipeline Date: {server.get('pipeline_date', 'unknown')}")
    print(f"⏱️  Uptime: {server.get('uptime', 0):.1f} seconds")
    
    print(f"\n🤖 Available Models ({len(info.get('models', {}))} total):")
    print(f"   Default: {info.get('default_model', 'unknown')}")
    for name, model_info in info.get("models", {}).items():
        is_default = "⭐" if model_info.get("is_default") else "  "
        print(f"   {is_default} {name}")
        print(f"      {model_info.get('description', 'No description')}")
    
    features = info.get("features", {})
    print(f"\n🔢 Features:")
    print(f"   Count: {features.get('count', 0)}")
    print(f"   Safe features only: {features.get('safe_features_only', False)}")
    print(f"   No outcome features: {features.get('no_outcome_features', False)}")
    
    if "stats" in info:
        stats = info["stats"]
        print(f"\n📊 Statistics:")
        print(f"   Total requests: {stats.get('total_requests', 0)}")
        print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"   Requests/min: {stats.get('requests_per_minute', 0)}")
        
        latency = stats.get('latency_ms', {})
        if latency:
            print(f"   Latency (ms): mean={latency.get('mean', 0):.2f} "
                  f"p95={latency.get('p95', 0):.2f} p99={latency.get('p99', 0):.2f}")
    
    print("="*70 + "\n")

def print_stats(stats: Dict[str, Any]):
    """Pretty print server statistics."""
    print("\n" + "="*70)
    print("SERVER STATISTICS")
    print("="*70)
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"⏱️  Uptime: {stats.get('uptime_seconds', 0):.1f} seconds")
    print(f"📨 Total Requests: {stats.get('total_requests', 0)}")
    print(f"✅ Successful: {stats.get('successful_requests', 0)}")
    print(f"❌ Failed: {stats.get('failed_requests', 0)}")
    print(f"📊 Success Rate: {stats.get('success_rate', 0):.1%}")
    print(f"⚡ Requests/min: {stats.get('requests_per_minute', 0)}")
    
    latency = stats.get('latency_ms', {})
    if latency:
        print(f"\n⏱️  Latency (ms):")
        print(f"   Mean:   {latency.get('mean', 0):8.2f}")
        print(f"   Median: {latency.get('median', 0):8.2f}")
        print(f"   P95:    {latency.get('p95', 0):8.2f}")
        print(f"   P99:    {latency.get('p99', 0):8.2f}")
        print(f"   Min:    {latency.get('min', 0):8.2f}")
        print(f"   Max:    {latency.get('max', 0):8.2f}")
    
    model_usage = stats.get('model_usage', {})
    if model_usage:
        print(f"\n🤖 Model Usage:")
        for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model:20s}: {count:6d} requests")
    
    error_counts = stats.get('error_counts', {})
    if error_counts:
        print(f"\n⚠️  Error Summary:")
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error[:50]:50s}: {count:4d}")
    
    print("="*70 + "\n")

def print_models(models_info: Dict[str, Any]):
    """Pretty print available models."""
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    
    if "error" in models_info:
        print(f"❌ Error: {models_info['error']}")
        return
    
    print(f"✨ Default Model: {models_info.get('default', 'unknown')}\n")
    
    for name, description in models_info.get("models", {}).items():
        is_default = "⭐" if name == models_info.get('default') else "  "
        print(f"{is_default} {name}")
        print(f"   {description}")
        print()
    
    print("="*70 + "\n")

# ================== BATCH PROCESSING ==================
def run_batch(client: MLClient, batch_file: str):
    """Run predictions from a batch file."""
    print(f"📄 Loading batch file: {batch_file}")
    
    try:
        with open(batch_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"✅ Loaded {len(lines)} predictions\n")
        
        results = []
        for i, line in enumerate(lines, 1):
            parts = line.split()
            
            # Parse features and optional model name
            model_name = None
            if len(parts) > 9 and not is_float(parts[-1]):
                model_name = parts[-1]
                features = [float(x) for x in parts[:-1]]
            else:
                features = [float(x) for x in parts]
            
            print(f"[{i}/{len(lines)}] Running prediction with model={model_name or 'default'}...")
            result = client.predict(features, model_name)
            results.append((features, model_name, result))
            
            if result.get("success"):
                print(f"   ✅ rateIdx={result['rateIdx']} latency={result['latencyMs']:.2f}ms")
            else:
                print(f"   ❌ Error: {result.get('error', 'unknown')}")
        
        # Summary
        print("\n" + "="*70)
        print("BATCH SUMMARY")
        print("="*70)
        successful = sum(1 for _, _, r in results if r.get("success"))
        print(f"Total: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Success Rate: {successful/len(results):.1%}")
        
        avg_latency = sum(r['latencyMs'] for _, _, r in results if r.get('success')) / max(successful, 1)
        print(f"Average Latency: {avg_latency:.2f} ms")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print(f"❌ Batch file not found: {batch_file}")
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")

def is_float(s: str) -> bool:
    """Check if string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced ML Client v3.0 (New Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get server info
  python3 6b_ml_client.py --info
  
  # Get server statistics
  python3 6b_ml_client.py --stats
  
  # List available models
  python3 6b_ml_client.py --models
  
  # Make prediction (default model = oracle_aggressive)
  python3 6b_ml_client.py --predict 25 25 25 0 0.01 0.99 0.5 20 0.5
  
  # Make prediction with specific model
  python3 6b_ml_client.py --predict 25 25 25 0 0.01 0.99 0.5 20 0.5 oracle_balanced
  
  # Batch predictions from file
  python3 6b_ml_client.py --batch test_features.txt
  
  # Shutdown server
  python3 6b_ml_client.py --shutdown

Feature order (9 features):
  1. lastSnr (dB)
  2. snrFast (dB)
  3. snrSlow (dB)
  4. snrTrendShort
  5. snrStabilityIndex
  6. snrPredictionConfidence
  7. snrVariance
  8. channelWidth (MHz)
  9. mobilityMetric
        """
    )
    
    # Connection options
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--timeout", type=float, default=5.0, help="Connection timeout")
    
    # Commands (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--info", action="store_true", help="Get server information")
    group.add_argument("--stats", action="store_true", help="Get server statistics")
    group.add_argument("--models", action="store_true", help="List available models")
    group.add_argument("--predict", nargs='+', help="Make prediction (9 features + optional model name)")
    group.add_argument("--batch", help="Run batch predictions from file")
    group.add_argument("--shutdown", action="store_true", help="Shutdown server")
    
    args = parser.parse_args()
    
    # Create client
    client = MLClient(host=args.host, port=args.port, timeout=args.timeout)
    
    try:
        if args.info:
            result = client.get_info()
            print_info(result)
        
        elif args.stats:
            result = client.get_stats()
            print_stats(result)
        
        elif args.models:
            result = client.get_models()
            print_models(result)
        
        elif args.predict:
            # Parse features and optional model name
            model_name = None
            if len(args.predict) > 9 and not is_float(args.predict[-1]):
                model_name = args.predict[-1]
                features = [float(x) for x in args.predict[:-1]]
            else:
                features = [float(x) for x in args.predict]
            
            if len(features) != 9:
                print(f"❌ Expected 9 features, got {len(features)}")
                print(f"💡 Use --help to see feature order")
                sys.exit(1)
            
            result = client.predict(features, model_name)
            print_prediction(result, features, model_name)
        
        elif args.batch:
            run_batch(client, args.batch)
        
        elif args.shutdown:
            print("🛑 Requesting server shutdown...")
            result = client.shutdown_server()
            if result.get("ok"):
                print("✅ Server shutdown initiated")
            else:
                print(f"❌ Shutdown failed: {result.get('error', 'unknown')}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Client error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()