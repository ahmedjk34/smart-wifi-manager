#!/usr/bin/env python3
"""
ML Inference Server - Keeps models loaded in memory
Usage: python ml_inference_server.py --port 8765
"""

import json
import time
import argparse
import warnings
import joblib
import numpy as np
import socket
import threading
from pathlib import Path

warnings.filterwarnings("ignore")

class MLInferenceServer:
    def __init__(self, model_path, scaler_path, port=8765):
        self.port = port
        self.model = None
        self.scaler = None
        self.load_models(model_path, scaler_path)
        
    def load_models(self, model_path, scaler_path):
        """Load models once at startup"""
        print(f"[SERVER] Loading model: {model_path}")
        start_time = time.time()
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        load_time = (time.time() - start_time) * 1000
        print(f"[SERVER] Models loaded in {load_time:.1f}ms")
        
    def predict(self, features):
        """Make prediction using loaded models"""
        start_time = time.time()
        
        # Validate features
        if len(features) != 18:
            raise ValueError(f"Expected 18 features, got {len(features)}")
            
        features_array = np.array(features).reshape(1, -1)
        
        # Clamp features to realistic ranges
        features_array[0, 0] = max(5.0, min(50.0, features_array[0, 0]))   # lastSnr
        features_array[0, 1] = max(5.0, min(50.0, features_array[0, 1]))   # snrFast  
        features_array[0, 2] = max(5.0, min(50.0, features_array[0, 2]))   # snrSlow
        features_array[0, 3] = max(0.0, min(1.0, features_array[0, 3]))    # shortSuccRatio
        features_array[0, 4] = max(0.0, min(1.0, features_array[0, 4]))    # medSuccRatio
        features_array[0, 7] = max(0.0, min(1.0, features_array[0, 7]))    # severity
        features_array[0, 8] = max(0.0, min(1.0, features_array[0, 8]))    # confidence
        features_array[0, 16] = max(0.0, min(1.0, features_array[0, 16]))  # mobilityMetric
        
        # Scale and predict
        scaled_features = self.scaler.transform(features_array)
        prediction = self.model.predict(scaled_features)[0]
        
        # Clamp to valid range
        rate_idx = max(0, min(7, int(prediction)))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "rateIdx": rate_idx,
            "latencyMs": elapsed_ms,
            "success": True
        }
        
    def handle_client(self, conn, addr):
        """Handle client connection"""
        try:
            # Receive data
            data = conn.recv(1024).decode('utf-8').strip()
            
            if data == "SHUTDOWN":
                print("[SERVER] Shutdown requested")
                return "SHUTDOWN"
                
            # Parse features
            try:
                features = [float(x) for x in data.split()]
                result = self.predict(features)
            except Exception as e:
                result = {
                    "rateIdx": 3,
                    "latencyMs": 0,
                    "success": False,
                    "error": str(e)
                }
            
            # Send response
            response = json.dumps(result) + "\n"
            conn.sendall(response.encode('utf-8'))
            
        except Exception as e:
            error_response = json.dumps({
                "rateIdx": 3,
                "latencyMs": 0,
                "success": False,
                "error": str(e)
            }) + "\n"
            conn.sendall(error_response.encode('utf-8'))
        finally:
            conn.close()
            
        return "OK"
        
    def run(self):
        """Run the server"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('localhost', self.port))
        sock.listen(5)
        
        print(f"[SERVER] ML Inference Server listening on port {self.port}")
        
        try:
            while True:
                conn, addr = sock.accept()
                result = self.handle_client(conn, addr)
                if result == "SHUTDOWN":
                    break
        finally:
            sock.close()
            print("[SERVER] Server stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ahmedjk34/ns-allinone-3.41/ns-3.41/step3_xgb_oracle_best_rateIdx_model_FIXED.joblib")
    parser.add_argument("--scaler", default="/home/ahmedjk34/ns-allinone-3.41/ns-3.41/step3_scaler_FIXED.joblib")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    
    server = MLInferenceServer(args.model, args.scaler, args.port)
    server.run()