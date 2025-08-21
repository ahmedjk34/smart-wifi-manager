#!/usr/bin/env python3
"""
ML Client - Fast client for inference server
"""

import socket
import json
import sys
import argparse

def query_server(features, port=8765):
    """Query the ML inference server"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        
        # Send features
        message = " ".join(map(str, features)) + "\n"
        sock.sendall(message.encode('utf-8'))
        
        # Receive response
        response = sock.recv(1024).decode('utf-8').strip()
        sock.close()
        
        return json.loads(response)
        
    except Exception as e:
        return {
            "rateIdx": 3,
            "latencyMs": 0,
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-format", default="json")
    args = parser.parse_args()
    
    if len(args.features) != 18:
        print(f"Error: Expected 18 features, got {len(args.features)}")
        sys.exit(1)
    
    features = [float(f) for f in args.features]
    result = query_server(features, args.port)
    
    if args.output_format == "json":
        print(json.dumps(result))
    else:
        print(result.get("rateIdx", 3))