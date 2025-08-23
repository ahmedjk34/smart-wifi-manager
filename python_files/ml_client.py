#!/usr/bin/env python3
"""
ML Client - robust, newline-delimited
"""

import socket
import json
import sys
import argparse


def recv_until_newline(sock: socket.socket) -> str:
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


def send_all(sock: socket.socket, text: str) -> None:
    """Send all bytes for the given text."""
    b = text.encode("utf-8")
    total = 0
    while total < len(b):
        sent = sock.send(b[total:])
        if sent <= 0:
            raise RuntimeError("send() failed")
        total += sent


def query_server(features, port=8765):
    """Query the ML inference server with a list of 22 features."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("localhost", port))

        # Send features (newline-delimited)
        message = " ".join(map(str, features)) + "\n"
        send_all(sock, message)

        # Receive full JSON line
        response = recv_until_newline(sock)
        sock.close()

        return json.loads(response)

    except Exception as e:
        return {
            "rateIdx": 3,
            "latencyMs": 0,
            "success": False,
            "error": str(e),
            "confidence": 0.0,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-format", choices=["json", "rate"], default="json")
    args = parser.parse_args()

    # ***** MUST be 22 to match C++ + server *****
    if len(args.features) != 22:
        print(f"Error: Expected 22 features, got {len(args.features)}")
        sys.exit(1)

    features = [float(f) for f in args.features]
    result = query_server(features, args.port)

    if args.output_format == "json":
        print(json.dumps(result))
    else:
        print(result.get("rateIdx", 3))
