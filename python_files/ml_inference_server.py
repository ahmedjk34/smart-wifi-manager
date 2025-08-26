#!/usr/bin/env python3
"""
ML Inference Server - Keeps models loaded in memory and serves newline-delimited requests.
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


def recv_until_newline(conn) -> str:
    """Receive bytes from a socket until a newline is seen or the peer closes."""
    chunks = []
    while True:
        data = conn.recv(1024)
        if not data:
            break
        chunks.append(data)
        if b"\n" in data:
            break
    return b"".join(chunks).decode("utf-8").strip()


def send_all(conn, text: str) -> None:
    """Send all bytes for the given text."""
    b = text.encode("utf-8")
    total = 0
    while total < len(b):
        sent = conn.send(b[total:])
        if sent <= 0:
            raise RuntimeError("send() failed")
        total += sent


class MLInferenceServer:
    def __init__(self, model_path, scaler_path, port=8765):
        self.port = port
        self.model = None
        self.scaler = None
        self._stop_event = threading.Event()
        self.load_models(model_path, scaler_path)

    def load_models(self, model_path, scaler_path):
        """Load models once at startup."""
        print(f"[SERVER] Loading model: {model_path}")
        start_time = time.time()
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        load_time = (time.time() - start_time) * 1000
        print(f"[SERVER] Models loaded in {load_time:.1f} ms")

    def _clamp_features_inplace(self, arr: np.ndarray) -> None:
        """
        Clamp features to realistic ranges IN-PLACE.

        Feature order (22 total) must match the C++ ExtractFeatures():
          0: rateIdx                      [0..7]
          1: phyRate (bps)                [1e3..1e9] (loose clamp)
          2: lastSnr (dB)                 [-5..40]
          3: snrFast (dB)                 [-5..40]
          4: snrSlow (dB)                 [-5..40]
          5: shortSuccRatio               [0..1]
          6: medSuccRatio                 [0..1]
          7: consecSuccess                [0.. inf]
          8: consecFailure                [0.. inf]
          9: severity                     [0..1]
         10: confidence                   [0..1]
         11: T1 (ms)                      [0..1e6]
         12: T2 (ms)                      [0..1e6]
         13: T3 (ms)                      [0..1e6]
         14: decisionReason               [0..100] (unknown space; keep wide)
         15: lastPacketSuccess (0/1)      [0..1]
         16: offeredLoad                  [0..1e9]
         17: queueLen                     [0..1e7]
         18: retryCount                   [0..100]
         19: channelWidth (MHz)           [5..160]
         20: mobilityMetric               [0..1]
         21: snrVariance                  [0..100]
        """
        # Scalars
        arr[0] = np.clip(arr[0], 0, 7)
        arr[1] = np.clip(arr[1], 1e3, 1e9)
        arr[2] = np.clip(arr[2], -5.0, 40.0)
        arr[3] = np.clip(arr[3], -5.0, 40.0)
        arr[4] = np.clip(arr[4], -5.0, 40.0)
        arr[5] = np.clip(arr[5], 0.0, 1.0)
        arr[6] = np.clip(arr[6], 0.0, 1.0)
        # arr[7] = np.clip(arr[7], 0.0, 100.0) no clip on consecSuccess
        # arr[8] = np.clip(arr[8], 0.0, 100.0) no clip on consecFailure
        arr[9] = np.clip(arr[9], 0.0, 1.0)
        arr[10] = np.clip(arr[10], 0.0, 1.0)
        arr[11] = np.clip(arr[11], 0.0, 1e6)
        arr[12] = np.clip(arr[12], 0.0, 1e6)
        arr[13] = np.clip(arr[13], 0.0, 1e6)
        arr[14] = np.clip(arr[14], 0.0, 100.0)
        arr[15] = np.clip(arr[15], 0.0, 1.0)
        arr[16] = np.clip(arr[16], 0.0, 1e9)
        arr[17] = np.clip(arr[17], 0.0, 1e7)
        arr[18] = np.clip(arr[18], 0.0, 100.0)
        arr[19] = np.clip(arr[19], 5.0, 160.0)
        arr[20] = np.clip(arr[20], 0.0, 1.0)
        arr[21] = np.clip(arr[21], 0.0, 100.0)

    def predict(self, features):
        """Make prediction using loaded models."""
        start_time = time.time()

        # Validate features
        if len(features) != 22:
            raise ValueError(f"Expected 22 features, got {len(features)}")

        features_array = np.array(features, dtype=float).reshape(1, -1)

        # Clamp to realistic ranges (in-place)
        self._clamp_features_inplace(features_array[0])

        # Log features sent for prediction (compact)
        features_str = " ".join([f"{x:.6g}" for x in features_array[0]])
        print(f"[INFER] Features: {features_str}")

        # Scale and predict
        scaled = self.scaler.transform(features_array)
        pred = self.model.predict(scaled)[0]

        # Clamp to valid MCS/rate index range
        rate_idx = int(np.clip(pred, 0, 7))

        # Try to compute a confidence if supported
        confidence = 1.0
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(scaled)
                if proba is not None and len(proba.shape) == 2:
                    # Probability of predicted class
                    confidence = float(np.max(proba[0]))
        except Exception:
            # Silently ignore if not supported
            pass

        elapsed_ms = (time.time() - start_time) * 1000.0
        print(f"[RESULT] rateIdx={rate_idx} latencyMs={elapsed_ms:.2f} conf={confidence:.3f}")

        return {
            "rateIdx": rate_idx,
            "latencyMs": elapsed_ms,
            "success": True,
            "confidence": confidence,
        }

    def handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection (runs in its own thread)."""
        try:
            data = recv_until_newline(conn)
            if not data:
                return

            if data.strip() == "SHUTDOWN":
                print("[SERVER] Shutdown requested")
                send_all(conn, json.dumps({"ok": True}) + "\n")
                self._stop_event.set()
                return

            try:
                features = [float(x) for x in data.split()]
                result = self.predict(features)
            except Exception as e:
                print(f"[ERROR] {str(e)} for data: {data}")
                result = {
                    "rateIdx": 3,
                    "latencyMs": 0.0,
                    "success": False,
                    "error": str(e),
                    "confidence": 0.0,
                }

            response = json.dumps(result) + "\n"
            send_all(conn, response)

        except Exception as e:
            print(f"[ERROR] {str(e)} during connection")
            try:
                send_all(
                    conn,
                    json.dumps(
                        {"rateIdx": 3, "latencyMs": 0.0, "success": False, "error": str(e), "confidence": 0.0}
                    )
                    + "\n",
                )
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def run(self):
        """Run the server (thread-per-connection, newline-delimited protocol)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", self.port))
        sock.listen(32)
        sock.settimeout(1.0)  # allow periodic stop checks

        print(f"[SERVER] ML Inference Server listening on port {self.port}")

        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                t.start()
        finally:
            try:
                sock.close()
            except Exception:
                pass
            print("[SERVER] Server stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="step3_rf_oracle_best_rateIdx_model_FIXED.joblib",
    )
    parser.add_argument(
        "--scaler",
        default="step3_scaler_FIXED.joblib",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = MLInferenceServer(args.model, args.scaler, args.port)
    server.run()
