#!/usr/bin/env python3
"""
ml_rate_inference.py

Robust, single-shot inference script for WiFi Rate Adaptation models.

Usage:
  python ml_rate_inference.py \
      --model /path/to/model.joblib \
      --scaler /path/to/scaler.joblib \
      --features 12.3 11.9 12.1 0.93 0.91 7 0 0.2 0.85 3 9 18 4.2 12 1 20 0.05 0.33 \
      --output-format plain

Features order (must match training):
[lastSnr, snrFast, snrSlow, shortSuccRatio, medSuccRatio,
 consecSuccess, consecFailure, severity, confidence, T1, T2, T3,
 offeredLoad, queueLen, retryCount, channelWidth, mobilityMetric, snrVariance]

Exit codes:
 0 success
 2 feature validation error
 3 artifact load error
 4 inference error
"""

import os
import sys
import json
import time
import math
import traceback
import argparse
from typing import List, Optional
import warnings
import joblib
import numpy as np

# Suppress all sklearn warnings to avoid feature name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")



REQUIRED_FEATURE_COUNT = 18
FEATURE_NAMES = [
    "lastSnr","snrFast","snrSlow","shortSuccRatio","medSuccRatio",
    "consecSuccess","consecFailure","severity","confidence","T1","T2","T3",
    "offeredLoad","queueLen","retryCount","channelWidth","mobilityMetric","snrVariance"
]

def parse_args():
    p = argparse.ArgumentParser(description="WiFi ML Rate Adaptation Inference")
    p.add_argument("--model", required=True, help="Path to .joblib model file")
    p.add_argument("--scaler", required=True, help="Path to scaler .joblib file")
    p.add_argument("--features", nargs="+", required=True, help=f"{REQUIRED_FEATURE_COUNT} numeric feature values")
    p.add_argument("--output-format", choices=["plain", "json"], default="plain",
                   help="plain prints predicted rateIdx only; json returns structured output")
    p.add_argument("--round", type=int, default=None,
                   help="Round floating inputs to this many decimals before inference")
    p.add_argument("--probabilities", action="store_true",
                   help="Include class probabilities in JSON output (if model supports)")
    p.add_argument("--validate-range", action="store_true",
                   help="Clamp certain feature ranges (snr, ratios, widths)")
    p.add_argument("--strict", action="store_true",
                   help="Fail on NaN/Inf instead of auto-fixing to 0")
    return p.parse_args()

def coerce_and_validate_features(raw: List[str],
                                 round_decimals: Optional[int],
                                 validate_range: bool,
                                 strict: bool) -> np.ndarray:
    if len(raw) != REQUIRED_FEATURE_COUNT:
        raise ValueError(f"Expected {REQUIRED_FEATURE_COUNT} features, got {len(raw)}")

    vals = []
    for idx, r in enumerate(raw):
        try:
            v = float(r)
            if math.isnan(v) or math.isinf(v):
                if strict:
                    raise ValueError(f"Feature {FEATURE_NAMES[idx]} is NaN/Inf.")
                v = 0.0
            if round_decimals is not None:
                v = round(v, round_decimals)
            vals.append(v)
        except Exception:
            raise ValueError(f"Cannot parse feature {idx} ({FEATURE_NAMES[idx]})='{r}'")
    arr = np.array(vals, dtype=float).reshape(1, -1)

    if validate_range:
        # SNR clipping
        for name in ["lastSnr","snrFast","snrSlow"]:
            i = FEATURE_NAMES.index(name)
            arr[0, i] = max(-20.0, min(90.0, arr[0, i]))
        # Ratios 0..1
        for name in ["shortSuccRatio","medSuccRatio","confidence","severity"]:
            i = FEATURE_NAMES.index(name)
            arr[0, i] = max(0.0, min(1.0, arr[0, i]))
        # Non-negative ints
        for name in ["consecSuccess","consecFailure","retryCount","queueLen","T1","T2","T3"]:
            i = FEATURE_NAMES.index(name)
            arr[0, i] = max(0.0, arr[0, i])
        # channelWidth typical
        i_cw = FEATURE_NAMES.index("channelWidth")
        arr[0, i_cw] = max(5.0, min(320.0, arr[0, i_cw]))
    return arr

def load_artifact(path: str, kind: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{kind} file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {kind} '{path}': {e}")

def main():
    t0 = time.time()
    args = parse_args()

    try:
        features = coerce_and_validate_features(
            args.features, args.round, args.validate_range, args.strict
        )
    except Exception as e:
        print(f"[ERROR] Feature validation: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        scaler = load_artifact(args.scaler, "Scaler")
        model = load_artifact(args.model, "Model")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(3)

    try:
        # Suppress warnings during scaling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = scaler.transform(features)
    except Exception as e:
        print(f"[ERROR] Scaling failed: {e}", file=sys.stderr)
        sys.exit(4)

    try:
        pred = model.predict(scaled)
        rate_idx = int(pred[0])
        
        # Clamp rate index to valid range (0-7 for 802.11g)
        rate_idx = max(0, min(7, rate_idx))
        
        probs = None
        if args.probabilities and hasattr(model, "predict_proba"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    probs = model.predict_proba(scaled)[0].tolist()
            except Exception:
                probs = None
        latency_ms = (time.time() - t0) * 1000.0

        if args.output_format == "plain":
            print(rate_idx)
        else:
            out = {
                "rateIdx": rate_idx,
                "latencyMs": latency_ms,
                "modelPath": args.model,
                "scalerPath": args.scaler,
                "features": {FEATURE_NAMES[i]: float(features[0, i]) for i in range(REQUIRED_FEATURE_COUNT)},
                "probabilities": probs
            }
            print(json.dumps(out))
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()