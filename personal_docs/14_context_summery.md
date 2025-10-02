# 📋 **CONTEXT SUMMARY - CRITICAL INFO FOR CC FILES**

**Current Time:** 2025-10-02 17:53:36 UTC  
**User:** ahmedjk34  
**Status:** ✅ ML Pipeline Complete - Ready for ns-3 Integration

---

## **🎯 MOST CRITICAL FACTS FOR CC FILES**

### **1. MODEL SPECIFICATIONS (MUST USE EXACTLY THIS)**

```cpp
// Feature count: 9 SAFE FEATURES ONLY (NO outcome features!)
const int NUM_FEATURES = 9;

// Feature order (MUST match training pipeline):
// 0: lastSnr (dB)           - Most recent SNR measurement
// 1: snrFast (dB)           - Fast-moving average SNR
// 2: snrSlow (dB)           - Slow-moving average SNR
// 3: snrTrendShort          - Short-term SNR trend
// 4: snrStabilityIndex      - SNR stability metric
// 5: snrPredictionConfidence - Confidence in SNR prediction
// 6: snrVariance            - SNR variance
// 7: channelWidth (MHz)     - Channel bandwidth (20, 40, 80, 160)
// 8: mobilityMetric         - Node mobility metric

// Rate classes: 8 (0-7, corresponding to 802.11a rates)
// Rate 0: 6 Mbps, Rate 1: 9 Mbps, ..., Rate 7: 54 Mbps
```

### **2. SERVER CONNECTION DETAILS**

```cpp
// Default server configuration
const char* ML_SERVER_HOST = "localhost";
const int ML_SERVER_PORT = 8765;
const int SOCKET_TIMEOUT_MS = 5000;

// Protocol: Space-separated features + newline
// Example: "25.0 25.0 25.0 0.0 0.01 0.99 0.5 20.0 0.5\n"

// Optional model selection (append model name):
// "25.0 25.0 25.0 0.0 0.01 0.99 0.5 20.0 0.5 oracle_aggressive\n"
```

### **3. AVAILABLE MODELS (Priority Order)**

```cpp
enum MLModel {
    ORACLE_AGGRESSIVE,  // Default - 62.8% test accuracy
    ORACLE_BALANCED,    // 45.3% test accuracy
    ORACLE_CONSERVATIVE,// 47.5% test accuracy
    RATEIDX             // 46.1% test accuracy (Minstrel-HT behavior)
};

// Model names for server requests:
const char* MODEL_NAMES[] = {
    "oracle_aggressive",
    "oracle_balanced",
    "oracle_conservative",
    "rateIdx"
};
```

---

## **📊 MODEL PERFORMANCE CHARACTERISTICS**

### **oracle_aggressive (RECOMMENDED DEFAULT)**

```
Test Accuracy: 62.8%
Validation Accuracy: 62.9% (consistent!)

Behavior:
- Prefers higher rates (faster throughput)
- 45% stay at base rate, 30% increase by 1, 15% increase by 2
- Best for stable, high-SNR environments
- Expected throughput: 10-20% better than Minstrel-HT

Confusion Matrix Pattern:
- Good at high rates (Rate 6-7: 75-85% recall)
- Decent at mid rates (Rate 3-5: 63-68% recall)
- Struggles with low rates (Rate 0-2: 24-48% recall)

When to use:
✅ SNR > 20 dB
✅ Low mobility
✅ Stable channel conditions
❌ High packet loss environments
```

### **oracle_balanced**

```
Test Accuracy: 45.3%
Validation Accuracy: 46.9%

Behavior:
- Symmetric exploration (equal chance up/down)
- 35% stay, 25% increase, 25% decrease
- Most exploration, least predictable
- Good for learning/adaptive scenarios

When to use:
✅ Variable SNR conditions
✅ Unknown network quality
✅ Testing/benchmarking
❌ Production (too conservative)
```

### **oracle_conservative**

```
Test Accuracy: 47.5%
Validation Accuracy: 48.1%

Behavior:
- Prefers lower rates (safer, more reliable)
- 45% stay, 30% decrease by 1, 15% decrease by 2
- Best for high packet loss environments
- Lower throughput, higher reliability

When to use:
✅ SNR < 15 dB
✅ High mobility
✅ High interference
✅ Mission-critical applications
```

---

## **🔧 CRITICAL IMPLEMENTATION DETAILS**

### **Feature Calculation (MUST DO IN CC)**

```cpp
// 1. SNR Calculation (CRITICAL - most important features)
double lastSnr = 10.0 * log10(rxPowerW / noisePowerW);  // Current SNR in dB
double snrFast = 0.8 * snrFast_prev + 0.2 * lastSnr;   // EWMA α=0.2
double snrSlow = 0.95 * snrSlow_prev + 0.05 * lastSnr; // EWMA α=0.05

// 2. SNR Trend (short-term derivative)
double snrTrendShort = snrFast - snrSlow;  // Range: -10 to +10 dB

// 3. SNR Stability (variance over window)
double snrStabilityIndex = CalculateVariance(snrHistory, 10);  // Last 10 samples

// 4. SNR Prediction Confidence (how stable is prediction?)
double snrPredictionConfidence = 1.0 / (1.0 + snrVariance);  // Range: 0-1

// 5. SNR Variance (critical for oracle decision)
double snrVariance = CalculateVariance(snrHistory, 100);  // Last 100 samples

// 6. Channel Width (from PHY layer)
double channelWidth = GetChannelWidth();  // 20, 40, 80, or 160 MHz

// 7. Mobility Metric (node movement speed)
double mobilityMetric = CalculateNodeSpeed();  // m/s or normalized 0-1
```

### **Socket Communication (Python-style in C++)**

```cpp
// Send request
std::string request = FormatFeatures(features) + "\n";
send(sockfd, request.c_str(), request.length(), 0);

// Receive response
char buffer[4096];
ssize_t bytes = recv(sockfd, buffer, sizeof(buffer), 0);
std::string response(buffer, bytes);

// Parse JSON response
// {"rateIdx": 5, "latencyMs": 2.34, "success": true, "confidence": 0.87, ...}
```

---

## **⚠️ COMMON PITFALLS TO AVOID**

### **1. Feature Order Mismatch**

```cpp
// ❌ WRONG - will produce garbage predictions!
double features[9] = {channelWidth, lastSnr, snrFast, ...};

// ✅ CORRECT - must match training order!
double features[9] = {lastSnr, snrFast, snrSlow, snrTrendShort,
                      snrStabilityIndex, snrPredictionConfidence,
                      snrVariance, channelWidth, mobilityMetric};
```

### **2. Feature Value Ranges**

```cpp
// Server auto-clamps, but better to clamp before sending:
lastSnr = std::clamp(lastSnr, -5.0, 40.0);           // dB
snrFast = std::clamp(snrFast, -5.0, 40.0);           // dB
snrTrendShort = std::clamp(snrTrendShort, -10.0, 10.0);
snrStabilityIndex = std::clamp(snrStabilityIndex, 0.0, 10.0);
snrPredictionConfidence = std::clamp(snrPredictionConfidence, 0.0, 1.0);
snrVariance = std::clamp(snrVariance, 0.0, 100.0);
channelWidth = std::clamp(channelWidth, 5.0, 160.0);  // MHz
mobilityMetric = std::clamp(mobilityMetric, 0.0, 50.0);
```

### **3. Missing Features (NO OUTCOME FEATURES!)**

```cpp
// ❌ REMOVED - these are OUTCOME features (data leakage!)
// shortSuccRatio
// medSuccRatio
// packetLossRate
// severity
// confidence (server-side metric, not input!)
// consecSuccess, consecFailure
// retrySuccessRatio
// recentRateChanges, timeSinceLastRateChange, rateStabilityScore
// packetSuccess

// ✅ ONLY use the 9 safe features listed above!
```

---

## **📈 BENCHMARKING STRATEGY**

### **Baseline: Minstrel-HT**

```
Expected Performance:
- Throughput: 60-80% of theoretical maximum
- Packet Loss: 5-15% (exploration overhead)
- Rate adaptation time: 100-500 ms
```

### **ML Model: oracle_aggressive**

```
Expected Performance:
- Throughput: 70-85% of theoretical maximum (10-20% better!)
- Packet Loss: 3-10% (less exploration)
- Rate adaptation time: 5-10 ms (100x faster!)
```

### **Test Scenarios (MUST TEST)**

```cpp
// 1. Static High SNR (SNR > 25 dB)
//    Expected: ML should match or beat Minstrel
//    ML advantage: Faster convergence to Rate 7

// 2. Static Low SNR (SNR < 15 dB)
//    Expected: ML slightly worse (overfits to high rates)
//    Mitigation: Use oracle_conservative

// 3. Dynamic SNR (fluctuating 10-30 dB)
//    Expected: ML advantage due to faster adaptation
//    Key metric: Rate switching frequency

// 4. High Mobility (speed > 10 m/s)
//    Expected: Both struggle, ML slightly better
//    Use mobilityMetric feature!

// 5. High Interference (SNR variance > 5 dB)
//    Expected: ML advantage (uses snrVariance!)
//    Key: oracle responds to variance immediately
```

---

## **🎯 KEY METRICS TO TRACK**

```cpp
struct BenchmarkMetrics {
    // Throughput
    double avgThroughputMbps;
    double maxThroughputMbps;
    double throughputStdDev;

    // Latency
    double avgPredictionLatencyMs;  // Should be ~2-5ms
    double p95PredictionLatencyMs;  // Should be <10ms

    // Rate adaptation
    double avgRateIndex;            // Should be 5-7 for good SNR
    int rateChangesPerSecond;       // ML should be lower (more stable)
    double timeToOptimalRate;       // ML should be faster

    // Reliability
    double packetLossRate;          // Should be <10%
    double retransmissionRate;      // Should be <15%

    // Comparison
    double throughputGainVsMinstrel;  // Target: +10-20%
    double latencyReduction;          // Target: -50% (faster decisions)
};
```

---

## **🚀 NEXT STEPS FOR CC FILES**

### **1. Create ML Rate Control Algorithm Class**

```cpp
class MLRateControl : public WifiRemoteStationManager {
private:
    int m_mlServerSocket;
    std::string m_mlServerHost;
    int m_mlServerPort;
    MLModel m_activeModel;

    // Feature tracking
    double m_lastSnr;
    double m_snrFast;
    double m_snrSlow;
    std::deque<double> m_snrHistory;

public:
    // Core methods
    WifiTxVector DoGetDataTxVector(WifiRemoteStation* station) override;
    void DoReportRxOk(WifiRemoteStation* station, double rxSnr, ...) override;
    void DoReportDataFailed(WifiRemoteStation* station) override;

    // ML-specific
    int QueryMLServer(const std::vector<double>& features);
    void UpdateFeatures(double newSnr);
};
```

### **2. Socket Communication Helper**

```cpp
class MLServerClient {
public:
    static int ConnectToServer(const std::string& host, int port);
    static std::string SendRequest(int socket, const std::string& request);
    static int ParseRateFromResponse(const std::string& response);
    static void CloseConnection(int socket);
};
```

### **3. Feature Calculator**

```cpp
class WiFiFeatureCalculator {
public:
    static std::vector<double> CalculateFeatures(
        double lastSnr,
        const std::deque<double>& snrHistory,
        double channelWidth,
        double mobilityMetric
    );

private:
    static double CalculateEWMA(double prev, double curr, double alpha);
    static double CalculateVariance(const std::deque<double>& data);
    static double CalculateTrend(double fast, double slow);
};
```

---

## **📝 FINAL CHECKLIST BEFORE CC IMPLEMENTATION**

- [ ] Understand 9 safe features (NO outcome features!)
- [ ] Know feature order (lastSnr first, mobilityMetric last)
- [ ] Default model is oracle_aggressive (62.8% accuracy)
- [ ] Server expects space-separated features + newline
- [ ] Rate index range: 0-7 (802.11a rates)
- [ ] Expected latency: 2-5ms per prediction
- [ ] Fallback to Minstrel if server unavailable
- [ ] Track both ML and Minstrel metrics for comparison
- [ ] Test scenarios: static high/low SNR, dynamic, mobility, interference
- [ ] Target: +10-20% throughput vs Minstrel

---

**Ready for CC files! All context preserved. Let me know when you need the actual ns-3 implementation!** 🚀
