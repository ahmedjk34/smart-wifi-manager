/*
 * Smart WiFi Manager with 9 Safe Features - NEW PIPELINE
 * Compatible with ahmedjk34's probabilistic oracle models (9 features, 45-63% realistic accuracy)
 *
 * CRITICAL UPDATES (2025-10-02 18:03:33 UTC):
 * ============================================================================
 * WHAT WE CHANGED FROM 14-FEATURE VERSION:
 * 1. Feature count: 14 → 9 (removed 5 outcome features)
 * 2. Removed previous/current window tracking (shortSuccRatio, medSuccRatio)
 * 3. Removed packetLossRate, severity, confidence tracking
 * 4. Updated ExtractFeatures() to return 9 features (not 14)
 * 5. Model paths: root → python_files/trained_models/
 * 6. Python client integration via socket (port 8765)
 * 7. Default oracle: oracle_aggressive (62.8% test accuracy)
 *
 * WHY WE CHANGED IT:
 * - Your File 3 removed 5 outcome features due to data leakage
 * - Your trained models (File 4) expect exactly 9 features
 * - oracle_aggressive performs best (62.8% vs 45-48% for others)
 * - Python server handles ML inference (clean separation)
 * - Socket communication to localhost:8765 (your 6a server)
 *
 * NEW FEATURE LIST (9 safe features):
 * ✓ 1. lastSnr (dB)               - Most recent realistic SNR
 * ✓ 2. snrFast (dB)               - Fast-moving average (α=0.1)
 * ✓ 3. snrSlow (dB)               - Slow-moving average (α=0.01)
 * ✓ 4. snrTrendShort              - Short-term trend
 * ✓ 5. snrStabilityIndex          - Stability metric (0-10)
 * ✓ 6. snrPredictionConfidence    - Prediction confidence (0-1)
 * ✓ 7. snrVariance                - SNR variance (0-100)
 * ✓ 8. channelWidth (MHz)         - Channel bandwidth (20, 40, 80, 160)
 * ✓ 9. mobilityMetric             - Node mobility (0-50)
 *
 * REMOVED FEATURES (from 14-feature version):
 * ❌ shortSuccRatio               - Outcome-based (removed in File 3)
 * ❌ medSuccRatio                 - Outcome-based (removed in File 3)
 * ❌ packetLossRate               - Outcome-based (removed in File 3)
 * ❌ severity                     - Derived from outcomes (removed in File 3)
 * ❌ confidence                   - Derived from outcomes (removed in File 3)
 *
 * PYTHON SERVER INTEGRATION:
 * - Server: python_files/6a_ml_inference_server.py (must be running!)
 * - Port: 8765 (default, configurable via InferenceServerPort attribute)
 * - Protocol: Space-separated features + optional model name + newline
 * - Example: "25.0 25.0 25.0 0.0 0.01 0.99 0.5 20.0 0.5 oracle_aggressive\n"
 * - Response: JSON with rateIdx, confidence, latencyMs, success, model
 *
 * USAGE:
 * 1. Start Python server: python3 python_files/6a_ml_inference_server.py
 * 2. Run ns-3 simulation with SmartWifiManagerRf
 * 3. C++ sends 9 features → Python server → ML inference → C++ receives rate
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-10-02 18:03:33 UTC
 * Version: 6.0 (NEW PIPELINE - 9 Features, Probabilistic Oracle)
 * License: Copyright (c) 2005,2006 INRIA
 */

#ifndef SMART_WIFI_MANAGER_RF_H
#define SMART_WIFI_MANAGER_RF_H

#include "ns3/mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/traced-value.h"
#include "ns3/vector.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-remote-station-manager.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <fcntl.h> // For fcntl, O_NONBLOCK
#include <mutex>
#include <string>
#include <sys/select.h> // For select()
#include <vector>

namespace ns3
{

/**
 * Enhanced WiFi context classification for intelligent rate adaptation
 * Based on REALISTIC SNR and network conditions
 */
enum class WifiContextType
{
    EMERGENCY,        // Critical network conditions - use lowest rates
    POOR_UNSTABLE,    // Poor signal with instability
    MARGINAL,         // Marginal signal quality
    GOOD_UNSTABLE,    // Good signal but unstable
    GOOD_STABLE,      // Good signal and stable
    EXCELLENT_STABLE, // Excellent signal and very stable
    UNKNOWN           // Unknown/unclassified state
};

// Forward declarations
class SmartWifiManagerRfState;
class SmartWifiManagerRf;

/**
 * \brief Safety assessment structure for network conditions
 */
struct SafetyAssessment
{
    WifiContextType context;
    double riskLevel;
    uint32_t recommendedSafeRate;
    bool requiresEmergencyAction;
    double confidenceInAssessment;
    std::string contextStr;

    SmartWifiManagerRf* managerRef;
    uint32_t stationId;

    SafetyAssessment()
        : context(WifiContextType::UNKNOWN),
          riskLevel(0.0),
          recommendedSafeRate(3),
          requiresEmergencyAction(false),
          confidenceInAssessment(1.0),
          contextStr("unknown"),
          managerRef(nullptr),
          stationId(0)
    {
    }
};

/**
 * \brief NEW PIPELINE Smart Rate control algorithm using Python ML server
 * \ingroup wifi
 *
 * NEW PIPELINE VERSION (2025-10-02):
 * - 9 safe features (no temporal leakage, no outcome features)
 * - Python server integration via socket (port 8765)
 * - oracle_aggressive default (62.8% test accuracy)
 * - Models located in python_files/trained_models/
 * - Thread-safe operations
 * - Realistic SNR conversion (-30dB to +45dB)
 *
 * Key changes from 14-feature version:
 * - Removed 5 outcome features
 * - Removed previous/current window tracking
 * - Updated feature extraction to 9 features
 * - Python client integration (socket communication)
 * - Model expects exactly 9 features
 */
class SmartWifiManagerRf : public WifiRemoteStationManager
{
  public:
    static TypeId GetTypeId();
    SmartWifiManagerRf();
    ~SmartWifiManagerRf() override;

    /**
     * \brief Result structure for ML inference operations
     */
    struct InferenceResult
    {
        uint32_t rateIdx;
        double latencyMs;
        bool success;
        std::string error;
        double confidence;
        std::string model;
        std::vector<double> classProbabilities;

        InferenceResult()
            : rateIdx(3),
              latencyMs(0.0),
              success(false),
              confidence(0.0),
              model("none")
        {
        }
    };

    // Configuration methods (thread-safe)
    void SetBenchmarkDistance(double distance);
    void SetModelName(const std::string& modelName);
    void SetOracleStrategy(const std::string& strategy);
    void SetCurrentInterferers(uint32_t interferers);
    void UpdateFromBenchmarkGlobals(double distance, uint32_t interferers);
    void DebugPrintCurrentConfig() const;

    // Thread-safe getters
    double GetCurrentBenchmarkDistance() const
    {
        return m_benchmarkDistance.load();
    }

    uint32_t GetCurrentInterfererCount() const
    {
        return m_currentInterferers.load();
    }

    // Station registry for safe access
    SmartWifiManagerRfState* GetStationById(uint32_t stationId) const;
    uint32_t RegisterStation(SmartWifiManagerRfState* station);

  private:
    // Core WifiRemoteStationManager interface
    void DoInitialize() override;
    WifiRemoteStation* DoCreateStation() const override;
    void DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode) override;
    void DoReportRtsFailed(WifiRemoteStation* station) override;
    void DoReportDataFailed(WifiRemoteStation* station) override;
    void DoReportRtsOk(WifiRemoteStation* station,
                       double ctsSnr,
                       WifiMode ctsMode,
                       double rtsSnr) override;
    void DoReportDataOk(WifiRemoteStation* station,
                        double ackSnr,
                        WifiMode ackMode,
                        double dataSnr,
                        uint16_t dataChannelWidth,
                        uint8_t dataNss) override;
    void DoReportFinalRtsFailed(WifiRemoteStation* station) override;
    void DoReportFinalDataFailed(WifiRemoteStation* station) override;
    WifiTxVector DoGetDataTxVector(WifiRemoteStation* station, uint16_t allowedWidth) override;
    WifiTxVector DoGetRtsTxVector(WifiRemoteStation* station) override;

    // NEW PIPELINE: ML inference with 9 safe features via Python server
    InferenceResult RunMLInference(const std::vector<double>& features) const;
    std::vector<double> ExtractFeatures(WifiRemoteStation* station) const;
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr);

    // NEW PIPELINE: Safe feature calculation methods (9 features only!)
    double GetSnrTrendShort(WifiRemoteStation* station) const;
    double GetSnrStabilityIndex(WifiRemoteStation* station) const;
    double GetSnrPredictionConfidence(WifiRemoteStation* station) const;
    double GetMobilityMetric(WifiRemoteStation* station) const;

    // Enhanced SNR modeling with consistent pipeline
    double ConvertToRealisticSnr(double ns3Snr) const;

    // Context and safety assessment
    SafetyAssessment AssessNetworkSafety(SmartWifiManagerRfState* station);
    WifiContextType ClassifyNetworkContext(SmartWifiManagerRfState* station) const;
    std::string ContextTypeToString(WifiContextType type) const;
    double CalculateRiskLevel(SmartWifiManagerRfState* station) const;
    uint32_t GetContextSafeRate(SmartWifiManagerRfState* station, WifiContextType context) const;

    // Enhanced rate decision algorithms
    uint32_t GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station,
                                      const SafetyAssessment& safety) const;
    uint32_t FuseMLAndRuleBased(uint32_t mlRate,
                                uint32_t ruleRate,
                                double mlConfidence,
                                const SafetyAssessment& safety,
                                SmartWifiManagerRfState* station) const;

    // Configuration parameters
    std::string m_modelPath;
    std::string m_scalerPath;
    std::string m_modelType;
    std::string m_modelName;
    std::string m_oracleStrategy;
    uint16_t m_inferenceServerPort;

    bool m_useRealisticSnr;
    double m_maxSnrDb;
    double m_minSnrDb;
    double m_snrOffset;

    double m_confidenceThreshold;
    double m_riskThreshold;
    uint32_t m_failureThreshold;
    double m_mlGuidanceWeight;
    uint32_t m_mlCacheTime;
    bool m_enableAdaptiveWeighting;

    uint32_t m_inferencePeriod;
    uint32_t m_fallbackRate;
    uint32_t m_windowSize;
    double m_snrAlpha;

    std::atomic<double> m_benchmarkDistance;
    std::atomic<uint32_t> m_currentInterferers;
    std::atomic<double> m_benchmarkSpeed;
    std::atomic<uint32_t> m_benchmarkPacketSize{1200};

    std::vector<WifiMode> m_supportedRates;
    bool m_enableDetailedLogging;

    TracedValue<uint64_t> m_currentRate;
    TracedValue<uint32_t> m_mlInferences;
    TracedValue<uint32_t> m_mlFailures;
    TracedValue<uint32_t> m_mlCacheHits;
    TracedValue<double> m_avgMlLatency;

    mutable std::mutex m_mlCacheMutex;
    mutable uint32_t m_lastMlRate;
    mutable Time m_lastMlTime;
    mutable double m_lastMlConfidence;
    mutable std::string m_lastMlModel;

    mutable std::mutex m_stationRegistryMutex;
    mutable std::map<uint32_t, SmartWifiManagerRfState*> m_stationRegistry;
    mutable std::atomic<uint32_t> m_nextStationId;

    double CalculateAdaptiveConfidenceThreshold(SmartWifiManagerRfState* station,
                                                WifiContextType context) const;

    double GetBenchmarkSpeedAttribute() const;

    // 🚀 PHASE 2: Scenario-aware selection
    bool m_enableScenarioAwareSelection;
    mutable std::string m_currentModelName;

    // 🚀 PHASE 3: Hysteresis configuration
    uint32_t m_hysteresisStreak;

    // Add to SmartWifiManagerRf class private methods (around line 300-400):

    // 🚀 PHASE 1A
    void UpdateEnhancedFeatures(SmartWifiManagerRfState* station);

    // 🚀 PHASE 2
    std::string SelectBestModel(SmartWifiManagerRfState* station) const;

    // 🚀 PHASE 3
    uint8_t ApplyHysteresis(SmartWifiManagerRfState* station,
                            uint8_t currentRate,
                            uint8_t predictedRate) const;

    // 🚀 PHASE 4
    double CalculateAdaptiveTrust(double mlConfidence, SmartWifiManagerRfState* station) const;
    uint32_t AdaptiveFusion(uint8_t mlRate,
                            uint8_t ruleRate,
                            double mlConfidence,
                            SmartWifiManagerRfState* station) const;

    // Update signature of RunMLInference:
    InferenceResult RunMLInference(const std::vector<double>& features,
                                   const std::string& modelName = "") const;

    // 🚀 ATTRIBUTE SETTERS/GETTERS (for guaranteed sync)
    void SetBenchmarkDistanceAttribute(double dist);
    double GetBenchmarkDistanceAttribute() const;
    void SetInterferersAttribute(uint32_t count);
    uint32_t GetInterferersAttribute() const;

    void SetBenchmarkSpeed(double speed);

    uint32_t GetBenchmarkPacketSizeAttribute() const;
    void SetBenchmarkPacketSizeAttribute(uint32_t pktSize);
};

/**
 * \brief NEW PIPELINE SmartWifiManagerRf station state (9 features)
 *
 * CRITICAL CHANGES:
 * - Removed previousShortWindow, previousMedWindow (outcome features)
 * - Removed currentShortWindow, currentMedWindow (outcome features)
 * - Removed severity, confidence (outcome features)
 * - Removed all packet tracking windows
 * - Only SNR metrics and environmental features remain
 */
struct SmartWifiManagerRfState : public WifiRemoteStation
{
    uint32_t stationId;

    // NEW PIPELINE: Core SNR metrics (SAFE - pre-decision measurements)
    double lastSnr;
    double lastRawSnr;
    double snrFast;
    double snrSlow;
    double snrTrendShort;
    double snrStabilityIndex;
    double snrPredictionConfidence;
    double snrVariance;

    // Timing features (SAFE - environmental)
    Time lastUpdateTime;
    Time lastInferenceTime;
    Time lastRateChangeTime;

    // Network state (SAFE - environmental)
    double mobilityMetric;
    Vector lastPosition;
    uint32_t currentRateIndex;
    uint32_t previousRateIndex;

    // Context tracking (SAFE - assessment only)
    WifiContextType lastContext;
    double lastRiskLevel;

    // Packet tracking (SAFE - historical only)
    uint32_t totalPackets;
    uint32_t lostPackets;

    // SNR history (SAFE - measurements)
    std::deque<double> snrHistory;
    std::deque<double> rawSnrHistory;

    // ML interaction tracking
    uint32_t mlInferencesReceived;
    uint32_t mlInferencesSuccessful;
    double avgMlConfidence;
    std::string preferredModel;

    // ML performance tracking
    uint32_t lastMLInfluencedRate;
    Time lastMLInfluenceTime;
    double mlPerformanceScore;
    uint32_t mlSuccessfulPredictions;
    double mlContextConfidence[6];
    uint32_t mlContextUsage[6];
    double recentMLAccuracy;
    Time lastMLPerformanceUpdate;

    static constexpr uint32_t WINDOW_SIZE = 50;

    // 🚀 PHASE 1B: NEW FEATURE TRACKING (4 features)
    double rssiVariance;      // RSSI variance (dB²)
    double interferenceLevel; // Interference level (0-1)
    double distanceMetric;    // Distance from AP (meters)
    double avgPacketSize;     // Average packet size (bytes)

    double retryRate{0.0};
    double frameErrorRate{0.0};

    uint32_t packetsSinceRateChange{0};
    uint32_t successfulPackets{0};
    uint32_t failedPackets{0};
    std::deque<uint32_t> recentRateHistory;

    // 🚀 PHASE 3: Hysteresis tracking
    uint32_t ratePredictionStreak{0};
    uint8_t lastPredictedRate{3};
    uint32_t rateStableCount{0};

    // In SmartWifiManagerRfState struct:
    uint32_t consecutiveFailures = 0;  // Track consecutive failures (AARF-style)
    uint32_t consecutiveSuccesses = 0; // Track consecutive successes (AARF-style)

    Time lastModelSwitchTime{Seconds(0)}; // Add this line

    SmartWifiManagerRfState()
        : stationId(0),
          lastSnr(0.0),
          lastRawSnr(0.0),
          snrFast(0.0),
          snrSlow(0.0),
          snrTrendShort(0.0),
          snrStabilityIndex(1.0),
          snrPredictionConfidence(0.8),
          snrVariance(0.1),
          lastUpdateTime(Seconds(0)),
          lastInferenceTime(Seconds(0)),
          lastRateChangeTime(Seconds(0)),
          mobilityMetric(0.0),
          lastPosition(Vector(0, 0, 0)),
          currentRateIndex(3),
          previousRateIndex(3),
          lastContext(WifiContextType::UNKNOWN),
          lastRiskLevel(0.0),
          totalPackets(0),
          lostPackets(0),
          mlInferencesReceived(0),
          mlInferencesSuccessful(0),
          avgMlConfidence(0.3),
          preferredModel("oracle_aggressive"),
          lastMLInfluencedRate(3),
          lastMLInfluenceTime(Seconds(0)),
          mlPerformanceScore(0.5),
          mlSuccessfulPredictions(0),
          recentMLAccuracy(0.5),
          lastMLPerformanceUpdate(Seconds(0))
    {
        for (int i = 0; i < 6; i++)
        {
            mlContextConfidence[i] = 0.3;
            mlContextUsage[i] = 0;
        }
    }
};

} // namespace ns3

#endif /* SMART_WIFI_MANAGER_RF_H */