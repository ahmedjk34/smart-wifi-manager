/*
 * Smart WiFi Manager with 14 Safe Features - FIXED FOR ZERO-LEAKAGE PIPELINE
 * Compatible with ahmedjk34's FIXED ML Pipeline (14 features, realistic 65-75% accuracy)
 *
 * CRITICAL FIXES (2025-10-01):
 * - Issue #1: Removed ALL 7 temporal leakage features
 * - Issue #33: Success ratios from PREVIOUS window (not current packet)
 * - Issue #4: Compatible with scenario_file for train/test splitting
 * - 14 safe features matching Files 1-5b fixed pipeline
 * - 802.11a support (8 rates: 0-7)
 * - Realistic accuracy expectations: 65-75% (not 95%+ fake accuracy)
 *
 * REMOVED FEATURES (Temporal Leakage):
 * ❌ consecSuccess, consecFailure - outcomes of CURRENT rate
 * ❌ retrySuccessRatio - derived from outcomes
 * ❌ timeSinceLastRateChange, rateStabilityScore, recentRateChanges - rate history
 * ❌ packetSuccess - literal packet outcome
 *
 * SAFE FEATURES (14 total):
 * ✓ SNR features (7): lastSnr, snrFast, snrSlow, snrTrendShort, snrStabilityIndex,
 *                      snrPredictionConfidence, snrVariance
 * ✓ Previous window success (2): shortSuccRatio, medSuccRatio (from PREVIOUS window)
 * ✓ Previous window loss (1): packetLossRate (from PREVIOUS window)
 * ✓ Network state (2): channelWidth, mobilityMetric
 * ✓ Assessment (2): severity, confidence (from previous window)
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-10-01 14:47:23 UTC
 * Version: 5.0 (FIXED - Zero Temporal Leakage)
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
#include <mutex>
#include <string>
#include <vector>

namespace ns3
{

/**
 * Enhanced WiFi context classification for intelligent rate adaptation
 * Based on REALISTIC SNR, success rate, and network conditions
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
 * FIXED: Proper design with no const_cast hacks
 */
struct SafetyAssessment
{
    WifiContextType context;       // Current network context
    double riskLevel;              // Risk level (0.0-1.0)
    uint32_t recommendedSafeRate;  // Safe rate for current conditions
    bool requiresEmergencyAction;  // Whether emergency action needed
    double confidenceInAssessment; // Confidence in assessment
    std::string contextStr;        // Human-readable context string

    // Reference management
    SmartWifiManagerRf* managerRef; // Manager reference
    uint32_t stationId;             // Station identifier

    // Constructor with proper initialization
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
 * \brief FIXED Smart Rate control algorithm using Random Forest ML models
 * \ingroup wifi
 *
 * FIXED VERSION (2025-10-01):
 * - 14 safe features (no temporal leakage)
 * - Success ratios from PREVIOUS window (Issue #33)
 * - Compatible with fixed pipeline (65-75% realistic accuracy)
 * - 802.11a support (8 rates: 0-7)
 * - Thread-safe operations
 *
 * Key changes from v4.0:
 * - Removed 7 temporal leakage features
 * - Added previous/current window tracking
 * - Updated feature extraction to 14 features
 * - Model expects 14 features (not 21)
 * - Realistic SNR conversion (-30dB to +45dB)
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
        uint32_t rateIdx;                       // Predicted rate index (0-7)
        double latencyMs;                       // Inference latency in milliseconds
        bool success;                           // Whether inference succeeded
        std::string error;                      // Error message if failed
        double confidence;                      // ML model confidence (0.0-1.0)
        std::string model;                      // Model name used for prediction
        std::vector<double> classProbabilities; // Per-class probabilities (optional)

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

    // FIXED: ML inference with 14 safe features
    InferenceResult RunMLInference(const std::vector<double>& features) const;
    std::vector<double> ExtractFeatures(WifiRemoteStation* station) const;
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr);

    // FIXED: Safe feature calculation methods (14 features)
    double GetOfferedLoad() const;
    double GetMobilityMetric(WifiRemoteStation* station) const;
    double GetPacketLossRate(WifiRemoteStation* station) const;
    double GetSnrTrendShort(WifiRemoteStation* station) const;
    double GetSnrStabilityIndex(WifiRemoteStation* station) const;
    double GetSnrPredictionConfidence(WifiRemoteStation* station) const;

    // FIXED: Issue #33 - Success ratios from PREVIOUS window
    double GetPreviousShortSuccessRatio(WifiRemoteStation* station) const;
    double GetPreviousMedSuccessRatio(WifiRemoteStation* station) const;
    void UpdateWindowState(WifiRemoteStation* station);

    // Enhanced SNR modeling with consistent pipeline
    double CalculateDistanceBasedSnr(WifiRemoteStation* st) const;
    double ApplyRealisticSnrBounds(double snr) const;
    double ConvertToRealisticSnr(double ns3Snr) const;

    // Context and safety assessment
    SafetyAssessment AssessNetworkSafety(SmartWifiManagerRfState* station);
    WifiContextType ClassifyNetworkContext(SmartWifiManagerRfState* station) const;
    std::string ContextTypeToString(WifiContextType type) const;
    double CalculateRiskLevel(SmartWifiManagerRfState* station) const;
    uint32_t GetContextSafeRate(SmartWifiManagerRfState* station, WifiContextType context) const;

    // Enhanced rate decision algorithms
    uint32_t GetRuleBasedRate(SmartWifiManagerRfState* station) const;
    uint32_t GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station,
                                      const SafetyAssessment& safety) const;
    uint32_t FuseMLAndRuleBased(uint32_t mlRate,
                                uint32_t ruleRate,
                                double mlConfidence,
                                const SafetyAssessment& safety,
                                SmartWifiManagerRfState* station) const;

    // Enhanced logging
    void LogContextAndDecision(const SafetyAssessment& safety,
                               uint32_t mlRate,
                               uint32_t ruleRate,
                               uint32_t finalRate) const;
    void LogFeatureVector(const std::vector<double>& features, const std::string& context) const;

    // Configuration parameters
    std::string m_modelPath;        // Path to ML model file
    std::string m_scalerPath;       // Path to scaler file
    std::string m_pythonScript;     // Legacy python script path
    std::string m_modelType;        // Model type (oracle/v3)
    std::string m_modelName;        // Specific model name
    std::string m_oracleStrategy;   // Oracle strategy selection
    bool m_enableProbabilities;     // Enable probability output
    bool m_enableValidation;        // Enable feature validation
    uint32_t m_maxInferenceTime;    // Max inference time (ms)
    uint32_t m_windowSize;          // Success ratio window size
    double m_snrAlpha;              // SNR exponential smoothing alpha
    uint32_t m_inferencePeriod;     // ML inference period
    uint32_t m_fallbackRate;        // Fallback rate index
    bool m_enableFallback;          // Enable fallback mechanism
    uint16_t m_inferenceServerPort; // ML inference server port

    // Enhanced SNR modeling parameters
    bool m_useRealisticSnr; // Use realistic SNR calculation
    double m_maxSnrDb;      // Maximum realistic SNR (dB)
    double m_minSnrDb;      // Minimum realistic SNR (dB)
    double m_snrOffset;     // SNR offset for calibration (dB)

    // Fusion and safety parameters
    double m_confidenceThreshold;   // Min ML confidence threshold
    double m_riskThreshold;         // Max risk threshold
    uint32_t m_failureThreshold;    // Consecutive failure threshold
    double m_mlGuidanceWeight;      // ML guidance weight
    uint32_t m_mlCacheTime;         // ML result cache time (ms)
    bool m_enableAdaptiveWeighting; // Enable adaptive ML weighting
    double m_conservativeBoost;     // Conservative rate boost factor

    // Distance tracking (thread-safe)
    std::atomic<double> m_benchmarkDistance;
    std::atomic<uint32_t> m_currentInterferers;

    // Available data rates
    std::vector<WifiMode> m_supportedRates;

    // Enhanced logging
    bool m_enableDetailedLogging;

    // Traced values for monitoring
    TracedValue<uint64_t> m_currentRate;
    TracedValue<uint32_t> m_mlInferences;
    TracedValue<uint32_t> m_mlFailures;
    TracedValue<uint32_t> m_mlCacheHits;
    TracedValue<double> m_avgMlLatency;

    // ML result caching (thread-safe)
    mutable std::mutex m_mlCacheMutex;
    mutable uint32_t m_lastMlRate;
    mutable Time m_lastMlTime;
    mutable double m_lastMlConfidence;
    mutable std::string m_lastMlModel;

    // Station registry
    mutable std::mutex m_stationRegistryMutex;
    mutable std::map<uint32_t, SmartWifiManagerRfState*> m_stationRegistry;
    mutable std::atomic<uint32_t> m_nextStationId;

    double CalculateAdaptiveConfidenceThreshold(SmartWifiManagerRfState* station,
                                                WifiContextType context) const;
};

/**
 * \brief FIXED SmartWifiManagerRf station state with PREVIOUS window tracking
 *
 * CRITICAL FIXES (Issue #33):
 * - Added previousShortWindow, previousMedWindow for safe success ratios
 * - Added currentShortWindow, currentMedWindow for ongoing tracking
 * - Window state updated at packet boundaries (not during decision)
 * - Success ratios calculated from PREVIOUS window only
 *
 * REMOVED FEATURES (Issue #1 - Temporal Leakage):
 * - consecSuccess, consecFailure (outcome of CURRENT rate)
 * - retryWindow, retryCount tracking (retry outcomes)
 * - rateHistory, rateChangeCount (rate adaptation history)
 * - lastPacketSuccess (literal outcome)
 */
struct SmartWifiManagerRfState : public WifiRemoteStation
{
    // Station identification
    uint32_t stationId;

    // FIXED: Core SNR metrics (SAFE - pre-decision measurements)
    double lastSnr;                 // Most recent REALISTIC SNR (-30 to +45 dB)
    double lastRawSnr;              // Raw NS-3 SNR (for debugging)
    double snrFast;                 // Fast-moving REALISTIC SNR average
    double snrSlow;                 // Slow-moving REALISTIC SNR average
    double snrTrendShort;           // Short-term SNR trend
    double snrStabilityIndex;       // SNR stability metric
    double snrPredictionConfidence; // Confidence in SNR predictions
    double snrVariance;             // SNR variance

    // FIXED: Issue #33 - PREVIOUS window tracking (SAFE)
    std::deque<bool> previousShortWindow; // Previous short window (SAFE for logging)
    std::deque<bool> previousMedWindow;   // Previous medium window (SAFE for logging)
    uint32_t previousWindowSuccess;       // Success count in previous window
    uint32_t previousWindowTotal;         // Total packets in previous window
    uint32_t previousWindowLosses;        // Losses in previous window

    // FIXED: Issue #33 - CURRENT window tracking (NOT logged)
    std::deque<bool> currentShortWindow; // Current short window (becomes previous)
    std::deque<bool> currentMedWindow;   // Current medium window (becomes previous)
    uint32_t currentWindowPackets;       // Packet count in current window

    // Network condition assessment (SAFE - from previous window)
    double severity;   // Network condition severity (0.0-1.0)
    double confidence; // Confidence in current assessment (0.0-1.0)

    // Timing features (SAFE - environmental)
    uint32_t T1, T2, T3;
    Time lastUpdateTime;
    Time lastInferenceTime;
    Time lastRateChangeTime;

    // Network state (SAFE - environmental)
    double mobilityMetric;
    Vector lastPosition;
    uint32_t currentRateIndex;
    uint32_t previousRateIndex;
    uint32_t queueLength;

    // Context tracking (SAFE - assessment only)
    WifiContextType lastContext;
    double lastRiskLevel;
    uint32_t decisionReason;

    // Packet tracking (SAFE - historical only)
    uint32_t totalPackets;
    uint32_t lostPackets;

    // SNR history (SAFE - measurements)
    std::deque<double> snrHistory;
    std::deque<double> rawSnrHistory;
    std::deque<Time> changeTimeHistory;

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

    // Window size constant
    static constexpr uint32_t WINDOW_SIZE = 50;

    // FIXED: Constructor with proper initialization
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
          previousWindowSuccess(0),
          previousWindowTotal(0),
          previousWindowLosses(0),
          currentWindowPackets(0),
          severity(0.0),
          confidence(1.0),
          T1(0),
          T2(0),
          T3(0),
          lastUpdateTime(Seconds(0)),
          lastInferenceTime(Seconds(0)),
          lastRateChangeTime(Seconds(0)),
          mobilityMetric(0.0),
          lastPosition(Vector(0, 0, 0)),
          currentRateIndex(3),
          previousRateIndex(3),
          queueLength(0),
          lastContext(WifiContextType::UNKNOWN),
          lastRiskLevel(0.0),
          decisionReason(0),
          totalPackets(0),
          lostPackets(0),
          mlInferencesReceived(0),
          mlInferencesSuccessful(0),
          avgMlConfidence(0.3),
          preferredModel("oracle_balanced"),
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