/*
 * Enhanced Smart WiFi Manager with 21 Safe Features - FIXED SNR ENGINE
 * Compatible with ahmedjk34's Enhanced ML Pipeline (49.9% realistic accuracy)
 *
 * Features:
 * - 21 safe features (no data leakage)  // FIXED: Changed from 28 to 21
 * - Multiple oracle strategy support (oracle_balanced, oracle_conservative, oracle_aggressive,
 * rateIdx)
 * - Production-ready inference server integration
 * - Enhanced context classification and safety assessment
 * - Real-time rate adaptation with ML guidance
 * - FIXED: Consistent realistic SNR calculation engine
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-09-28  // FIXED: Updated date
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

#include <chrono>
#include <deque>
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

class SmartWifiManagerRfState; // Forward declaration

/**
 * \brief Enhanced Smart Rate control algorithm using Random Forest ML models - FIXED SNR ENGINE
 * \ingroup wifi
 *
 * This class implements an intelligent WiFi rate adaptation algorithm that combines:
 * - Machine Learning guidance from trained Random Forest models (98.1% accuracy)
 * - Rule-based safety mechanisms for reliability
 * - Context-aware risk assessment
 * - 21 safe features with no data leakage
 * - FIXED: Consistent realistic SNR calculation (-30dB to +45dB)
 *
 * Key innovations:
 * - Supports multiple oracle strategies (balanced, conservative, aggressive)
 * - Real-time inference server integration with caching
 * - Enhanced SNR modeling for realistic simulation
 * - Production-grade error handling and fallback mechanisms
 * - FIXED: Unified SNR processing pipeline
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
    };

    /**
     * \brief Safety assessment structure for network conditions
     */
    struct SafetyAssessment
    {
        WifiContextType context;          // Current network context
        double riskLevel;                 // Risk level (0.0-1.0)
        uint32_t recommendedSafeRate;     // Safe rate for current conditions
        bool requiresEmergencyAction;     // Whether emergency action needed
        double confidenceInAssessment;    // Confidence in assessment
        std::string contextStr;           // Human-readable context string
        SmartWifiManagerRfState* station; // Pointer to station for fusion access
    };

    // Enhanced configuration methods
    void SetBenchmarkDistance(double distance);
    void SetModelName(const std::string& modelName);
    void SetOracleStrategy(const std::string& strategy);
    void SetCurrentInterferers(uint32_t interferers);

    // FIXED: Synchronization method for benchmark coordination
    void UpdateFromBenchmarkGlobals(double distance, uint32_t interferers);

    // Config debugging
    void DebugPrintCurrentConfig() const;

    double GetCurrentBenchmarkDistance() const
    {
        return m_benchmarkDistance;
    }

    uint32_t GetCurrentInterfererCount() const
    {
        return m_currentInterferers;
    }

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

    // Enhanced ML inference with 21 safe features
    InferenceResult RunMLInference(const std::vector<double>& features) const;
    std::vector<double> ExtractFeatures(WifiRemoteStation* station) const;
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr);

    // Enhanced feature calculation methods (21 safe features)
    double GetOfferedLoad() const;
    double GetMobilityMetric(WifiRemoteStation* station) const;
    double GetRetrySuccessRatio(WifiRemoteStation* station) const;
    uint32_t GetRecentRateChanges(WifiRemoteStation* station) const;
    double GetTimeSinceLastRateChange(WifiRemoteStation* station) const;
    double GetRateStabilityScore(WifiRemoteStation* station) const;
    double GetPacketLossRate(WifiRemoteStation* station) const;
    double GetSnrTrendShort(WifiRemoteStation* station) const;
    double GetSnrStabilityIndex(WifiRemoteStation* station) const;
    double GetSnrPredictionConfidence(WifiRemoteStation* station) const;

    // FIXED: Enhanced SNR modeling with consistent pipeline
    double CalculateDistanceBasedSnr(WifiRemoteStation* st) const;
    double ApplyRealisticSnrBounds(double snr) const;
    double ConvertToRealisticSnr(double ns3Snr) const;

    // Enhanced context and safety assessment
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
                                const SafetyAssessment& safety) const;

    // Enhanced logging and debugging
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
    std::string m_modelName;        // Specific model name (oracle_balanced, etc.)
    std::string m_oracleStrategy;   // Oracle strategy selection
    bool m_enableProbabilities;     // Enable probability output
    bool m_enableValidation;        // Enable feature validation
    uint32_t m_maxInferenceTime;    // Max inference time (ms)
    uint32_t m_windowSize;          // Success ratio window size
    double m_snrAlpha;              // SNR exponential smoothing alpha
    uint32_t m_inferencePeriod;     // ML inference period (transmissions)
    uint32_t m_fallbackRate;        // Fallback rate index
    bool m_enableFallback;          // Enable fallback mechanism
    uint16_t m_inferenceServerPort; // ML inference server port

    // FIXED: Enhanced SNR modeling parameters
    bool m_useRealisticSnr; // Use realistic SNR calculation
    double m_maxSnrDb;      // Maximum realistic SNR (dB)
    double m_minSnrDb;      // Minimum realistic SNR (dB)
    double m_snrOffset;     // SNR offset for calibration (dB)

    // Enhanced fusion and safety parameters
    double m_confidenceThreshold;   // Min ML confidence threshold
    double m_riskThreshold;         // Max risk threshold
    uint32_t m_failureThreshold;    // Consecutive failure threshold
    double m_mlGuidanceWeight;      // ML guidance weight (0.0-1.0)
    uint32_t m_mlCacheTime;         // ML result cache time (ms)
    bool m_enableAdaptiveWeighting; // Enable adaptive ML weighting
    double m_conservativeBoost;     // Conservative rate boost factor

    // FIXED: Distance for realistic SNR conversion - CRITICAL
    double m_benchmarkDistance;
    uint32_t m_currentInterferers;

    // Available data rates for 802.11g
    std::vector<WifiMode> m_supportedRates;

    // Enhanced logging
    bool m_enableDetailedLogging; // For enhanced logging

    // Enhanced traced values for monitoring
    TracedValue<uint64_t> m_currentRate;  // Current data rate
    TracedValue<uint32_t> m_mlInferences; // Total ML inferences
    TracedValue<uint32_t> m_mlFailures;   // Total ML failures
    TracedValue<uint32_t> m_mlCacheHits;  // ML cache hits
    TracedValue<double> m_avgMlLatency;   // Average ML latency

    // Enhanced ML result caching
    mutable uint32_t m_lastMlRate;     // Cached ML rate result
    mutable Time m_lastMlTime;         // Time of last ML inference
    mutable double m_lastMlConfidence; // Cached ML confidence
    mutable std::string m_lastMlModel; // Cached model name

    double CalculateAdaptiveConfidenceThreshold(SmartWifiManagerRfState* station,
                                                WifiContextType context) const;
};

/**
 * \brief FIXED SmartWifiManagerRf station state with consistent SNR handling
 *
 * This structure maintains all necessary state for intelligent rate adaptation
 * including the 21 safe features required by the enhanced ML pipeline.
 * FIXED: Consistent SNR storage and processing.
 */
struct SmartWifiManagerRfState : public WifiRemoteStation
{
    // FIXED: Core SNR metrics with clear separation
    double lastSnr;                 // Most recent REALISTIC SNR measurement (-30 to +45 dB)
    double lastRawSnr;              // Raw NS-3 SNR value (for debugging)
    double snrFast;                 // Fast-moving REALISTIC SNR average
    double snrSlow;                 // Slow-moving REALISTIC SNR average
    double snrTrendShort;           // Short-term SNR trend
    double snrStabilityIndex;       // SNR stability metric
    double snrPredictionConfidence; // Confidence in SNR predictions
    double snrVariance;             // SNR variance

    // Success tracking windows
    std::deque<bool> shortWindow;  // Short-term success window
    std::deque<bool> mediumWindow; // Medium-term success window
    std::deque<bool> retryWindow;  // Retry success tracking
    uint32_t consecSuccess;        // Consecutive successes
    uint32_t consecFailure;        // Consecutive failures

    // Network condition assessment
    double severity;   // Network condition severity (0.0-1.0)
    double confidence; // Confidence in current assessment (0.0-1.0)

    // Enhanced timing features
    uint32_t T1, T2, T3;     // Timing features for ML
    Time lastUpdateTime;     // Last metrics update time
    Time lastInferenceTime;  // Last ML inference time
    Time lastRateChangeTime; // Last rate change time

    // Enhanced tracking metrics
    uint32_t retryCount;        // Current retry count
    double mobilityMetric;      // Mobility assessment metric
    Vector lastPosition;        // Last known position
    uint32_t currentRateIndex;  // Current rate index (0-7)
    uint32_t previousRateIndex; // Previous rate index
    uint32_t queueLength;       // Queue length estimate
    uint32_t rateChangeCount;   // Recent rate changes count

    // Context and risk tracking
    WifiContextType lastContext; // Last classified context
    double lastRiskLevel;        // Last calculated risk level
    uint32_t decisionReason;     // Decision reason code
    bool lastPacketSuccess;      // Last packet success status

    // Enhanced packet tracking for 21 features
    uint32_t totalPackets;      // Total packets transmitted
    uint32_t lostPackets;       // Total packets lost
    uint32_t totalRetries;      // Total retry attempts
    uint32_t successfulRetries; // Successful retries

    // FIXED: Performance history with clear SNR separation
    std::deque<uint32_t> rateHistory;   // Recent rate history
    std::deque<double> snrHistory;      // Recent REALISTIC SNR history
    std::deque<double> rawSnrHistory;   // Recent RAW NS-3 SNR history (for debugging)
    std::deque<Time> changeTimeHistory; // Rate change timing history

    // ML interaction tracking
    uint32_t mlInferencesReceived;   // ML inferences for this station
    uint32_t mlInferencesSuccessful; // Successful ML inferences
    double avgMlConfidence;          // Running average ML confidence
    std::string preferredModel;      // Preferred model for this station

    // ENHANCED ML PERFORMANCE TRACKING AND LEARNING
    uint32_t lastMLInfluencedRate;    // Last rate set with ML influence
    Time lastMLInfluenceTime;         // When ML last influenced a decision
    double mlPerformanceScore;        // Running score of ML performance (0.0-1.0)
    uint32_t mlSuccessfulPredictions; // Count of successful ML predictions
    double mlContextConfidence[6];    // Per-context ML confidence tracking
    uint32_t mlContextUsage[6];       // Per-context ML usage count
    double recentMLAccuracy;          // Recent ML prediction accuracy estimate
    Time lastMLPerformanceUpdate;     // Last time ML performance was evaluated
};

} // namespace ns3

#endif /* SMART_WIFI_MANAGER_RF_H */