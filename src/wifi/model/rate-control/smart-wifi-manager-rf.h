/*
 * Smart WiFi Manager - PHASE 1-4 COMPLETE (FIXED HEADER)
 *
 * This header includes ALL required fixes for the optimized pipeline:
 * - Per-station ML cache (FIX #1.3)
 * - Per-station call counter (FIX #1.2)
 * - Per-station model tracking (FIX #2.3)
 * - Hysteresis tracking (FIX #3.1)
 * - Unified fusion function (FIX #5)
 *
 * Author: ahmedjk34
 * Date: 2025-01-04
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
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/select.h>
#include <vector>

namespace ns3
{

/**
 * Enhanced WiFi context classification
 */
enum class WifiContextType
{
    EMERGENCY,
    POOR_UNSTABLE,
    MARGINAL,
    GOOD_UNSTABLE,
    GOOD_STABLE,
    EXCELLENT_STABLE,
    UNKNOWN
};

// Forward declarations
class SmartWifiManagerRfState;
class SmartWifiManagerRf;

/**
 * Safety assessment structure
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
 * Smart WiFi Manager - PHASE 1-4 COMPLETE
 */
class SmartWifiManagerRf : public WifiRemoteStationManager
{
  public:
    static TypeId GetTypeId();
    SmartWifiManagerRf();
    ~SmartWifiManagerRf() override;

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

    // Configuration methods
    void SetBenchmarkDistance(double distance);
    void SetModelName(const std::string& modelName);
    void SetOracleStrategy(const std::string& strategy);
    void SetCurrentInterferers(uint32_t interferers);
    void UpdateFromBenchmarkGlobals(double distance, uint32_t interferers);
    void DebugPrintCurrentConfig() const;
    void SetBenchmarkSpeed(double speed);

    double GetCurrentBenchmarkDistance() const
    {
        return m_benchmarkDistance.load();
    }

    uint32_t GetCurrentInterfererCount() const
    {
        return m_currentInterferers.load();
    }

    // Station registry
    SmartWifiManagerRfState* GetStationById(uint32_t stationId) const;
    uint32_t RegisterStation(SmartWifiManagerRfState* station);

  private:
    // Core interface
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

    // ML inference
    InferenceResult RunMLInference(const std::vector<double>& features,
                                   const std::string& modelName = "") const;
    std::vector<double> ExtractFeatures(WifiRemoteStation* station) const;
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr);

    // Feature extraction
    double GetSnrTrendShort(WifiRemoteStation* station) const;
    double GetSnrStabilityIndex(WifiRemoteStation* station) const;
    double GetSnrPredictionConfidence(WifiRemoteStation* station) const;
    double GetMobilityMetric(WifiRemoteStation* station) const;

    // SNR modeling
    double ConvertToRealisticSnr(double ns3Snr) const;

    // Context assessment
    SafetyAssessment AssessNetworkSafety(SmartWifiManagerRfState* station);
    WifiContextType ClassifyNetworkContext(SmartWifiManagerRfState* station) const;
    std::string ContextTypeToString(WifiContextType type) const;
    double CalculateRiskLevel(SmartWifiManagerRfState* station) const;
    uint32_t GetContextSafeRate(SmartWifiManagerRfState* station, WifiContextType context) const;

    // Rate decision
    uint32_t GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station,
                                      const SafetyAssessment& safety) const;
    uint32_t FuseMLAndRuleBased(uint32_t mlRate,
                                uint32_t ruleRate,
                                double mlConfidence,
                                const SafetyAssessment& safety,
                                SmartWifiManagerRfState* station) const;

    // 🚀 PHASE 1A: Enhanced features
    void UpdateEnhancedFeatures(SmartWifiManagerRfState* station);

    // 🚀 PHASE 2: Scenario-aware model selection
    std::string SelectBestModel(SmartWifiManagerRfState* station) const;

    // 🚀 PHASE 3: Hysteresis
    uint8_t ApplyHysteresis(SmartWifiManagerRfState* station,
                            uint8_t currentRate,
                            uint8_t predictedRate) const;

    // 🚀 PHASE 4: Adaptive fusion
    double CalculateAdaptiveTrust(double mlConfidence, SmartWifiManagerRfState* station) const;
    uint32_t AdaptiveFusion(uint8_t mlRate,
                            uint8_t ruleRate,
                            double mlConfidence,
                            SmartWifiManagerRfState* station) const;

    // 🚀 FIX #5: Unified adaptive fusion (replaces backwards logic)
    uint32_t UnifiedAdaptiveFusion(uint8_t mlRate,
                                   uint8_t ruleRate,
                                   double mlConfidence,
                                   SmartWifiManagerRfState* station,
                                   const SafetyAssessment& safety) const;

    // Adaptive confidence
    double CalculateAdaptiveConfidenceThreshold(SmartWifiManagerRfState* station,
                                                WifiContextType context) const;

    // Attribute accessors
    void SetBenchmarkDistanceAttribute(double dist);
    double GetBenchmarkDistanceAttribute() const;
    void SetInterferersAttribute(uint32_t count);
    uint32_t GetInterferersAttribute() const;
    double GetBenchmarkSpeedAttribute() const;
    uint32_t GetBenchmarkPacketSizeAttribute() const;
    void SetBenchmarkPacketSizeAttribute(uint32_t pktSize);

    // Configuration
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

    // 🚀 PHASE 2 & 3
    bool m_enableScenarioAwareSelection;
    mutable std::string m_currentModelName;
    uint32_t m_hysteresisStreak;
};

/**
 * 🚀 FIXED: SmartWifiManagerRfState with all per-station tracking
 */
struct SmartWifiManagerRfState : public WifiRemoteStation
{
    uint32_t stationId;

    // Core SNR metrics
    double lastSnr;
    double lastRawSnr;
    double snrFast;
    double snrSlow;
    double snrTrendShort;
    double snrStabilityIndex;
    double snrPredictionConfidence;
    double snrVariance;

    // Timing
    Time lastUpdateTime;
    Time lastInferenceTime;
    Time lastRateChangeTime;

    // Network state
    double mobilityMetric;
    Vector lastPosition;
    uint32_t currentRateIndex;
    uint32_t previousRateIndex;

    // Context
    WifiContextType lastContext;
    double lastRiskLevel;

    // Packet tracking
    uint32_t totalPackets;
    uint32_t lostPackets;
    uint32_t successfulPackets;
    uint32_t failedPackets;

    // SNR history
    std::deque<double> snrHistory;
    std::deque<double> rawSnrHistory;

    // ML tracking
    uint32_t mlInferencesReceived;
    uint32_t mlInferencesSuccessful;
    double avgMlConfidence;
    std::string preferredModel;
    uint32_t lastMLInfluencedRate;
    Time lastMLInfluenceTime;
    double mlPerformanceScore;
    uint32_t mlSuccessfulPredictions;
    double mlContextConfidence[6];
    uint32_t mlContextUsage[6];
    double recentMLAccuracy;
    Time lastMLPerformanceUpdate;

    // 🚀 PHASE 1B: Enhanced features
    double rssiVariance;
    double interferenceLevel;
    double distanceMetric;
    double avgPacketSize;
    double retryRate;
    double frameErrorRate;
    uint32_t packetsSinceRateChange;
    std::deque<uint32_t> recentRateHistory;

    // 🚀 PHASE 3: Hysteresis tracking
    uint32_t ratePredictionStreak;
    uint8_t lastPredictedRate;
    uint32_t rateStableCount;

    // Consecutive tracking
    uint32_t consecutiveFailures;
    uint32_t consecutiveSuccesses;

    // 🚀 FIX #1.3: Per-station ML cache
    struct MLCache
    {
        Time timestamp;
        uint32_t rateIdx;
        double confidence;
        std::string modelUsed;
        double snrAtInference;
        double distanceAtInference;
        uint32_t interferersAtInference;

        MLCache()
            : timestamp(Seconds(0)),
              rateIdx(3),
              confidence(0.0),
              modelUsed("none"),
              snrAtInference(0.0),
              distanceAtInference(0.0),
              interferersAtInference(0)
        {
        }
    };

    MLCache mlCache;
    bool mlCacheValid;

    // 🚀 FIX #1.2: Per-station call counter
    uint64_t callCounter;

    // 🚀 FIX #2.3: Per-station model tracking
    std::string currentModelName;
    Time lastModelSwitchTime;

    // 🚀 FIX #6: Per-station ML failure tracking
    uint32_t consecutiveMlFailures;
    uint32_t packetsSinceMLRetry;

    // Feature stability for cache invalidation
    double lastInferenceSnr;
    double lastInferenceDistance;

    static constexpr uint32_t WINDOW_SIZE = 50;

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
          successfulPackets(0),
          failedPackets(0),
          mlInferencesReceived(0),
          mlInferencesSuccessful(0),
          avgMlConfidence(0.3),
          preferredModel("oracle_aggressive"),
          lastMLInfluencedRate(3),
          lastMLInfluenceTime(Seconds(0)),
          mlPerformanceScore(0.5),
          mlSuccessfulPredictions(0),
          recentMLAccuracy(0.5),
          lastMLPerformanceUpdate(Seconds(0)),
          rssiVariance(0.0),
          interferenceLevel(0.0),
          distanceMetric(20.0),
          avgPacketSize(1200.0),
          retryRate(0.0),
          frameErrorRate(0.0),
          packetsSinceRateChange(0),
          ratePredictionStreak(0),
          lastPredictedRate(3),
          rateStableCount(0),
          consecutiveFailures(0),
          consecutiveSuccesses(0),
          mlCacheValid(false),
          callCounter(0),
          currentModelName("oracle_aggressive"),
          lastModelSwitchTime(Seconds(0)),
          consecutiveMlFailures(0),
          packetsSinceMLRetry(0),
          lastInferenceSnr(0.0),
          lastInferenceDistance(0.0)
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