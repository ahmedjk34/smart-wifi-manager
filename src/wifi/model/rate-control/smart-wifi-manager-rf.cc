/*
 * Smart WiFi Manager with 15 Safe Features - PHASE 1-4 COMPLETE
 * Compatible with ahmedjk34's enhanced pipeline (15 features, 75-80% accuracy)
 *
 * ============================================================================
 * 🚀 PHASE 1A: ENHANCED FEATURES (9 → 15 features)
 * ============================================================================
 * NEW FEATURES ADDED (6 more for +67% information):
 * 9.  retryRate          - Retry rate (past performance)
 * 10. frameErrorRate     - Error rate (PHY feedback)
 * 11. channelBusyRatio   - Channel occupancy (interference)
 * 12. recentRateAvg      - Recent rate average (temporal context)
 * 13. rateStability      - Rate stability (change frequency)
 * 14. sinceLastChange    - Time since last rate change (stability)
 *
 * ============================================================================
 * 🚀 PHASE 2: SCENARIO-AWARE MODEL SELECTION
 * ============================================================================
 * DYNAMIC MODEL SWITCHING based on network difficulty:
 * - Easy scenarios (SNR>25, few interferers) → oracle_aggressive
 * - Medium scenarios (SNR 13-25) → oracle_balanced
 * - Hard scenarios (SNR<13, high interference) → oracle_conservative
 *
 * Difficulty scoring factors:
 * - SNR quality (40% weight)
 * - Interference level (30% weight)
 * - Mobility (20% weight)
 * - Channel busy ratio (10% weight)
 *
 * ============================================================================
 * 🚀 PHASE 3: HYSTERESIS (RATE THRASHING FIX)
 * ============================================================================
 * PREVENTS EXCESSIVE RATE CHANGES:
 * - Requires 3 consecutive identical predictions before changing rate
 * - Tracks prediction streak per station
 * - Expected: 100+ rate changes → 30-50 per test (67% reduction!)
 *
 * ============================================================================
 * 🚀 PHASE 4: ADAPTIVE ML FUSION
 * ============================================================================
 * DYNAMIC TRUST CALCULATION:
 * - Base trust = ML confidence
 * - +20% if SNR is stable (snrStabilityIndex > 0.8)
 * - -20% if channel is busy (channelBusyRatio > 0.7)
 * - -30% if mobile (mobilityMetric > 10.0)
 * - Variable ML vs rule-based weights (not fixed 70/30)
 *
 * ============================================================================
 * CRITICAL UPDATES (2025-10-02 20:09:48 UTC):
 * ============================================================================
 * WHAT WE CHANGED:
 * 1. Feature extraction: ExtractFeatures() now returns 15 features (not 9)
 * 2. Added 6 new safe features (retryRate, frameErrorRate, channelBusyRatio, etc.)
 * 3. Dynamic model selection based on scenario difficulty (Phase 2)
 * 4. Hysteresis to prevent rate thrashing (Phase 3)
 * 5. Adaptive ML fusion with variable trust (Phase 4)
 * 6. Python client integration via socket to localhost:8765
 * 7. Model paths: python_files/trained_models/
 * 8. Default oracle: oracle_aggressive (75-80% test accuracy expected)
 *
 * WHY WE CHANGED IT:
 * - Phase 1A: 15 features → 75-80% accuracy (up from 62.8%)
 * - Phase 2: Scenario-aware selection → +15-20% adaptability
 * - Phase 3: Hysteresis → 67% fewer rate changes (stability)
 * - Phase 4: Adaptive fusion → Better edge case handling
 *
 * EXPECTED IMPROVEMENTS:
 * - Model accuracy: 62.8% → 75-80%
 * - Rate changes: 100+ → 30-50 per test
 * - Throughput (clean): +10-15%
 * - Stability: +50% (fewer PHY reconfigurations)
 *
 * FEATURE ORDER (CRITICAL - MUST MATCH TRAINING):
 * 0.  lastSnr (dB)               - Most recent realistic SNR
 * 1.  snrFast (dB)               - Fast-moving average (α=0.1)
 * 2.  snrSlow (dB)               - Slow-moving average (α=0.01)
 * 3.  snrTrendShort              - Short-term SNR trend
 * 4.  snrStabilityIndex          - SNR stability (0-10)
 * 5.  snrPredictionConfidence    - Prediction confidence (0-1)
 * 6.  snrVariance                - SNR variance (0-100)
 * 7.  channelWidth (MHz)         - Channel bandwidth
 * 8.  mobilityMetric             - Node mobility (0-50)
 * 9.  retryRate                  - 🚀 NEW! Retry ratio (0-1)
 * 10. frameErrorRate             - 🚀 NEW! Error ratio (0-1)
 * 11. channelBusyRatio           - 🚀 NEW! Channel busy (0-1)
 * 12. recentRateAvg              - 🚀 NEW! Recent rate avg (0-7)
 * 13. rateStability              - 🚀 NEW! Rate stability (0-1)
 * 14. sinceLastChange            - 🚀 NEW! Time since change (0-1)
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-10-02 20:09:48 UTC
 * Version: 7.0 (PHASE 1-4 COMPLETE - 15 Features, Scenario-Aware, Hysteresis, Adaptive Fusion)
 */

#include "smart-wifi-manager-rf.h"

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SmartWifiManagerRf");
NS_OBJECT_ENSURE_REGISTERED(SmartWifiManagerRf);

// ============================================================================
// Global realistic SNR conversion function (matches benchmark)
// ============================================================================
enum SnrModel
{
    LOG_MODEL,
    SOFT_MODEL,
    INTF_MODEL
};

double
ConvertNS3ToRealisticSnr(double ns3Value, double distance, uint32_t interferers, SnrModel model)
{
    if (distance <= 0.0)
        distance = 1.0;
    if (distance > 200.0)
        distance = 200.0;
    if (interferers > 10)
        interferers = 10;

    double realisticSnr = 0.0;

    switch (model)
    {
    case LOG_MODEL: {
        double snr0 = 40.0;
        double pathLossExp = 2.2;
        realisticSnr = snr0 - 10 * pathLossExp * log10(distance);
        realisticSnr -= (interferers * 1.5);
        break;
    }

    case SOFT_MODEL: {
        if (distance <= 20.0)
            realisticSnr = 35.0 - (distance * 0.8);
        else if (distance <= 50.0)
            realisticSnr = 19.0 - ((distance - 20.0) * 0.5);
        else if (distance <= 100.0)
            realisticSnr = 4.0 - ((distance - 50.0) * 0.3);
        else
            realisticSnr = -11.0 - ((distance - 100.0) * 0.2);

        realisticSnr -= (interferers * 2.0);
        break;
    }

    case INTF_MODEL: {
        realisticSnr = 38.0 - 10 * log10(distance * distance);
        realisticSnr -= (pow(interferers, 1.2) * 1.2);
        break;
    }
    }

    double variation = fmod(std::abs(ns3Value), 12.0) - 6.0;
    realisticSnr += variation * 0.4;

    realisticSnr = std::max(-30.0, std::min(45.0, realisticSnr));
    return realisticSnr;
}

// ============================================================================
// TypeId and Constructor
// ============================================================================
TypeId
SmartWifiManagerRf::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SmartWifiManagerRf")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<SmartWifiManagerRf>()
            // Model paths point to python_files/trained_models/
            .AddAttribute(
                "ModelPath",
                "Path to the Random Forest model file (.joblib)",
                StringValue("python_files/trained_models/step4_rf_oracle_aggressive_FIXED.joblib"),
                MakeStringAccessor(&SmartWifiManagerRf::m_modelPath),
                MakeStringChecker())
            .AddAttribute(
                "ScalerPath",
                "Path to the scaler file (.joblib)",
                StringValue(
                    "python_files/trained_models/step4_scaler_oracle_aggressive_FIXED.joblib"),
                MakeStringAccessor(&SmartWifiManagerRf::m_scalerPath),
                MakeStringChecker())
            // Default model is oracle_aggressive (75-80% accuracy expected)
            .AddAttribute("ModelName",
                          "Specific model name for inference server",
                          StringValue("oracle_aggressive"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelName),
                          MakeStringChecker())
            .AddAttribute("ModelType",
                          "Model type (oracle/v3)",
                          StringValue("oracle"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelType),
                          MakeStringChecker())
            .AddAttribute(
                "OracleStrategy",
                "Oracle strategy (oracle_aggressive, oracle_balanced, oracle_conservative)",
                StringValue("oracle_aggressive"),
                MakeStringAccessor(&SmartWifiManagerRf::m_oracleStrategy),
                MakeStringChecker())
            .AddAttribute("InferenceServerPort",
                          "Port number of ML inference server (your Python server)",
                          UintegerValue(8765),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferenceServerPort),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("ConfidenceThreshold",
                          "Base ML confidence threshold (adaptive)",
                          DoubleValue(0.20),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_confidenceThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("MLGuidanceWeight",
                          "Weight of ML guidance in fusion (0.0-1.0)",
                          DoubleValue(0.70),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_mlGuidanceWeight),
                          MakeDoubleChecker<double>())
            .AddAttribute("InferencePeriod",
                          "Period between ML inferences (packets)",
                          UintegerValue(25),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferencePeriod),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EnableAdaptiveWeighting",
                          "Enable adaptive ML weighting based on performance",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableAdaptiveWeighting),
                          MakeBooleanChecker())
            .AddAttribute("MLCacheTime",
                          "ML result cache time in milliseconds",
                          UintegerValue(200),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_mlCacheTime),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("WindowSize",
                          "Success ratio window size (packets)",
                          UintegerValue(50),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_windowSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SnrAlpha",
                          "SNR exponential smoothing alpha (0.0-1.0)",
                          DoubleValue(0.1),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_snrAlpha),
                          MakeDoubleChecker<double>())
            .AddAttribute("FallbackRate",
                          "Fallback rate index on ML failure (0-7)",
                          UintegerValue(3),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_fallbackRate),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RiskThreshold",
                          "Risk threshold for emergency actions (0.0-1.0)",
                          DoubleValue(0.7),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_riskThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("FailureThreshold",
                          "Consecutive failures threshold for emergency backoff",
                          UintegerValue(5),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_failureThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("UseRealisticSnr",
                          "Use realistic SNR calculation (-30dB to +45dB)",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_useRealisticSnr),
                          MakeBooleanChecker())
            .AddAttribute("SnrOffset",
                          "SNR offset for calibration (dB)",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_snrOffset),
                          MakeDoubleChecker<double>())
            // 🚀 PHASE 3: Hysteresis configuration
            .AddAttribute("HysteresisStreak",
                          "Number of consecutive predictions required before rate change (Phase 3)",
                          UintegerValue(3),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_hysteresisStreak),
                          MakeUintegerChecker<uint32_t>())
            // 🚀 PHASE 2: Scenario-aware model selection
            .AddAttribute("EnableScenarioAwareSelection",
                          "Enable dynamic model selection based on scenario difficulty (Phase 2)",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableScenarioAwareSelection),
                          MakeBooleanChecker())
            .AddTraceSource("Rate",
                            "Remote station data rate changed",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_currentRate),
                            "ns3::TracedValueCallback::Uint64")
            .AddTraceSource("MLInferences",
                            "Number of ML inferences made",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_mlInferences),
                            "ns3::TracedValueCallback::Uint32")
            .AddAttribute("BenchmarkDistance",
                          "Initial benchmark distance (meters) - set via attributes",
                          DoubleValue(20.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::SetBenchmarkDistanceAttribute,
                                             &SmartWifiManagerRf::GetBenchmarkDistanceAttribute),
                          MakeDoubleChecker<double>())
            .AddAttribute("BenchmarkInterferers",
                          "Initial interferer count - set via attributes",
                          UintegerValue(0),
                          MakeUintegerAccessor(&SmartWifiManagerRf::SetInterferersAttribute,
                                               &SmartWifiManagerRf::GetInterferersAttribute),
                          MakeUintegerChecker<uint32_t>());
    return tid;
}

SmartWifiManagerRf::SmartWifiManagerRf()
    : m_currentRate(0),
      m_mlInferences(0),
      m_benchmarkDistance(20.0),
      m_currentInterferers(0),
      m_mlFailures(0),
      m_mlCacheHits(0),
      m_avgMlLatency(0.0),
      m_lastMlRate(3),
      m_lastMlTime(Seconds(0)),
      m_lastMlConfidence(0.0),
      m_snrAlpha(0.1),
      m_modelType("oracle"),
      m_inferenceServerPort(8765),
      m_modelPath("python_files/trained_models/step4_rf_oracle_aggressive_FIXED.joblib"),
      m_scalerPath("python_files/trained_models/step4_scaler_oracle_aggressive_FIXED.joblib"),
      m_modelName("oracle_aggressive"),
      m_oracleStrategy("oracle_aggressive"),
      m_confidenceThreshold(0.20),
      m_riskThreshold(0.7),
      m_failureThreshold(5),
      m_mlGuidanceWeight(0.70),
      m_mlCacheTime(200),
      m_enableAdaptiveWeighting(true),
      m_inferencePeriod(25),
      m_fallbackRate(3),
      m_windowSize(50),
      m_useRealisticSnr(true),
      m_maxSnrDb(45.0),
      m_minSnrDb(-30.0),
      m_snrOffset(0.0),
      m_enableDetailedLogging(true),
      m_nextStationId(1),
      m_hysteresisStreak(3),                  // 🚀 PHASE 3
      m_enableScenarioAwareSelection(true),   // 🚀 PHASE 2
      m_currentModelName("oracle_aggressive") // 🚀 PHASE 2
{
    NS_LOG_FUNCTION(this);
    std::cout << "============================================================================"
              << std::endl;
    std::cout << "🚀 SmartWifiManagerRf v8.0 - PHASE 1B COMPLETE" << std::endl;
    std::cout << "============================================================================"
              << std::endl;
    std::cout << "✅ PHASE 1B: 14 features (7 SNR + 1 network + 2 Phase 1A + 4 Phase 1B)"
              << std::endl;
    std::cout << "✅ PHASE 2: Scenario-aware model selection (dynamic switching)" << std::endl;
    std::cout << "✅ PHASE 3: Hysteresis (3-streak confirmation, rate thrashing fix)" << std::endl;
    std::cout << "✅ PHASE 4: Adaptive ML fusion (dynamic trust calculation)" << std::endl;
    std::cout << "============================================================================"
              << std::endl;
    std::cout << "📊 Expected Improvements:" << std::endl;
    std::cout << "   - Accuracy: 62.8% → 68-75% (Phase 1B)" << std::endl;
    std::cout << "   - Rate changes: 100+ → 30-50 per test (Phase 3)" << std::endl;
    std::cout << "   - Adaptability: +15-20% (Phase 2)" << std::endl;
    std::cout << "   - Stability: +50% fewer PHY reconfigurations (Phase 3)" << std::endl;
    std::cout << "============================================================================"
              << std::endl;
    std::cout << "🔗 Python Server: localhost:" << m_inferenceServerPort << std::endl;
    std::cout << "🎯 Default Model: " << m_oracleStrategy << " (will switch dynamically)"
              << std::endl;
    std::cout << "============================================================================"
              << std::endl;
}

SmartWifiManagerRf::~SmartWifiManagerRf()
{
    NS_LOG_FUNCTION(this);
    std::lock_guard<std::mutex> lock(m_stationRegistryMutex);
    m_stationRegistry.clear();
}

void
SmartWifiManagerRf::DoInitialize()
{
    NS_LOG_FUNCTION(this);

    if (GetHtSupported() || GetVhtSupported() || GetHeSupported())
    {
        NS_FATAL_ERROR("SmartWifiManagerRf does not support HT/VHT/HE modes");
    }

    std::cout << "[INIT] Model: " << m_modelName << " | Strategy: " << m_oracleStrategy
              << std::endl;
    std::cout << "[INIT] Features: 14 (7 SNR + 1 network + 2 Phase 1A + 4 Phase 1B)" << std::endl;
    std::cout << "[INIT] Python Server: localhost:" << m_inferenceServerPort << std::endl;
    std::cout << "[INIT] Hysteresis: " << m_hysteresisStreak << "-streak confirmation" << std::endl;
    std::cout << "[INIT] Scenario-Aware: "
              << (m_enableScenarioAwareSelection ? "ENABLED" : "DISABLED") << std::endl;

    WifiRemoteStationManager::DoInitialize();
}

// ============================================================================
// Station registry for safe access
// ============================================================================
uint32_t
SmartWifiManagerRf::RegisterStation(SmartWifiManagerRfState* station)
{
    std::lock_guard<std::mutex> lock(m_stationRegistryMutex);
    uint32_t id = m_nextStationId.fetch_add(1);
    m_stationRegistry[id] = station;
    station->stationId = id;

    // 🚀 PHASE 3: Initialize hysteresis tracking
    station->ratePredictionStreak = 0;
    station->lastPredictedRate = 3; // Default middle rate
    station->rateStableCount = 0;

    return id;
}

SmartWifiManagerRfState*
SmartWifiManagerRf::GetStationById(uint32_t stationId) const
{
    std::lock_guard<std::mutex> lock(m_stationRegistryMutex);
    auto it = m_stationRegistry.find(stationId);
    return (it != m_stationRegistry.end()) ? it->second : nullptr;
}

WifiRemoteStation*
SmartWifiManagerRf::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    SmartWifiManagerRfState* station = new SmartWifiManagerRfState;

    double currentDistance = m_benchmarkDistance.load();
    uint32_t currentInterferers = m_currentInterferers.load();

    double initialSnr =
        ConvertNS3ToRealisticSnr(100.0, currentDistance, currentInterferers, SOFT_MODEL);

    station->lastSnr = initialSnr;
    station->lastRawSnr = 0.0;
    station->snrFast = initialSnr;
    station->snrSlow = initialSnr;
    station->snrTrendShort = 0.0;
    station->snrStabilityIndex = 1.0;
    station->snrPredictionConfidence = 0.8;
    station->snrVariance = 0.1;

    // 🚀 PHASE 1A: Initialize new feature tracking
    station->retryRate = 0.0;
    station->frameErrorRate = 0.0;
    // station->channelBusyRatio = 0.0;
    // station->recentRateAvg = 4.0; // Middle rate
    // station->rateStability = 1.0;
    // station->sinceLastChange = 0.0;
    // station->packetsSinceRateChange = 0;

    // 🚀 PHASE 1B: Initialize new features (4 features)
    station->rssiVariance = 0.1;
    station->interferenceLevel = 0.0;
    station->distanceMetric = m_benchmarkDistance.load();
    station->avgPacketSize = 1200.0; // Default MTU

    // 🚀 PHASE 3: Initialize hysteresis tracking
    station->ratePredictionStreak = 0;
    station->lastPredictedRate = 3;
    station->rateStableCount = 0;

    const_cast<SmartWifiManagerRf*>(this)->RegisterStation(station);

    std::cout << "[STATION CREATED] ID=" << station->stationId << " | Initial SNR=" << initialSnr
              << "dB | Distance=" << currentDistance << "m | Interferers=" << currentInterferers
              << std::endl;

    return station;
}

// ============================================================================
// Configuration methods
// ============================================================================
void
SmartWifiManagerRf::SetBenchmarkDistance(double distance)
{
    if (distance <= 0.0 || distance > 200.0)
        return;
    m_benchmarkDistance.store(distance);
    NS_LOG_INFO("[CONFIG] Distance updated to " << distance << "m");
}

void
SmartWifiManagerRf::SetModelName(const std::string& modelName)
{
    m_modelName = modelName;
    m_currentModelName = modelName;
}

void
SmartWifiManagerRf::SetOracleStrategy(const std::string& strategy)
{
    m_oracleStrategy = strategy;
    m_modelName = strategy;
    m_currentModelName = strategy;
}

void
SmartWifiManagerRf::SetCurrentInterferers(uint32_t interferers)
{
    m_currentInterferers.store(interferers);
}

void
SmartWifiManagerRf::UpdateFromBenchmarkGlobals(double distance, uint32_t interferers)
{
    m_benchmarkDistance.store(distance);
    m_currentInterferers.store(interferers);
    NS_LOG_INFO("[SYNC] Updated distance=" << distance << "m, interferers=" << interferers);
}

double
SmartWifiManagerRf::ConvertToRealisticSnr(double ns3Snr) const
{
    return ConvertNS3ToRealisticSnr(ns3Snr,
                                    m_benchmarkDistance.load(),
                                    m_currentInterferers.load(),
                                    SOFT_MODEL);
}

// ============================================================================
// 🚀 PHASE 2: SCENARIO-AWARE MODEL SELECTION (PHASE 1B ENHANCED)
// ============================================================================
std::string
SmartWifiManagerRf::SelectBestModel(SmartWifiManagerRfState* station) const
{
    NS_LOG_FUNCTION(this << station);

    if (!m_enableScenarioAwareSelection)
    {
        return m_oracleStrategy;
    }

    double currentDistance = m_benchmarkDistance.load();
    uint32_t currentInterferers = m_currentInterferers.load();

    // CRITICAL FIX: Force SNR recalculation if parameters don't match initialized values
    bool snrIsInitValue = (station->snrSlow > 18.0 && station->snrSlow < 18.5);
    bool paramsChanged = (currentDistance != 20.0 || currentInterferers != 0);
    bool snrTooHighForDistance = (currentDistance > 50.0 && station->snrSlow > 10.0);

    if ((snrIsInitValue && paramsChanged) || snrTooHighForDistance)
    {
        double realisticSnr =
            ConvertNS3ToRealisticSnr(100.0, currentDistance, currentInterferers, SOFT_MODEL);
        station->lastSnr = realisticSnr;
        station->snrFast = realisticSnr;
        station->snrSlow = realisticSnr;

        NS_LOG_INFO("[PHASE 2] Corrected stale SNR: " << station->snrSlow << " → " << realisticSnr
                                                      << " dB");
    }

    // ========================================================================
    // 🚀 PHASE 1B: ENHANCED DIFFICULTY CALCULATION (5 FACTORS)
    // ========================================================================
    double difficultyScore = 0.0;
    double avgSnr = station->snrSlow;

    // Factor 1: SNR quality (40% weight) - PRIMARY INDICATOR
    // SNR < 5 dB is very hard, SNR > 30 dB is easy
    // Use exponential decay for better sensitivity at low SNR
    double snrScore;
    if (avgSnr < 5.0)
    {
        snrScore = 1.0; // Very hard
    }
    else if (avgSnr > 30.0)
    {
        snrScore = 0.0; // Very easy
    }
    else
    {
        // Exponential mapping: harder at low SNR, easier at high SNR
        snrScore = std::exp(-(avgSnr - 5.0) / 10.0);
    }
    difficultyScore += snrScore * 0.40;

    // Factor 2: Interference level (25% weight) - CRITICAL FOR POOR CONDITIONS
    // 0-5 interferers expected, exponential impact
    double intfScore = std::min(1.0, std::pow(static_cast<double>(currentInterferers) / 5.0, 0.8));
    difficultyScore += intfScore * 0.25;

    // Factor 3: 🚀 PHASE 1B - RSSI Variance (15% weight) - SIGNAL STABILITY
    // High variance = unstable channel = harder
    double rssiVarianceScore = std::min(1.0, station->rssiVariance / 10.0); // Normalize 0-10 dB²
    difficultyScore += rssiVarianceScore * 0.15;

    // Factor 4: 🚀 PHASE 1B - Interference Level from feature (10% weight)
    // Direct measure of collision rate
    double measuredIntfScore = station->interferenceLevel; // Already 0-1
    difficultyScore += measuredIntfScore * 0.10;

    // Factor 5: Mobility (10% weight) - REDUCED IMPACT
    // High mobility = harder to predict
    double mobilityScore = std::min(1.0, station->mobilityMetric / 20.0);
    difficultyScore += mobilityScore * 0.10;

    // ========================================================================
    // ADAPTIVE THRESHOLDS BASED ON PHASE 1B FEATURES
    // ========================================================================

    // Base thresholds
    double aggressiveThreshold = 0.30;   // Below this: use aggressive
    double conservativeThreshold = 0.65; // Above this: use conservative

    // 🚀 PHASE 1B: Adjust thresholds based on distance
    // Farther distance = use conservative earlier
    if (currentDistance > 70.0)
    {
        conservativeThreshold = 0.55; // Earlier switch to conservative
    }
    else if (currentDistance < 30.0)
    {
        aggressiveThreshold = 0.35; // Can stay aggressive longer
    }

    // 🚀 PHASE 1B: Adjust based on RSSI variance (signal stability)
    // High variance = need more conservative
    if (station->rssiVariance > 5.0)
    {
        conservativeThreshold = 0.60; // Earlier switch
        aggressiveThreshold = 0.25;   // Harder to stay aggressive
    }

    // Select model based on difficulty with ADAPTIVE thresholds
    std::string selectedModel;

    if (difficultyScore < aggressiveThreshold)
    {
        selectedModel = "oracle_aggressive";
        NS_LOG_INFO("[PHASE 2] EASY (score="
                    << difficultyScore << "/" << aggressiveThreshold << "): SNR=" << avgSnr << "dB"
                    << ", intf=" << currentInterferers << ", rssiVar=" << station->rssiVariance
                    << ", dist=" << currentDistance << "m"
                    << " → oracle_aggressive");
    }
    else if (difficultyScore < conservativeThreshold)
    {
        selectedModel = "oracle_balanced";
        NS_LOG_INFO("[PHASE 2] MEDIUM (score=" << difficultyScore << "/" << conservativeThreshold
                                               << "): SNR=" << avgSnr << "dB"
                                               << ", intf=" << currentInterferers
                                               << ", rssiVar=" << station->rssiVariance
                                               << ", dist=" << currentDistance << "m"
                                               << " → oracle_balanced");
    }
    else
    {
        selectedModel = "oracle_conservative";
        NS_LOG_INFO("[PHASE 2] HARD (score=" << difficultyScore << " >" << conservativeThreshold
                                             << "): SNR=" << avgSnr << "dB"
                                             << ", intf=" << currentInterferers
                                             << ", rssiVar=" << station->rssiVariance
                                             << ", dist=" << currentDistance << "m"
                                             << " → oracle_conservative");
    }

    // 🚀 PHASE 1B: Emergency override based on distance + interference combo
    // If distance > 80m AND interferers > 3, FORCE conservative
    if (currentDistance > 80.0 && currentInterferers > 3)
    {
        if (selectedModel != "oracle_conservative")
        {
            NS_LOG_WARN("[PHASE 2] EMERGENCY: Forcing conservative (dist="
                        << currentDistance << "m, intf=" << currentInterferers << ")");
            selectedModel = "oracle_conservative";
        }
    }

    return selectedModel;
}

// ============================================================================
// 🚀 PHASE 3: HYSTERESIS (PHASE 1B ENHANCED - RATE THRASHING FIX)
// ============================================================================
uint8_t
SmartWifiManagerRf::ApplyHysteresis(SmartWifiManagerRfState* station,
                                    uint8_t currentRate,
                                    uint8_t predictedRate) const
{
    NS_LOG_FUNCTION(this << station << (uint32_t)currentRate << (uint32_t)predictedRate);

    // Don't change rate if prediction is same
    if (predictedRate == currentRate)
    {
        station->rateStableCount++;
        return currentRate;
    }

    // Check if this is the same prediction as last time
    if (predictedRate == station->lastPredictedRate)
    {
        // Same prediction - increment streak
        station->ratePredictionStreak++;
    }
    else
    {
        // Different prediction - reset streak
        station->ratePredictionStreak = 1;
        station->lastPredictedRate = predictedRate;
    }

    // ========================================================================
    // 🚀 PHASE 1B: ADAPTIVE HYSTERESIS THRESHOLD
    // ========================================================================

    uint32_t requiredStreak = m_hysteresisStreak; // Default: 3

    // Adjust streak based on RSSI variance (signal stability)
    // High variance = require MORE confirmation (prevent thrashing in unstable conditions)
    if (station->rssiVariance > 8.0)
    {
        requiredStreak = m_hysteresisStreak + 2; // Need 5 confirmations
        NS_LOG_DEBUG("[PHASE 3] High RSSI variance ("
                     << station->rssiVariance << "), increased streak to " << requiredStreak);
    }
    else if (station->rssiVariance > 5.0)
    {
        requiredStreak = m_hysteresisStreak + 1; // Need 4 confirmations
        NS_LOG_DEBUG("[PHASE 3] Moderate RSSI variance ("
                     << station->rssiVariance << "), increased streak to " << requiredStreak);
    }

    // Adjust streak based on interference level
    // High interference = require MORE confirmation (prevent bad rate changes)
    if (station->interferenceLevel > 0.7)
    {
        requiredStreak = std::max(requiredStreak, m_hysteresisStreak + 2); // At least 5
        NS_LOG_DEBUG("[PHASE 3] High interference ("
                     << station->interferenceLevel << "), increased streak to " << requiredStreak);
    }

    // ========================================================================
    // 🚀 PHASE 1B: EMERGENCY BYPASS
    // ========================================================================

    // If SNR is critically low (<5 dB) AND prediction is to decrease rate,
    // bypass hysteresis (immediate protection)
    bool emergencyDowngrade = (station->lastSnr < 5.0) && (predictedRate < currentRate);

    // If SNR is excellent (>28 dB) AND stable (rssiVar < 2.0) AND prediction is to increase,
    // reduce required streak (faster adaptation in good conditions)
    bool fastUpgrade =
        (station->lastSnr > 28.0) && (station->rssiVariance < 2.0) && (predictedRate > currentRate);

    if (emergencyDowngrade)
    {
        NS_LOG_WARN("[PHASE 3] EMERGENCY BYPASS: SNR="
                    << station->lastSnr << "dB (critical) - immediate downgrade "
                    << (uint32_t)currentRate << " → " << (uint32_t)predictedRate);

        station->ratePredictionStreak = 0;
        station->rateStableCount = 0;
        station->lastPredictedRate = predictedRate;

        return predictedRate;
    }

    if (fastUpgrade && requiredStreak > 2)
    {
        requiredStreak = 2; // Only need 2 confirmations in perfect conditions
        NS_LOG_DEBUG("[PHASE 3] FAST UPGRADE: SNR="
                     << station->lastSnr << "dB (excellent), rssiVar=" << station->rssiVariance
                     << " - reduced streak to " << requiredStreak);
    }

    // ========================================================================
    // STANDARD HYSTERESIS CHECK
    // ========================================================================

    // Require adaptive streak consecutive predictions before changing
    if (station->ratePredictionStreak >= requiredStreak)
    {
        NS_LOG_INFO("[PHASE 3] Rate change CONFIRMED after "
                    << station->ratePredictionStreak << "/" << requiredStreak
                    << " consecutive predictions: " << (uint32_t)currentRate << " → "
                    << (uint32_t)predictedRate << " (rssiVar=" << station->rssiVariance
                    << ", intf=" << station->interferenceLevel << ")");

        // Reset counters
        station->ratePredictionStreak = 0;
        station->rateStableCount = 0;
        station->lastPredictedRate = predictedRate;

        return predictedRate;
    }

    // Not enough confirmation, keep current rate
    NS_LOG_DEBUG("[PHASE 3] Rate change SUPPRESSED (streak="
                 << station->ratePredictionStreak << "/" << requiredStreak << "): staying at "
                 << (uint32_t)currentRate << " (need "
                 << (requiredStreak - station->ratePredictionStreak) << " more confirmations)");

    return currentRate;
}

// ============================================================================
// 🚀 PHASE 4: ADAPTIVE ML FUSION (PHASE 1B ENHANCED)
// ============================================================================
double
SmartWifiManagerRf::CalculateAdaptiveTrust(double mlConfidence,
                                           SmartWifiManagerRfState* station) const
{
    NS_LOG_FUNCTION(this << mlConfidence << station);

    // Start with base ML confidence
    double mlTrust = mlConfidence;

    // ========================================================================
    // 🚀 PHASE 1B: ENHANCED TRUST CALCULATION (6 FACTORS)
    // ========================================================================

    // Factor 1: SNR stability (+25% if very stable, -15% if unstable)
    // snrStabilityIndex range: 0-10 (higher = more stable)
    if (station->snrStabilityIndex > 8.0)
    {
        mlTrust *= 1.25; // Very stable → trust ML more
        NS_LOG_DEBUG("[PHASE 4] SNR very stable (" << station->snrStabilityIndex
                                                   << ") → +25% trust");
    }
    else if (station->snrStabilityIndex > 6.0)
    {
        mlTrust *= 1.10; // Stable → slight boost
        NS_LOG_DEBUG("[PHASE 4] SNR stable (" << station->snrStabilityIndex << ") → +10% trust");
    }
    else if (station->snrStabilityIndex < 3.0)
    {
        mlTrust *= 0.85; // Unstable → reduce trust
        NS_LOG_DEBUG("[PHASE 4] SNR unstable (" << station->snrStabilityIndex << ") → -15% trust");
    }

    // Factor 2: 🚀 PHASE 1B - RSSI Variance (-30% if high variance)
    // High variance = unreliable signal = lower ML trust
    if (station->rssiVariance > 10.0)
    {
        mlTrust *= 0.70; // Very unstable signal
        NS_LOG_DEBUG("[PHASE 4] High RSSI variance (" << station->rssiVariance
                                                      << " dB²) → -30% trust");
    }
    else if (station->rssiVariance > 5.0)
    {
        mlTrust *= 0.85; // Moderate variance
        NS_LOG_DEBUG("[PHASE 4] Moderate RSSI variance (" << station->rssiVariance
                                                          << " dB²) → -15% trust");
    }
    else if (station->rssiVariance < 2.0)
    {
        mlTrust *= 1.10; // Very stable signal → boost trust
        NS_LOG_DEBUG("[PHASE 4] Low RSSI variance (" << station->rssiVariance
                                                     << " dB²) → +10% trust");
    }

    // Factor 3: 🚀 PHASE 1B - Interference Level (-35% if high interference)
    // High interference = unpredictable outcomes = lower ML trust
    if (station->interferenceLevel > 0.8)
    {
        mlTrust *= 0.65; // Severe interference
        NS_LOG_DEBUG("[PHASE 4] Severe interference (" << station->interferenceLevel
                                                       << ") → -35% trust");
    }
    else if (station->interferenceLevel > 0.5)
    {
        mlTrust *= 0.80; // Moderate interference
        NS_LOG_DEBUG("[PHASE 4] Moderate interference (" << station->interferenceLevel
                                                         << ") → -20% trust");
    }
    else if (station->interferenceLevel < 0.2)
    {
        mlTrust *= 1.15; // Low interference → boost trust
        NS_LOG_DEBUG("[PHASE 4] Low interference (" << station->interferenceLevel
                                                    << ") → +15% trust");
    }

    // Factor 4: Mobility (-25% if very mobile, +5% if stationary)
    // High mobility = harder to predict = lower ML trust
    if (station->mobilityMetric > 15.0)
    {
        mlTrust *= 0.75; // Very mobile
        NS_LOG_DEBUG("[PHASE 4] Very mobile (" << station->mobilityMetric << " m/s) → -25% trust");
    }
    else if (station->mobilityMetric > 5.0)
    {
        mlTrust *= 0.90; // Mobile
        NS_LOG_DEBUG("[PHASE 4] Mobile (" << station->mobilityMetric << " m/s) → -10% trust");
    }
    else if (station->mobilityMetric < 1.0)
    {
        mlTrust *= 1.05; // Stationary → slight boost
        NS_LOG_DEBUG("[PHASE 4] Stationary (" << station->mobilityMetric << " m/s) → +5% trust");
    }

    // Factor 5: 🚀 PHASE 1B - Distance-based adjustment
    // Far distance = harder conditions = adjust trust based on SNR
    double currentDistance = m_benchmarkDistance.load();
    if (currentDistance > 80.0)
    {
        // Far distance: only trust ML if SNR is still reasonable
        if (station->lastSnr < 10.0)
        {
            mlTrust *= 0.80; // Low SNR at far distance → reduce trust
            NS_LOG_DEBUG("[PHASE 4] Far distance (" << currentDistance << "m) + low SNR ("
                                                    << station->lastSnr << " dB) → -20% trust");
        }
    }
    else if (currentDistance < 30.0 && station->lastSnr > 25.0)
    {
        // Close distance + excellent SNR = ideal conditions → boost trust
        mlTrust *= 1.10;
        NS_LOG_DEBUG("[PHASE 4] Close distance (" << currentDistance << "m) + excellent SNR ("
                                                  << station->lastSnr << " dB) → +10% trust");
    }

    // Factor 6: Recent ML accuracy (if we have history)
    // recentMLAccuracy range: 0.0-1.0 (EWMA of past ML predictions)
    if (station->mlInferencesSuccessful > 20) // Only after enough samples
    {
        if (station->recentMLAccuracy > 0.75)
        {
            mlTrust *= 1.10; // ML has been accurate → trust more
            NS_LOG_DEBUG("[PHASE 4] High ML accuracy (" << station->recentMLAccuracy
                                                        << ") → +10% trust");
        }
        else if (station->recentMLAccuracy < 0.50)
        {
            mlTrust *= 0.85; // ML has been inaccurate → trust less
            NS_LOG_DEBUG("[PHASE 4] Low ML accuracy (" << station->recentMLAccuracy
                                                       << ") → -15% trust");
        }
    }

    // ========================================================================
    // FINAL CLAMPING AND REPORTING
    // ========================================================================

    // Clamp trust to [0.0, 1.0]
    mlTrust = std::min(1.0, std::max(0.0, mlTrust));

    NS_LOG_INFO("[PHASE 4] Adaptive trust: "
                << mlConfidence << " → " << mlTrust << " (SNRstab=" << station->snrStabilityIndex
                << ", rssiVar=" << station->rssiVariance << ", intf=" << station->interferenceLevel
                << ", mob=" << station->mobilityMetric << ", dist=" << m_benchmarkDistance.load()
                << "m)");

    return mlTrust;
}

uint32_t
SmartWifiManagerRf::AdaptiveFusion(uint8_t mlRate,
                                   uint8_t ruleRate,
                                   double mlConfidence,
                                   SmartWifiManagerRfState* station) const
{
    NS_LOG_FUNCTION(this << (uint32_t)mlRate << (uint32_t)ruleRate << mlConfidence << station);

    // Calculate adaptive trust (Phase 4)
    double mlTrust = CalculateAdaptiveTrust(mlConfidence, station);

    // Variable fusion weights based on trust
    double mlWeight = mlTrust;
    double ruleWeight = 1.0 - mlTrust;

    // Weighted average
    double fusedRate = (mlRate * mlWeight) + (ruleRate * ruleWeight);
    uint32_t finalRate = static_cast<uint32_t>(std::round(fusedRate));

    NS_LOG_INFO("[PHASE 4] Adaptive fusion: ML=" << (uint32_t)mlRate << " (trust=" << mlTrust
                                                 << "), Rule=" << (uint32_t)ruleRate
                                                 << " → Final=" << finalRate);

    return finalRate;
}

// ============================================================================
// 🚀 PHASE 1A: FEATURE EXTRACTION - 15 SAFE FEATURES
// ============================================================================
std::vector<double>
SmartWifiManagerRf::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // 🚀 PHASE 1B: Extract exactly 14 features matching Python pipeline
    std::vector<double> features(14);

    // SNR features (7)
    features[0] = station->lastSnr;                                     // lastSnr
    features[1] = station->snrFast;                                     // snrFast
    features[2] = station->snrSlow;                                     // snrSlow
    features[3] = GetSnrTrendShort(st);                                 // snrTrendShort
    features[4] = GetSnrStabilityIndex(st);                             // snrStabilityIndex
    features[5] = GetSnrPredictionConfidence(st);                       // snrPredictionConfidence
    features[6] = std::max(0.0, std::min(100.0, station->snrVariance)); // snrVariance

    // Network state (1 - removed channelWidth)
    features[7] = GetMobilityMetric(st); // mobilityMetric

    // 🚀 PHASE 1A: SAFE ONLY (2 features - removed channelBusyRatio)
    features[8] = station->retryRate;      // retryRate (0-1)
    features[9] = station->frameErrorRate; // frameErrorRate (0-1)

    // 🚀 PHASE 1B: NEW FEATURES (4)
    features[10] = station->rssiVariance;      // rssiVariance (dB²)
    features[11] = station->interferenceLevel; // interferenceLevel (0-1)
    features[12] = station->distanceMetric;    // distanceMetric (m)
    features[13] = station->avgPacketSize;     // avgPacketSize (bytes)

    // Validation: Ensure all features are finite
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (!std::isfinite(features[i]))
        {
            NS_LOG_WARN("Feature " << i << " is not finite, setting to 0.0");
            features[i] = 0.0;
        }
    }

    // Debug logging
    if (m_enableDetailedLogging)
    {
        NS_LOG_DEBUG("[PHASE 1B] Extracted 14 features: "
                     << "SNR=[" << features[0] << "," << features[1] << "," << features[2] << "] "
                     << "Trend=" << features[3] << " Stability=" << features[4] << " Confidence="
                     << features[5] << " Variance=" << features[6] << " Mobility=" << features[7]
                     << " | Phase 1A: Retry=" << features[8] << " Error=" << features[9]
                     << " | Phase 1B: RSSI=" << features[10] << " Intf=" << features[11]
                     << " Dist=" << features[12] << " PktSize=" << features[13]);
    }

    return features;
}

// ============================================================================
// ML Inference via Python server (socket communication)
// ============================================================================
SmartWifiManagerRf::InferenceResult
SmartWifiManagerRf::RunMLInference(const std::vector<double>& features,
                                   const std::string& modelName) const
{
    NS_LOG_FUNCTION(this << modelName);
    InferenceResult result;
    result.success = false;
    result.rateIdx = m_fallbackRate;
    result.latencyMs = 0.0;
    result.confidence = 0.0;
    result.model = modelName;

    // CRITICAL: Validate feature count (15, not 9!)
    if (features.size() != 14)
    {
        result.error = "Invalid feature count: expected 14, got " + std::to_string(features.size());
        NS_LOG_ERROR(result.error);
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        result.error = "socket creation failed";
        NS_LOG_ERROR(result.error);
        return result;
    }

    // Set timeout (150ms - fast response expected from Python server)
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 150000;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    // Connect to Python inference server (localhost:8765)
    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(m_inferenceServerPort);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    int conn_ret = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    if (conn_ret < 0)
    {
        close(sockfd);
        result.error = "connect failed to server on port " + std::to_string(m_inferenceServerPort);
        NS_LOG_WARN(result.error << " - Is Python server running?");
        return result;
    }

    // Build request matching Python server protocol
    // Format: "feat1 feat2 ... feat15 [model_name]\n"
    std::ostringstream featStream;
    for (size_t i = 0; i < features.size(); ++i)
    {
        featStream << std::fixed << std::setprecision(6) << features[i];
        if (i + 1 < features.size())
            featStream << " ";
    }

    // Append model name
    if (!modelName.empty())
    {
        featStream << " " << modelName;
    }
    featStream << "\n";

    std::string req = featStream.str();

    // Send request to Python server
    ssize_t sent = send(sockfd, req.c_str(), req.size(), 0);
    if (sent != static_cast<ssize_t>(req.size()))
    {
        close(sockfd);
        result.error = "send failed (partial send)";
        NS_LOG_ERROR(result.error);
        return result;
    }

    // Receive response from Python server
    std::string response;
    char buffer[4096];
    ssize_t received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);

    close(sockfd);

    if (received <= 0)
    {
        result.error = "no response from server (is Python server running?)";
        NS_LOG_WARN(result.error);
        return result;
    }

    buffer[received] = '\0';
    response = std::string(buffer);

    // Parse JSON response (simple string parsing)
    size_t rate_pos = response.find("\"rateIdx\":");
    if (rate_pos != std::string::npos)
    {
        size_t start = response.find(':', rate_pos) + 1;
        size_t end = response.find_first_of(",}", start);
        if (end != std::string::npos)
        {
            std::string rate_str = response.substr(start, end - start);
            try
            {
                double rate_val = std::stod(rate_str);
                result.rateIdx = static_cast<uint32_t>(std::max(0.0, std::min(7.0, rate_val)));
                result.success = true;

                // Parse confidence (optional)
                size_t conf_pos = response.find("\"confidence\":");
                if (conf_pos != std::string::npos)
                {
                    size_t conf_start = response.find(':', conf_pos) + 1;
                    size_t conf_end = response.find_first_of(",}", conf_start);
                    if (conf_end != std::string::npos)
                    {
                        std::string conf_str = response.substr(conf_start, conf_end - conf_start);
                        result.confidence = std::stod(conf_str);
                    }
                }
            }
            catch (const std::exception& e)
            {
                result.error = "parse error on response: " + std::string(e.what());
                result.success = false;
                NS_LOG_ERROR(result.error);
            }
        }
    }
    else
    {
        result.error = "Invalid response format from Python server";
        result.success = false;
        NS_LOG_ERROR(result.error << " - Response: " << response);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.latencyMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (result.success)
    {
        NS_LOG_DEBUG("ML inference successful: rate=" << result.rateIdx
                                                      << " confidence=" << result.confidence
                                                      << " latency=" << result.latencyMs << "ms"
                                                      << " model=" << modelName);
    }

    return result;
}

// ============================================================================
// Context and Safety Assessment (unchanged from original)
// ============================================================================
WifiContextType
SmartWifiManagerRf::ClassifyNetworkContext(SmartWifiManagerRfState* station) const
{
    double snr = station->lastSnr;

    WifiContextType result;

    if (snr < -20.0)
    {
        result = WifiContextType::EMERGENCY;
    }
    else if (snr < -10.0)
    {
        result = WifiContextType::POOR_UNSTABLE;
    }
    else if (snr < 5.0)
    {
        result = WifiContextType::MARGINAL;
    }
    else if (snr >= 25.0)
    {
        result = WifiContextType::EXCELLENT_STABLE;
    }
    else if (snr >= 15.0)
    {
        result = WifiContextType::GOOD_STABLE;
    }
    else
    {
        result = WifiContextType::GOOD_UNSTABLE;
    }

    return result;
}

std::string
SmartWifiManagerRf::ContextTypeToString(WifiContextType type) const
{
    switch (type)
    {
    case WifiContextType::EMERGENCY:
        return "emergency_recovery";
    case WifiContextType::POOR_UNSTABLE:
        return "poor_unstable";
    case WifiContextType::MARGINAL:
        return "marginal_conditions";
    case WifiContextType::GOOD_UNSTABLE:
        return "good_unstable";
    case WifiContextType::GOOD_STABLE:
        return "good_stable";
    case WifiContextType::EXCELLENT_STABLE:
        return "excellent_stable";
    default:
        return "unknown";
    }
}

double
SmartWifiManagerRf::CalculateAdaptiveConfidenceThreshold(SmartWifiManagerRfState* station,
                                                         WifiContextType context) const
{
    double baseThreshold = m_confidenceThreshold;
    double adaptiveThreshold = baseThreshold;

    double currentDistance = m_benchmarkDistance.load();
    uint32_t currentInterferers = m_currentInterferers.load();

    if (currentDistance <= 25.0 && currentInterferers <= 1)
    {
        adaptiveThreshold = std::max(0.15, baseThreshold - 0.05);
    }
    else if (currentDistance <= 40.0 && currentInterferers <= 2)
    {
        adaptiveThreshold = std::max(0.18, baseThreshold - 0.02);
    }
    else if (currentDistance > 55.0 || currentInterferers > 3)
    {
        adaptiveThreshold = std::min(0.30, baseThreshold + 0.10);
    }

    if (station->mlInferencesSuccessful > 15)
    {
        double performanceBonus = (station->recentMLAccuracy - 0.5) * 0.1;
        adaptiveThreshold = std::max(0.15, adaptiveThreshold + performanceBonus);
    }

    int contextIdx = static_cast<int>(context);
    if (contextIdx >= 0 && contextIdx < 6 && station->mlContextUsage[contextIdx] > 5)
    {
        double contextConfidence = station->mlContextConfidence[contextIdx];
        if (contextConfidence > 0.4)
        {
            adaptiveThreshold = std::max(0.15, adaptiveThreshold - 0.03);
        }
    }

    return adaptiveThreshold;
}

uint32_t
SmartWifiManagerRf::FuseMLAndRuleBased(uint32_t mlRate,
                                       uint32_t ruleRate,
                                       double mlConfidence,
                                       const SafetyAssessment& safety,
                                       SmartWifiManagerRfState* station) const
{
    if (safety.requiresEmergencyAction)
    {
        NS_LOG_INFO("Emergency override: using safe rate " << safety.recommendedSafeRate);
        return safety.recommendedSafeRate;
    }

    double dynamicThreshold = CalculateAdaptiveConfidenceThreshold(station, safety.context);

    if (mlConfidence >= dynamicThreshold)
    {
        uint32_t mlPrimary = mlRate;

        uint32_t maxJump = 2;
        if (mlConfidence > 0.35)
            maxJump = 3;

        uint32_t upperBound = std::min(ruleRate + maxJump, static_cast<uint32_t>(7));
        uint32_t lowerBound = (ruleRate > maxJump) ? ruleRate - maxJump : 0;

        if (mlPrimary > upperBound)
            mlPrimary = upperBound;
        else if (mlPrimary < lowerBound)
            mlPrimary = lowerBound;

        double mlWeight = 0.6 + (mlConfidence - dynamicThreshold) * 1.5;
        mlWeight = std::min(0.85, mlWeight);
        double ruleWeight = 1.0 - mlWeight;

        double fusedRate = (mlWeight * mlPrimary) + (ruleWeight * ruleRate);
        uint32_t finalRate = static_cast<uint32_t>(std::round(fusedRate));

        NS_LOG_DEBUG("ML-led fusion: mlRate=" << mlRate << " ruleRate=" << ruleRate << " mlWeight="
                                              << mlWeight << " final=" << finalRate);

        return finalRate;
    }
    else if (mlConfidence >= dynamicThreshold * 0.65)
    {
        double mlWeight = 0.35 + (mlConfidence / dynamicThreshold) * 0.25;
        double ruleWeight = 1.0 - mlWeight;

        double balancedRate = (mlWeight * mlRate) + (ruleWeight * ruleRate);
        uint32_t finalRate = static_cast<uint32_t>(std::round(balancedRate));

        finalRate = std::min(finalRate, std::max(mlRate, ruleRate) + 1);

        NS_LOG_DEBUG("Balanced fusion: mlRate=" << mlRate << " ruleRate=" << ruleRate
                                                << " mlWeight=" << mlWeight
                                                << " final=" << finalRate);

        return finalRate;
    }
    else if (mlConfidence >= dynamicThreshold * 0.4)
    {
        uint32_t ruleWithHint = ruleRate;

        if (std::abs(static_cast<int>(mlRate) - static_cast<int>(ruleRate)) <= 2)
        {
            ruleWithHint = (mlRate + ruleRate) / 2;
        }
        else if (mlRate > ruleRate && mlConfidence > 0.18)
        {
            ruleWithHint = std::min(ruleRate + 1, mlRate);
        }
        else if (mlRate < ruleRate && mlConfidence > 0.18)
        {
            ruleWithHint = std::max(ruleRate > 0 ? ruleRate - 1 : 0, mlRate);
        }

        NS_LOG_DEBUG("Rule with hint: mlRate=" << mlRate << " ruleRate=" << ruleRate
                                               << " final=" << ruleWithHint);

        return ruleWithHint;
    }
    else
    {
        NS_LOG_DEBUG("Rule-only: mlConf=" << mlConfidence
                                          << " below threshold, using ruleRate=" << ruleRate);
        return ruleRate;
    }
}

// ============================================================================
// Enhanced rule-based rate adaptation (unchanged from original)
// ============================================================================
uint32_t
SmartWifiManagerRf::GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station,
                                             const SafetyAssessment& safety) const
{
    double snr = station->lastSnr;
    double snrFast = station->snrFast;
    double snrSlow = station->snrSlow;
    double effectiveSnr = (snrFast * 0.7 + snrSlow * 0.3);

    uint32_t baseRate;
    if (effectiveSnr >= 35)
        baseRate = 7;
    else if (effectiveSnr >= 28)
        baseRate = 6;
    else if (effectiveSnr >= 22)
        baseRate = 6;
    else if (effectiveSnr >= 16)
        baseRate = 5;
    else if (effectiveSnr >= 11)
        baseRate = 4;
    else if (effectiveSnr >= 6)
        baseRate = 3;
    else if (effectiveSnr >= 1)
        baseRate = 2;
    else if (effectiveSnr >= -5)
        baseRate = 1;
    else
        baseRate = 0;

    double snrTrend = GetSnrTrendShort(
        const_cast<WifiRemoteStation*>(static_cast<const WifiRemoteStation*>(station)));
    double snrStability = GetSnrStabilityIndex(
        const_cast<WifiRemoteStation*>(static_cast<const WifiRemoteStation*>(station)));
    double snrVariance = station->snrVariance;

    int stabilityAdjustment = 0;
    if (snrStability > 8.0 && snrVariance < 2.0)
        stabilityAdjustment = +1;
    else if (snrStability < 4.0 || snrVariance > 10.0)
        stabilityAdjustment = -1;

    int trendAdjustment = 0;
    if (snrTrend > 2.0)
        trendAdjustment = +1;
    else if (snrTrend < -2.0)
        trendAdjustment = -1;

    double mobility = GetMobilityMetric(
        const_cast<WifiRemoteStation*>(static_cast<const WifiRemoteStation*>(station)));

    int mobilityAdjustment = 0;
    if (mobility > 0.7)
        mobilityAdjustment = -1;
    else if (mobility < 0.2 && snrStability > 7.0)
        mobilityAdjustment = +1;

    int contextAdjustment = 0;
    switch (safety.context)
    {
    case WifiContextType::EXCELLENT_STABLE:
        contextAdjustment = +1;
        break;
    case WifiContextType::GOOD_STABLE:
        contextAdjustment = 0;
        break;
    case WifiContextType::GOOD_UNSTABLE:
        contextAdjustment = -1;
        break;
    case WifiContextType::MARGINAL:
        contextAdjustment = -1;
        break;
    case WifiContextType::POOR_UNSTABLE:
        contextAdjustment = -2;
        break;
    case WifiContextType::EMERGENCY:
        return safety.recommendedSafeRate;
    default:
        contextAdjustment = 0;
    }

    int totalAdjustment =
        stabilityAdjustment + trendAdjustment + mobilityAdjustment + contextAdjustment;

    int adjustedRate = static_cast<int>(baseRate) + totalAdjustment;
    adjustedRate = std::max(0, std::min(7, adjustedRate));

    if (std::abs(static_cast<int>(adjustedRate) - static_cast<int>(station->currentRateIndex)) > 2)
    {
        if (adjustedRate > static_cast<int>(station->currentRateIndex))
            adjustedRate = station->currentRateIndex + 2;
        else
            adjustedRate = std::max(0, static_cast<int>(station->currentRateIndex) - 2);
    }

    if (safety.riskLevel > 0.6 && adjustedRate > 4)
    {
        adjustedRate = std::min(adjustedRate, 4);
    }

    NS_LOG_DEBUG("Enhanced rule: SNR=" << snr << " base=" << baseRate << " adjustments="
                                       << totalAdjustment << " final=" << adjustedRate);

    return static_cast<uint32_t>(adjustedRate);
}

SafetyAssessment
SmartWifiManagerRf::AssessNetworkSafety(SmartWifiManagerRfState* station)
{
    SafetyAssessment assessment;
    assessment.context = ClassifyNetworkContext(station);
    assessment.riskLevel = CalculateRiskLevel(station);
    assessment.recommendedSafeRate = GetContextSafeRate(station, assessment.context);

    assessment.requiresEmergencyAction = (assessment.context == WifiContextType::EMERGENCY ||
                                          assessment.riskLevel > m_riskThreshold);

    assessment.confidenceInAssessment = 1.0 - assessment.riskLevel;
    assessment.contextStr = ContextTypeToString(assessment.context);
    station->lastContext = assessment.context;
    station->lastRiskLevel = assessment.riskLevel;

    assessment.managerRef = this;
    assessment.stationId = station->stationId;

    return assessment;
}

double
SmartWifiManagerRf::CalculateRiskLevel(SmartWifiManagerRfState* station) const
{
    double risk = 0.0;
    risk += (station->snrVariance > 8.0) ? 0.2 : 0.0;
    risk += (station->lastSnr < 0.0) ? 0.3 : 0.0;

    double packetLossEstimate =
        (station->lostPackets > 0 && station->totalPackets > 0)
            ? static_cast<double>(station->lostPackets) / station->totalPackets
            : 0.0;
    risk += packetLossEstimate * 0.2;

    return std::min(1.0, risk);
}

uint32_t
SmartWifiManagerRf::GetContextSafeRate(SmartWifiManagerRfState* station,
                                       WifiContextType context) const
{
    switch (context)
    {
    case WifiContextType::EMERGENCY:
        return 0;
    case WifiContextType::POOR_UNSTABLE:
        return 1;
    case WifiContextType::MARGINAL:
        return 3;
    case WifiContextType::GOOD_UNSTABLE:
        return 5;
    case WifiContextType::GOOD_STABLE:
        return 6;
    case WifiContextType::EXCELLENT_STABLE:
        return 7;
    default:
        return m_fallbackRate;
    }
}

// ============================================================================
// 🚀 MAIN RATE DECISION ENGINE - DoGetDataTxVector (PHASE 1-4 INTEGRATED)
// ============================================================================
// ============================================================================
// 🚀 MAIN RATE DECISION ENGINE - DoGetDataTxVector (PHASE 1B ENHANCED)
// ============================================================================
WifiTxVector
SmartWifiManagerRf::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // ========================================================================
    // 🚀 PHASE 1B: UPDATE STATION STATE WITH CURRENT GLOBALS
    // ========================================================================
    double currentDistance = m_benchmarkDistance.load();
    uint32_t currentInterferers = m_currentInterferers.load();

    // Update distance metric (Phase 1B feature 12)
    station->distanceMetric = currentDistance;

    // If station SNR is still at initialization value, recalculate from current distance
    if (station->lastSnr > 15.0 && currentDistance > 50.0)
    {
        // Likely stale - recalculate
        double realisticSnr =
            ConvertNS3ToRealisticSnr(100.0, currentDistance, currentInterferers, SOFT_MODEL);
        station->lastSnr = realisticSnr;
        station->snrSlow = realisticSnr;
        station->snrFast = realisticSnr;

        NS_LOG_WARN("[PHASE 1B] SNR looks stale, recalculating: "
                    << realisticSnr << " dB (distance=" << currentDistance << "m)");
    }

    uint32_t supportedRates = GetNSupported(st);
    uint32_t maxRateIndex =
        std::min(supportedRates > 0 ? supportedRates - 1 : 0, static_cast<uint32_t>(7));

    // ========================================================================
    // STAGE 1: SAFETY ASSESSMENT
    // ========================================================================
    SafetyAssessment safety = AssessNetworkSafety(station);
    safety.managerRef = this;
    safety.stationId = station->stationId;

    // ========================================================================
    // STAGE 2: RULE-BASED BASELINE
    // ========================================================================
    uint32_t ruleRate = GetEnhancedRuleBasedRate(station, safety);

    // ========================================================================
    // 🚀 PHASE 2: SCENARIO-AWARE MODEL SELECTION
    // ========================================================================
    std::string selectedModel = SelectBestModel(station);

    // Check if model changed
    if (selectedModel != m_currentModelName)
    {
        NS_LOG_INFO("[PHASE 2] MODEL SWITCH: " << m_currentModelName << " → " << selectedModel);
        std::cout << "[PHASE 2] MODEL SWITCH: " << m_currentModelName << " → " << selectedModel
                  << " (difficulty changed, dist=" << currentDistance
                  << "m, intf=" << currentInterferers << ")" << std::endl;
        m_currentModelName = selectedModel;
    }

    // ========================================================================
    // STAGE 3: ML INFERENCE (WITH ADAPTIVE FREQUENCY)
    // ========================================================================
    uint32_t mlRate = ruleRate;
    double mlConfidence = 0.0;
    std::string mlStatus = "NO_ATTEMPT";

    static uint64_t s_callCounter = 0;
    ++s_callCounter;

    Time now = Simulator::Now();

    // Check ML cache
    bool canUseCachedMl = false;
    {
        std::lock_guard<std::mutex> lock(m_mlCacheMutex);
        canUseCachedMl =
            (now - m_lastMlTime) < MilliSeconds(m_mlCacheTime) && m_lastMlTime > Seconds(0);
    }

    // 🚀 PHASE 1B: SMARTER ADAPTIVE INFERENCE FREQUENCY
    // Consider distance, interferers, AND signal stability
    uint32_t adaptiveInferencePeriod = m_inferencePeriod;

    if (currentDistance <= 30.0 && currentInterferers <= 1 && station->rssiVariance < 2.0)
    {
        // Excellent conditions: can infer more frequently (faster adaptation)
        adaptiveInferencePeriod = std::max(static_cast<uint32_t>(10), m_inferencePeriod / 2);
        NS_LOG_DEBUG(
            "[PHASE 1B] Excellent conditions → inference period: " << adaptiveInferencePeriod);
    }
    else if (currentDistance > 70.0 || currentInterferers > 3 || station->rssiVariance > 8.0)
    {
        // Poor conditions: infer less frequently (more conservative)
        adaptiveInferencePeriod = std::min(static_cast<uint32_t>(100), m_inferencePeriod * 2);
        NS_LOG_DEBUG("[PHASE 1B] Poor conditions → inference period: " << adaptiveInferencePeriod);
    }

    bool needNewMlInference = !safety.requiresEmergencyAction &&
                              safety.riskLevel < m_riskThreshold && !canUseCachedMl &&
                              (s_callCounter % adaptiveInferencePeriod) == 0;

    if (canUseCachedMl)
    {
        std::lock_guard<std::mutex> lock(m_mlCacheMutex);
        mlRate = m_lastMlRate;
        mlConfidence = m_lastMlConfidence;
        mlStatus = "CACHED";
        m_mlCacheHits++;
    }
    else if (needNewMlInference)
    {
        mlStatus = "ATTEMPTING";

        // 🚀 PHASE 1B: Extract 14 features (not 15!)
        std::vector<double> features = ExtractFeatures(st);

        // 🚀 PHASE 2: Call Python server with dynamically selected model
        InferenceResult result = RunMLInference(features, selectedModel);

        if (result.success && result.confidence > 0.05)
        {
            m_mlInferences++;
            station->mlInferencesReceived++;
            station->mlInferencesSuccessful++;

            mlRate = std::min(result.rateIdx, maxRateIndex);
            mlConfidence = result.confidence;

            {
                std::lock_guard<std::mutex> lock(m_mlCacheMutex);
                m_lastMlRate = mlRate;
                m_lastMlTime = now;
                m_lastMlConfidence = mlConfidence;
            }

            mlStatus = "SUCCESS";

            // Update ML accuracy tracking
            station->recentMLAccuracy = 0.9 * station->recentMLAccuracy + 0.1 * mlConfidence;

            // Update context-specific confidence
            int contextIdx = static_cast<int>(safety.context);
            if (contextIdx >= 0 && contextIdx < 6)
            {
                station->mlContextConfidence[contextIdx] =
                    0.8 * station->mlContextConfidence[contextIdx] + 0.2 * mlConfidence;
                station->mlContextUsage[contextIdx]++;
            }

            NS_LOG_INFO("[PHASE 1B] ML SUCCESS: model="
                        << result.model << " rate=" << result.rateIdx << " conf=" << mlConfidence
                        << " SNR=" << station->lastSnr << "dB"
                        << " rssiVar=" << station->rssiVariance << " intf="
                        << station->interferenceLevel << " dist=" << currentDistance << "m"
                        << " latency=" << result.latencyMs << "ms");
        }
        else
        {
            m_mlFailures++;
            mlRate = ruleRate;
            mlConfidence = 0.0;
            mlStatus = "FAILED";
            NS_LOG_WARN("ML FAILED: " << result.error << " - Using rule fallback: " << ruleRate);
        }
    }

    // ========================================================================
    // 🚀 PHASE 4: ADAPTIVE ML FUSION
    // ========================================================================
    uint32_t fusedRate;
    if (mlConfidence >= CalculateAdaptiveConfidenceThreshold(station, safety.context))
    {
        // Use adaptive fusion (Phase 4)
        fusedRate = AdaptiveFusion(mlRate, ruleRate, mlConfidence, station);
    }
    else
    {
        // Fallback to rule-based with ML hint
        fusedRate = FuseMLAndRuleBased(mlRate, ruleRate, mlConfidence, safety, station);
    }

    // ========================================================================
    // 🚀 PHASE 3: APPLY HYSTERESIS TO PREVENT RATE THRASHING
    // ========================================================================
    uint32_t finalRate = ApplyHysteresis(station, station->currentRateIndex, fusedRate);

    // ========================================================================
    // STAGE 5: FINAL BOUNDS AND TRACKING
    // ========================================================================
    finalRate = std::min(finalRate, maxRateIndex);
    finalRate = std::max(finalRate, static_cast<uint32_t>(0));

    // Update station state
    if (finalRate != station->currentRateIndex)
    {
        bool wasMLInfluenced =
            (mlConfidence >= CalculateAdaptiveConfidenceThreshold(station, safety.context));
        if (wasMLInfluenced)
        {
            station->lastMLInfluencedRate = finalRate;
            station->lastMLInfluenceTime = now;
        }

        station->previousRateIndex = station->currentRateIndex;
        station->currentRateIndex = finalRate;
        station->lastRateChangeTime = now;
    }

    // 🚀 PHASE 1B: Update enhanced features (14 features)
    UpdateEnhancedFeatures(station);

    // Determine fusion type
    std::string fusionType =
        (mlConfidence >= CalculateAdaptiveConfidenceThreshold(station, safety.context))
            ? "ML-LED"
            : "RULE-LED";

    uint64_t finalDataRate = GetSupported(st, finalRate).GetDataRate(allowedWidth);

    // 🚀 PHASE 1B: Enhanced logging
    NS_LOG_INFO("[PHASE 1B DECISION] "
                << fusionType << " | Model=" << selectedModel << " | SNR=" << station->lastSnr
                << "dB"
                << " | Context=" << safety.contextStr << " | Rule=" << ruleRate
                << " | ML=" << mlRate << "(conf=" << mlConfidence << ")"
                << " | Fused=" << fusedRate << " | Final=" << finalRate << " (hysteresis="
                << station->ratePredictionStreak << "/" << m_hysteresisStreak << ")"
                << " | Rate=" << (finalDataRate / 1e6) << "Mbps"
                << " | Status=" << mlStatus << " | rssiVar=" << station->rssiVariance << " | intf="
                << station->interferenceLevel << " | dist=" << currentDistance << "m");

    WifiMode mode = GetSupported(st, finalRate);
    uint64_t rate = mode.GetDataRate(allowedWidth);

    if (m_currentRate != rate)
    {
        NS_LOG_INFO("Rate changed: " << m_currentRate << " -> " << rate << " (index " << finalRate
                                     << ")");
        m_currentRate = rate;
    }

    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        800,
        1,
        1,
        0,
        allowedWidth,
        GetAggregation(st));
}

WifiTxVector
SmartWifiManagerRf::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    WifiMode mode = GetSupported(st, 0);
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        800,
        1,
        1,
        0,
        GetChannelWidth(st),
        GetAggregation(st));
}

// ============================================================================
// 🚀 SNR REPORTING - DoReportRxOk (PHASE 1B ENHANCED)
// ============================================================================
void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lastRawSnr = rxSnr;
    double realisticSnr = ConvertToRealisticSnr(rxSnr);
    station->lastSnr = realisticSnr;

    // Update SNR history
    station->snrHistory.push_back(realisticSnr);
    station->rawSnrHistory.push_back(rxSnr);
    if (station->snrHistory.size() > 20)
    {
        station->snrHistory.pop_front();
        station->rawSnrHistory.pop_front();
    }

    // Update fast/slow SNR tracking
    if (station->snrFast == 0.0)
    {
        station->snrFast = realisticSnr;
        station->snrSlow = realisticSnr;
    }
    else
    {
        station->snrFast = m_snrAlpha * realisticSnr + (1 - m_snrAlpha) * station->snrFast;
        station->snrSlow =
            (m_snrAlpha / 10) * realisticSnr + (1 - m_snrAlpha / 10) * station->snrSlow;
    }

    // 🚀 PHASE 1B: Update rssiVariance (feature 10) from SNR history
    if (station->snrHistory.size() >= 5)
    {
        double mean = 0.0;
        for (double snr : station->snrHistory)
        {
            mean += snr;
        }
        mean /= station->snrHistory.size();

        double variance = 0.0;
        for (double snr : station->snrHistory)
        {
            double diff = snr - mean;
            variance += diff * diff;
        }
        variance /= station->snrHistory.size();

        // Exponential smoothing for rssiVariance
        station->rssiVariance = 0.8 * station->rssiVariance + 0.2 * variance;
    }

    UpdateMetrics(st, true, realisticSnr);
}

// ============================================================================
// 🚀 DATA SUCCESS REPORTING - DoReportDataOk (PHASE 1B ENHANCED)
// ============================================================================
void
SmartWifiManagerRf::DoReportDataOk(WifiRemoteStation* st,
                                   double ackSnr,
                                   WifiMode ackMode,
                                   double dataSnr,
                                   uint16_t dataChannelWidth,
                                   uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << dataNss);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lastRawSnr = dataSnr;
    double realisticDataSnr = ConvertToRealisticSnr(dataSnr);
    station->lastSnr = realisticDataSnr;

    station->totalPackets++;
    station->successfulPackets++;

    // 🚀 PHASE 1B: Update interferenceLevel (feature 11)
    // Success reduces interference estimate (exponential decay)
    station->interferenceLevel = 0.95 * station->interferenceLevel;

    // 🚀 PHASE 1B: Update average packet size (feature 13)
    // Track actual packet sizes if available (future enhancement)
    // For now, keep at default 1200 bytes

    UpdateMetrics(st, true, realisticDataSnr);
}

// ============================================================================
// 🚀 DATA FAILURE REPORTING - DoReportDataFailed (PHASE 1B ENHANCED)
// ============================================================================
void
SmartWifiManagerRf::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lostPackets++;
    station->failedPackets++;

    // 🚀 PHASE 1B: Update interferenceLevel (feature 11) from failures
    // Failed packets suggest interference/collisions
    // Use exponential moving average to track collision rate
    if (station->totalPackets > 0)
    {
        double failureRate = static_cast<double>(station->failedPackets) / station->totalPackets;

        // Update interference level (blend current failure rate with history)
        station->interferenceLevel = 0.8 * station->interferenceLevel + 0.2 * failureRate;

        // Clamp to [0, 1]
        station->interferenceLevel = std::min(1.0, std::max(0.0, station->interferenceLevel));
    }

    UpdateMetrics(st, false, station->lastSnr);
}

// ============================================================================
// 🚀 RTS/CTS REPORTING - Enhanced for Phase 1B
// ============================================================================
void
SmartWifiManagerRf::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // 🚀 PHASE 1B: RTS failure suggests channel is busy (interference)
    station->interferenceLevel = std::min(1.0, station->interferenceLevel + 0.05);

    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerRf::DoReportRtsOk(WifiRemoteStation* st,
                                  double ctsSnr,
                                  WifiMode ctsMode,
                                  double rtsSnr)
{
    NS_LOG_FUNCTION(this << st << ctsSnr << ctsMode << rtsSnr);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lastRawSnr = rtsSnr;
    double realisticRtsSnr = ConvertToRealisticSnr(rtsSnr);
    station->lastSnr = realisticRtsSnr;

    // 🚀 PHASE 1B: RTS success suggests channel is clear
    station->interferenceLevel = std::max(0.0, station->interferenceLevel - 0.02);
}

void
SmartWifiManagerRf::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lostPackets++;

    // 🚀 PHASE 1B: Final RTS failure = severe interference
    station->interferenceLevel = std::min(1.0, station->interferenceLevel + 0.10);

    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerRf::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lostPackets++;

    // 🚀 PHASE 1B: Final data failure = severe problem
    station->interferenceLevel = std::min(1.0, station->interferenceLevel + 0.15);

    UpdateMetrics(st, false, station->lastSnr);
}

// ============================================================================
// 🚀 PHASE 1A: UPDATE ENHANCED FEATURES
// ============================================================================
void
SmartWifiManagerRf::UpdateEnhancedFeatures(SmartWifiManagerRfState* station)
{
    NS_LOG_FUNCTION(this << station);

    // ============================================================
    // PHASE 1A: SAFE FEATURES (2 features)
    // ============================================================

    // Feature 8: Retry Rate (from MAC layer stats)
    if (station->totalPackets > 0)
    {
        double failureRate = static_cast<double>(station->failedPackets) / station->totalPackets;
        station->retryRate = std::min(1.0, failureRate * 1.5);
    }
    else
    {
        station->retryRate = 0.0;
    }

    // Feature 9: Frame Error Rate (from PHY layer stats)
    if (station->totalPackets > 0)
    {
        station->frameErrorRate =
            std::min(1.0, static_cast<double>(station->lostPackets) / station->totalPackets);
    }
    else
    {
        station->frameErrorRate = 0.0;
    }

    // ============================================================
    // PHASE 1B: NEW FEATURES (4 features)
    // ============================================================

    // Feature 10: RSSI Variance (from SNR history)
    if (station->snrHistory.size() >= 3)
    {
        double mean = 0.0;
        for (double snr : station->snrHistory)
        {
            mean += snr;
        }
        mean /= station->snrHistory.size();

        double variance = 0.0;
        for (double snr : station->snrHistory)
        {
            double diff = snr - mean;
            variance += diff * diff;
        }
        variance /= station->snrHistory.size();

        station->rssiVariance = variance; // dB²
    }
    else
    {
        station->rssiVariance = 0.0;
    }

    // Feature 11: Interference Level (from collision tracking)
    // Estimate from failure rate (proxy for collisions)
    if (station->totalPackets > 0)
    {
        station->interferenceLevel =
            std::min(1.0, static_cast<double>(station->failedPackets) / station->totalPackets);
    }
    else
    {
        station->interferenceLevel = 0.0;
    }

    // Feature 12: Distance Metric (from global benchmark)
    station->distanceMetric = m_benchmarkDistance.load();

    // Feature 13: Average Packet Size (constant for now, can be extended)
    station->avgPacketSize = 1200.0; // Default MTU, could track from MAC layer

    NS_LOG_DEBUG("[PHASE 1B] Enhanced features updated: "
                 << "retry=" << station->retryRate << " error=" << station->frameErrorRate
                 << " rssiVar=" << station->rssiVariance << " intf=" << station->interferenceLevel
                 << " dist=" << station->distanceMetric << " pktSize=" << station->avgPacketSize);
}

// ============================================================================
// Metrics update (simplified)
// ============================================================================
void
SmartWifiManagerRf::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();

    // Update SNR metrics
    if (snr >= -30.0 && snr <= 45.0)
    {
        if (station->snrFast == 0.0 && station->snrSlow == 0.0)
        {
            station->snrFast = snr;
            station->snrSlow = snr;
            station->snrVariance = 0.1;
        }
        else
        {
            double oldFast = station->snrFast;
            station->snrFast = m_snrAlpha * snr + (1 - m_snrAlpha) * station->snrFast;
            station->snrSlow = (m_snrAlpha / 10) * snr + (1 - m_snrAlpha / 10) * station->snrSlow;
            double diff = snr - oldFast;
            station->snrVariance = 0.9 * station->snrVariance + 0.1 * (diff * diff);
        }
    }

    station->lastUpdateTime = now;
}

// ============================================================================
// Helper functions (safe features)
// ============================================================================
double
SmartWifiManagerRf::GetSnrTrendShort(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    if (station->snrHistory.size() < 2)
        return 0.0;

    double recent = 0.0, older = 0.0;
    size_t halfSize = station->snrHistory.size() / 2;

    for (size_t i = halfSize; i < station->snrHistory.size(); ++i)
    {
        recent += station->snrHistory[i];
    }
    for (size_t i = 0; i < halfSize; ++i)
    {
        older += station->snrHistory[i];
    }

    if (halfSize > 0)
    {
        recent /= (station->snrHistory.size() - halfSize);
        older /= halfSize;
        return std::max(-10.0, std::min(10.0, recent - older));
    }
    return 0.0;
}

double
SmartWifiManagerRf::GetSnrStabilityIndex(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    return std::max(0.0, std::min(10.0, 10.0 / (1.0 + station->snrVariance)));
}

double
SmartWifiManagerRf::GetSnrPredictionConfidence(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    double stabilityFactor = GetSnrStabilityIndex(st) / 10.0;
    return std::max(0.0, std::min(1.0, stabilityFactor));
}

double
SmartWifiManagerRf::GetMobilityMetric(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // FIXED: Get actual node speed from MobilityModel (NOT SNR variance!)
    Ptr<WifiPhy> phy = GetPhy();
    if (phy == nullptr)
    {
        NS_LOG_WARN("No PHY available, using fallback mobility metric");
        station->mobilityMetric = 0.0;
        return 0.0;
    }

    Ptr<NetDevice> device = phy->GetDevice();
    if (device == nullptr)
    {
        NS_LOG_WARN("No device available, using fallback mobility metric");
        station->mobilityMetric = 0.0;
        return 0.0;
    }

    Ptr<Node> node = device->GetNode();
    if (node == nullptr)
    {
        NS_LOG_WARN("No node available, using fallback mobility metric");
        station->mobilityMetric = 0.0;
        return 0.0;
    }

    // Get MobilityModel from node
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    if (mobility == nullptr)
    {
        NS_LOG_DEBUG("No MobilityModel found, assuming stationary (speed = 0)");
        station->mobilityMetric = 0.0;
        return 0.0;
    }

    // Calculate actual speed from velocity vector
    Vector velocity = mobility->GetVelocity();
    double speed =
        std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);

    // Clamp speed to reasonable range (0-50 m/s)
    speed = std::max(0.0, std::min(50.0, speed));

    station->mobilityMetric = speed;

    NS_LOG_DEBUG("Mobility metric calculated: speed=" << speed << " m/s");

    return speed;
}

void
SmartWifiManagerRf::DebugPrintCurrentConfig() const
{
    std::cout << "============================================================================"
              << std::endl;
    std::cout << "🚀 SMART-RF CONFIG (PHASE 1-4 COMPLETE)" << std::endl;
    std::cout << "============================================================================"
              << std::endl;
    std::cout << "[CONFIG] Distance: " << m_benchmarkDistance.load() << "m" << std::endl;
    std::cout << "[CONFIG] Interferers: " << m_currentInterferers.load() << std::endl;
    std::cout << "[CONFIG] Strategy: " << m_oracleStrategy << " (will switch dynamically)"
              << std::endl;
    std::cout << "[CONFIG] Features: 14 (Phase 1B: 7 SNR + 1 network + 2 Phase 1A + 4 Phase 1B)"
              << std::endl;
    std::cout << "[CONFIG] Python Server Port: " << m_inferenceServerPort << std::endl;
    std::cout << "[CONFIG] Model Path: " << m_modelPath << std::endl;
    std::cout << "[CONFIG] Confidence Threshold: " << m_confidenceThreshold << std::endl;
    std::cout << "[CONFIG] ML Weight: " << m_mlGuidanceWeight << " (adaptive - Phase 4)"
              << std::endl;
    std::cout << "[CONFIG] Hysteresis Streak: " << m_hysteresisStreak << " (Phase 3)" << std::endl;
    std::cout << "[CONFIG] Scenario-Aware: "
              << (m_enableScenarioAwareSelection ? "ENABLED" : "DISABLED") << " (Phase 2)"
              << std::endl;
    std::cout << "============================================================================"
              << std::endl;
}

void
SmartWifiManagerRf::SetBenchmarkDistanceAttribute(double dist)
{
    m_benchmarkDistance.store(dist);
    std::cout << "[ATTR SET] BenchmarkDistance=" << dist << "m" << std::endl;
}

double
SmartWifiManagerRf::GetBenchmarkDistanceAttribute() const
{
    return m_benchmarkDistance.load();
}

void
SmartWifiManagerRf::SetInterferersAttribute(uint32_t count)
{
    m_currentInterferers.store(count);
    std::cout << "[ATTR SET] BenchmarkInterferers=" << count << std::endl;
}

uint32_t
SmartWifiManagerRf::GetInterferersAttribute() const
{
    return m_currentInterferers.load();
}

} // namespace ns3