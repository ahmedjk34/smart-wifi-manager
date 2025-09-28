/*
 * Enhanced Smart WiFi Manager Implementation - ML-FIRST INTELLIGENT SYSTEM
 * Compatible with ahmedjk34's Enhanced ML Pipeline (49.9% realistic accuracy)
 *
 * REVOLUTIONARY: ML-First approach with intelligent safety guards
 * ADAPTIVE: Dynamic confidence thresholds based on conditions
 * LEARNING: System builds trust in ML over time
 * BALANCED: High ML usage while preventing catastrophic failures
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34/smart-wifi-manager)
 * Date: 2025-09-28
 */

#include "smart-wifi-manager-rf.h"

#include "ns3/assert.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-phy.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SmartWifiManagerRf");
NS_OBJECT_ENSURE_REGISTERED(SmartWifiManagerRf);

// FIXED: Global realistic SNR conversion function
double
ConvertNS3ToRealisticSnr(double ns3Value, double distance, uint32_t interferers)
{
    double realisticSnr;

    // Distance-based realistic SNR calculation
    if (distance <= 10.0)
    {
        realisticSnr = 45.0 - (distance * 1.0);
    }
    else if (distance <= 30.0)
    {
        realisticSnr = 35.0 - ((distance - 10.0) * 0.75);
    }
    else if (distance <= 60.0)
    {
        realisticSnr = 20.0 - ((distance - 30.0) * 0.5);
    }
    else
    {
        realisticSnr = 5.0 - ((distance - 60.0) * 0.25);
    }

    realisticSnr -= (interferers * 3.0);
    double variation = fmod(ns3Value, 20.0) - 10.0;
    realisticSnr += variation * 0.3;
    realisticSnr = std::max(-30.0, std::min(45.0, realisticSnr));

    return realisticSnr;
}

TypeId
SmartWifiManagerRf::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SmartWifiManagerRf")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<SmartWifiManagerRf>()
            .AddAttribute("ModelPath",
                          "Path to the Random Forest model file (.joblib)",
                          StringValue("step3_rf_oracle_balanced_model_FIXED.joblib"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelPath),
                          MakeStringChecker())
            .AddAttribute("ScalerPath",
                          "Path to the scaler file (.joblib)",
                          StringValue("step3_scaler_oracle_balanced_FIXED.joblib"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_scalerPath),
                          MakeStringChecker())
            .AddAttribute("ModelName",
                          "Specific model name for inference server",
                          StringValue("oracle_balanced"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelName),
                          MakeStringChecker())
            .AddAttribute("OracleStrategy",
                          "Oracle strategy",
                          StringValue("oracle_balanced"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_oracleStrategy),
                          MakeStringChecker())
            .AddAttribute("InferenceServerPort",
                          "Port number of ML inference server",
                          UintegerValue(8765),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferenceServerPort),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("ConfidenceThreshold",
                          "Base ML confidence threshold (adaptive)",
                          DoubleValue(0.20), // LOWERED for more ML usage
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_confidenceThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("MLGuidanceWeight",
                          "Weight of ML guidance in fusion",
                          DoubleValue(0.75), // INCREASED for more ML influence
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_mlGuidanceWeight),
                          MakeDoubleChecker<double>())
            .AddAttribute("InferencePeriod",
                          "Period between ML inferences",
                          UintegerValue(25), // REDUCED for more frequent ML
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferencePeriod),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EnableAdaptiveWeighting",
                          "Enable adaptive ML weighting",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableAdaptiveWeighting),
                          MakeBooleanChecker())
            .AddAttribute("MLCacheTime",
                          "ML result cache time in ms",
                          UintegerValue(150), // REDUCED for fresher predictions
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_mlCacheTime),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("WindowSize",
                          "Success ratio window size",
                          UintegerValue(20),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_windowSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SnrAlpha",
                          "SNR exponential smoothing alpha",
                          DoubleValue(0.1),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_snrAlpha),
                          MakeDoubleChecker<double>())
            .AddAttribute("FallbackRate",
                          "Fallback rate index on ML failure",
                          UintegerValue(3),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_fallbackRate),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RiskThreshold",
                          "Risk threshold for emergency actions",
                          DoubleValue(0.7),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_riskThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("FailureThreshold",
                          "Consecutive failures for emergency",
                          UintegerValue(5), // INCREASED tolerance
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_failureThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddTraceSource("Rate",
                            "Remote station data rate changed",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_currentRate),
                            "ns3::TracedValueCallback::Uint64")
            .AddTraceSource("MLInferences",
                            "Number of ML inferences made",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_mlInferences),
                            "ns3::TracedValueCallback::Uint32");
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
      m_enableDetailedLogging(true),
      m_modelType("oracle"),
      m_enableProbabilities(true),
      m_enableValidation(true),
      m_inferenceServerPort(8765)
{
    NS_LOG_FUNCTION(this);
}

SmartWifiManagerRf::~SmartWifiManagerRf()
{
    NS_LOG_FUNCTION(this);
}

void
SmartWifiManagerRf::DoInitialize()
{
    NS_LOG_FUNCTION(this);

    if (GetHtSupported() || GetVhtSupported() || GetHeSupported())
    {
        NS_FATAL_ERROR("SmartWifiManagerRf does not support HT/VHT/HE modes");
    }

    std::cout << "[ML-FIRST ENGINE] SmartWifiManagerRf v4.0 initialized - ML-FORWARD APPROACH"
              << std::endl;
    std::cout << "[ML-FIRST ENGINE] Model: " << m_modelName << " | Strategy: " << m_oracleStrategy
              << std::endl;
    std::cout << "[ML-FIRST ENGINE] Confidence Threshold: " << m_confidenceThreshold
              << " (adaptive)" << std::endl;
    std::cout << "[ML-FIRST ENGINE] ML Weight: " << m_mlGuidanceWeight
              << " | Inference Period: " << m_inferencePeriod << std::endl;

    WifiRemoteStationManager::DoInitialize();
}

WifiRemoteStation*
SmartWifiManagerRf::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    SmartWifiManagerRfState* station = new SmartWifiManagerRfState;

    double initialSnr = ConvertNS3ToRealisticSnr(100.0, m_benchmarkDistance, m_currentInterferers);

    // Core SNR metrics
    station->lastSnr = initialSnr;
    station->lastRawSnr = 0.0;
    station->snrFast = initialSnr;
    station->snrSlow = initialSnr;
    station->snrTrendShort = 0.0;
    station->snrStabilityIndex = 1.0;
    station->snrPredictionConfidence = 0.8;
    station->snrVariance = 0.1;

    // Success tracking
    station->consecSuccess = 0;
    station->consecFailure = 0;
    station->severity = 0.0;
    station->confidence = 1.0;

    // Timing and tracking
    station->T1 = 0;
    station->T2 = 0;
    station->T3 = 0;
    station->lastUpdateTime = Simulator::Now();
    station->lastInferenceTime = Seconds(0);
    station->lastRateChangeTime = Simulator::Now();
    station->retryCount = 0;
    station->mobilityMetric = 0.0;
    station->lastPosition = Vector(0, 0, 0);
    station->currentRateIndex = 3;
    station->previousRateIndex = 3;
    station->queueLength = 0;
    station->rateChangeCount = 0;

    // Context tracking
    station->lastContext = WifiContextType::UNKNOWN;
    station->lastRiskLevel = 0.0;
    station->decisionReason = 0;
    station->lastPacketSuccess = true;

    // Packet tracking
    station->totalPackets = 0;
    station->lostPackets = 0;
    station->totalRetries = 0;
    station->successfulRetries = 0;

    // ENHANCED ML tracking initialization
    station->mlInferencesReceived = 0;
    station->mlInferencesSuccessful = 0;
    station->avgMlConfidence = 0.3; // Start with moderate confidence
    station->preferredModel = m_modelName;

    // NEW: ML learning and adaptation
    station->lastMLInfluencedRate = 3;
    station->lastMLInfluenceTime = Seconds(0);
    station->mlPerformanceScore = 0.5; // Neutral start
    station->mlSuccessfulPredictions = 0;
    station->recentMLAccuracy = 0.5;
    station->lastMLPerformanceUpdate = Simulator::Now();

    // Initialize per-context tracking
    for (int i = 0; i < 6; i++)
    {
        station->mlContextConfidence[i] = 0.3;
        station->mlContextUsage[i] = 0;
    }

    return station;
}

// Configuration methods
void
SmartWifiManagerRf::SetBenchmarkDistance(double distance)
{
    if (distance <= 0.0 || distance > 200.0)
        return;
    m_benchmarkDistance = distance;
    std::cout << "[ML-FIRST CONFIG] Distance updated to " << m_benchmarkDistance << "m"
              << std::endl;
}

void
SmartWifiManagerRf::SetModelName(const std::string& modelName)
{
    m_modelName = modelName;
}

void
SmartWifiManagerRf::SetOracleStrategy(const std::string& strategy)
{
    m_oracleStrategy = strategy;
    m_modelName = strategy;
}

void
SmartWifiManagerRf::SetCurrentInterferers(uint32_t interferers)
{
    m_currentInterferers = interferers;
}

void
SmartWifiManagerRf::UpdateFromBenchmarkGlobals(double distance, uint32_t interferers)
{
    if (std::abs(m_benchmarkDistance - distance) > 0.001)
    {
        m_benchmarkDistance = distance;
    }
    if (m_currentInterferers != interferers)
    {
        m_currentInterferers = interferers;
    }
}

double
SmartWifiManagerRf::ConvertToRealisticSnr(double ns3Snr) const
{
    return ConvertNS3ToRealisticSnr(ns3Snr, m_benchmarkDistance, m_currentInterferers);
}

// ENHANCED: ML-aware context assessment with learning
WifiContextType
SmartWifiManagerRf::ClassifyNetworkContext(SmartWifiManagerRfState* station) const
{
    double snr = station->lastSnr;
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty())
    {
        shortSuccRatio =
            static_cast<double>(
                std::count(station->shortWindow.begin(), station->shortWindow.end(), true)) /
            station->shortWindow.size();
    }

    WifiContextType result;

    if (snr < -20.0 || shortSuccRatio < 0.3 || station->consecFailure >= m_failureThreshold)
    {
        result = WifiContextType::EMERGENCY;
    }
    else if (snr < -10.0 || shortSuccRatio < 0.6)
    {
        result = WifiContextType::POOR_UNSTABLE;
    }
    else if (snr < 5.0 || shortSuccRatio < 0.8)
    {
        result = WifiContextType::MARGINAL;
    }
    else if (snr >= 25.0 && shortSuccRatio > 0.95)
    {
        result = WifiContextType::EXCELLENT_STABLE;
    }
    else if (snr >= 15.0 && shortSuccRatio > 0.85)
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

// REVOLUTIONARY: Adaptive confidence calculation based on conditions and learning
double
SmartWifiManagerRf::CalculateAdaptiveConfidenceThreshold(SmartWifiManagerRfState* station,
                                                         WifiContextType context) const
{
    double baseThreshold = m_confidenceThreshold;
    double adaptiveThreshold = baseThreshold;

    // CONDITION-BASED ADAPTATION
    if (m_benchmarkDistance <= 25.0 && m_currentInterferers <= 1)
    {
        // EXCELLENT conditions - trust ML more
        adaptiveThreshold = std::max(0.15, baseThreshold - 0.08);
    }
    else if (m_benchmarkDistance <= 40.0 && m_currentInterferers <= 2)
    {
        // GOOD conditions - slightly lower threshold
        adaptiveThreshold = std::max(0.18, baseThreshold - 0.04);
    }
    else if (m_benchmarkDistance > 55.0 || m_currentInterferers > 3)
    {
        // HARSH conditions - be more cautious
        adaptiveThreshold = std::min(0.35, baseThreshold + 0.08);
    }

    // LEARNING-BASED ADAPTATION
    if (station->mlInferencesSuccessful > 15)
    {
        double performanceBonus = (station->recentMLAccuracy - 0.5) * 0.1;
        adaptiveThreshold = std::max(0.12, adaptiveThreshold + performanceBonus);
    }

    // CONTEXT-SPECIFIC ADAPTATION
    int contextIdx = static_cast<int>(context);
    if (contextIdx >= 0 && contextIdx < 6 && station->mlContextUsage[contextIdx] > 5)
    {
        double contextConfidence = station->mlContextConfidence[contextIdx];
        if (contextConfidence > 0.4)
        {
            adaptiveThreshold = std::max(0.14, adaptiveThreshold - 0.03);
        }
    }

    return adaptiveThreshold;
}

// INTELLIGENT: ML-First fusion with graduated fallback
uint32_t
SmartWifiManagerRf::FuseMLAndRuleBased(uint32_t mlRate,
                                       uint32_t ruleRate,
                                       double mlConfidence,
                                       const SafetyAssessment& safety) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(
        const_cast<SafetyAssessment&>(safety).station); // Hack for access

    // EMERGENCY: Override everything
    if (safety.requiresEmergencyAction)
    {
        std::cout << "[EMERGENCY] Using safe rate=" << safety.recommendedSafeRate << std::endl;
        return safety.recommendedSafeRate;
    }

    // ADAPTIVE threshold calculation
    double dynamicThreshold = CalculateAdaptiveConfidenceThreshold(station, safety.context);

    // TIER 1: HIGH CONFIDENCE - ML LEADS (60-80% of time in good conditions)
    if (mlConfidence >= dynamicThreshold)
    {
        uint32_t mlPrimary = mlRate;

        // INTELLIGENT BOUNDS: Prevent extreme jumps but allow reasonable ones
        uint32_t maxJump = 2; // Standard max jump
        if (mlConfidence > 0.4)
            maxJump = 3; // Higher confidence = bigger jumps allowed
        if (mlConfidence > 0.5)
            maxJump = 4; // Very high confidence = even bigger jumps

        uint32_t upperBound = std::min(ruleRate + maxJump, static_cast<uint32_t>(7));
        uint32_t lowerBound = (ruleRate > maxJump) ? ruleRate - maxJump : 0;

        // SMART CLAMPING: Less restrictive than before
        if (mlPrimary > upperBound)
        {
            mlPrimary = upperBound;
            std::cout << "[ML-BOUND] Clamped ML rate from " << mlRate << " to " << mlPrimary
                      << std::endl;
        }
        else if (mlPrimary < lowerBound)
        {
            mlPrimary = lowerBound;
            std::cout << "[ML-BOUND] Raised ML rate from " << mlRate << " to " << mlPrimary
                      << std::endl;
        }

        // CONFIDENCE-WEIGHTED FUSION
        double mlWeight = 0.6 + (mlConfidence - dynamicThreshold) * 1.5; // 0.6-0.9 range
        mlWeight = std::min(0.85, mlWeight);                             // Cap at 85%
        double ruleWeight = 1.0 - mlWeight;

        double fusedRate = (mlWeight * mlPrimary) + (ruleWeight * ruleRate);
        uint32_t finalRate = static_cast<uint32_t>(std::round(fusedRate));

        std::cout << "[ML-LEADS] Conf=" << mlConfidence << ">=" << dynamicThreshold
                  << " | MLWeight=" << mlWeight << " | Final=" << finalRate << " [ML-DOMINATED]"
                  << std::endl;

        return finalRate;
    }
    // TIER 2: MEDIUM CONFIDENCE - BALANCED APPROACH (15-25% of time)
    else if (mlConfidence >= dynamicThreshold * 0.65)
    {
        double mlWeight = 0.35 + (mlConfidence / dynamicThreshold) * 0.25; // 0.35-0.6 range
        double ruleWeight = 1.0 - mlWeight;

        double balancedRate = (mlWeight * mlRate) + (ruleWeight * ruleRate);
        uint32_t finalRate = static_cast<uint32_t>(std::round(balancedRate));

        // GENTLE SAFETY: Less restrictive than before
        finalRate = std::min(finalRate, std::max(mlRate, ruleRate) + 1);

        std::cout << "[ML-BALANCED] Conf=" << mlConfidence << " (medium) | MLWeight=" << mlWeight
                  << " | Final=" << finalRate << " [BALANCED]" << std::endl;

        return finalRate;
    }
    // TIER 3: LOW CONFIDENCE - RULE WITH ML HINT (10-15% of time)
    else if (mlConfidence >= dynamicThreshold * 0.4)
    {
        uint32_t ruleWithHint = ruleRate;

        // INTELLIGENT HINT: Use ML to nudge rule-based decision
        if (std::abs(static_cast<int>(mlRate) - static_cast<int>(ruleRate)) <= 2)
        {
            // If ML and rule are close, trust the average
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

        std::cout << "[RULE-WITH-HINT] Conf=" << mlConfidence << " (low-med) | Rule=" << ruleRate
                  << " -> WithHint=" << ruleWithHint << " [RULE-DOMINATED]" << std::endl;

        return ruleWithHint;
    }
    // TIER 4: VERY LOW CONFIDENCE - PURE RULE (5-10% of time)
    else
    {
        std::cout << "[RULE-ONLY] Conf=" << mlConfidence << " (very low) -> Rule=" << ruleRate
                  << " [RULE-ONLY]" << std::endl;
        return ruleRate;
    }
}

// ENHANCED: Feature extraction (same as before - 21 features)
std::vector<double>
SmartWifiManagerRf::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    double shortSuccRatio = 0.5;
    double medSuccRatio = 0.5;
    if (!station->shortWindow.empty())
    {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        shortSuccRatio = static_cast<double>(successes) / station->shortWindow.size();
    }
    if (!station->mediumWindow.empty())
    {
        int successes =
            std::count(station->mediumWindow.begin(), station->mediumWindow.end(), true);
        medSuccRatio = static_cast<double>(successes) / station->mediumWindow.size();
    }

    std::vector<double> features(21);

    // SNR features (7 features)
    features[0] = station->lastSnr;
    features[1] = station->snrFast;
    features[2] = station->snrSlow;
    features[3] = GetSnrTrendShort(st);
    features[4] = GetSnrStabilityIndex(st);
    features[5] = GetSnrPredictionConfidence(st);
    features[6] = std::max(0.0, std::min(100.0, station->snrVariance));

    // Performance features (6 features)
    features[7] = std::max(0.0, std::min(1.0, shortSuccRatio));
    features[8] = std::max(0.0, std::min(1.0, medSuccRatio));
    features[9] = std::min(100.0, static_cast<double>(station->consecSuccess));
    features[10] = std::min(100.0, static_cast<double>(station->consecFailure));
    features[11] = GetPacketLossRate(st);
    features[12] = GetRetrySuccessRatio(st);

    // Rate adaptation features (3 features)
    features[13] = static_cast<double>(GetRecentRateChanges(st));
    features[14] = GetTimeSinceLastRateChange(st);
    features[15] = GetRateStabilityScore(st);

    // Network assessment features (3 features)
    features[16] = std::max(0.0, std::min(1.0, station->severity));
    features[17] = std::max(0.0, std::min(1.0, station->confidence));
    features[18] = station->lastPacketSuccess ? 1.0 : 0.0;

    // Network configuration features (2 features)
    features[19] = static_cast<double>(GetChannelWidth(st));
    features[20] = GetMobilityMetric(st);

    return features;
}

// ML Inference (same as before but with enhanced logging)
SmartWifiManagerRf::InferenceResult
SmartWifiManagerRf::RunMLInference(const std::vector<double>& features) const
{
    NS_LOG_FUNCTION(this);
    InferenceResult result;
    result.success = false;
    result.rateIdx = m_fallbackRate;
    result.latencyMs = 0.0;
    result.confidence = 0.0;
    result.model = m_modelName;

    if (features.size() != 21)
    {
        result.error = "Invalid feature count: expected 21, got " + std::to_string(features.size());
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        result.error = "socket failed";
        return result;
    }

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 150000; // 150ms timeout (reduced)
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(m_inferenceServerPort);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    int conn_ret = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    if (conn_ret < 0)
    {
        close(sockfd);
        result.error = "connect failed to server";
        return result;
    }

    std::ostringstream featStream;
    for (size_t i = 0; i < features.size(); ++i)
    {
        featStream << std::fixed << std::setprecision(6) << features[i];
        if (i + 1 < features.size())
            featStream << " ";
    }
    if (!m_modelName.empty())
    {
        featStream << " " << m_modelName;
    }
    featStream << "\n";
    std::string req = featStream.str();

    ssize_t sent = send(sockfd, req.c_str(), req.size(), 0);
    if (sent != static_cast<ssize_t>(req.size()))
    {
        close(sockfd);
        result.error = "send failed";
        return result;
    }

    std::string response;
    char buffer[4096];
    ssize_t received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);

    close(sockfd);

    if (received <= 0)
    {
        result.error = "no response from server";
        return result;
    }

    buffer[received] = '\0';
    response = std::string(buffer);

    // Parse JSON response
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
            catch (...)
            {
                result.error = "parse error on response";
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.latencyMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    return result;
}

// REVOLUTIONARY: The main rate decision engine - ML-FIRST APPROACH
WifiTxVector
SmartWifiManagerRf::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    uint32_t supportedRates = GetNSupported(st);
    uint32_t maxRateIndex =
        std::min(supportedRates > 0 ? supportedRates - 1 : 0, static_cast<uint32_t>(7));

    // Stage 1: Context and Safety Assessment
    SafetyAssessment safety = AssessNetworkSafety(station);
    safety.station = station; // Hack for fusion access

    // Stage 2: Rule-Based Baseline
    uint32_t ruleRate = GetEnhancedRuleBasedRate(station, safety);

    // Stage 3: AGGRESSIVE ML INFERENCE STRATEGY
    uint32_t mlRate = ruleRate;
    double mlConfidence = 0.0;
    std::string mlStatus = "NO_ATTEMPT";

    static uint64_t s_callCounter = 0;
    ++s_callCounter;

    Time now = Simulator::Now();
    bool canUseCachedMl =
        (now - m_lastMlTime) < MilliSeconds(m_mlCacheTime) && m_lastMlTime > Seconds(0);

    // ADAPTIVE INFERENCE FREQUENCY
    uint32_t adaptiveInferencePeriod = m_inferencePeriod;

    // MORE FREQUENT in good conditions
    if (m_benchmarkDistance <= 30.0 && m_currentInterferers <= 1)
    {
        adaptiveInferencePeriod = std::max(static_cast<uint32_t>(8), m_inferencePeriod / 3);
    }
    // EVEN MORE FREQUENT if ML is performing well
    if (station->recentMLAccuracy > 0.45)
    {
        adaptiveInferencePeriod = std::max(static_cast<uint32_t>(12), adaptiveInferencePeriod / 2);
    }

    bool needNewMlInference = !safety.requiresEmergencyAction &&
                              safety.riskLevel < m_riskThreshold && !canUseCachedMl &&
                              (s_callCounter % adaptiveInferencePeriod) == 0;

    // FORCE MORE ML USAGE: Also try ML if conditions changed significantly
    if (!needNewMlInference && !canUseCachedMl && !safety.requiresEmergencyAction)
    {
        // Force ML if we haven't used it recently and conditions aren't terrible
        Time timeSinceLastML = now - station->lastMLInfluenceTime;
        if (timeSinceLastML > MilliSeconds(500) && safety.riskLevel < 0.6)
        {
            needNewMlInference = true;
            std::cout << "[FORCE ML] Haven't used ML recently, forcing inference" << std::endl;
        }
    }

    if (canUseCachedMl)
    {
        mlRate = m_lastMlRate;
        mlConfidence = m_lastMlConfidence;
        mlStatus = "CACHED";
        m_mlCacheHits++;
    }
    else if (needNewMlInference)
    {
        mlStatus = "ATTEMPTING";
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);

        if (result.success && result.confidence > 0.1) // Very low minimum threshold
        {
            m_mlInferences++;
            station->mlInferencesReceived++;
            station->mlInferencesSuccessful++;

            mlRate = std::min(result.rateIdx, maxRateIndex);
            mlConfidence = result.confidence;

            // Update cache
            m_lastMlRate = mlRate;
            m_lastMlTime = now;
            m_lastMlConfidence = mlConfidence;

            mlStatus = "SUCCESS";

            // LEARNING: Update performance tracking
            station->recentMLAccuracy = 0.9 * station->recentMLAccuracy + 0.1 * mlConfidence;
            int contextIdx = static_cast<int>(safety.context);
            if (contextIdx >= 0 && contextIdx < 6)
            {
                station->mlContextConfidence[contextIdx] =
                    0.8 * station->mlContextConfidence[contextIdx] + 0.2 * mlConfidence;
                station->mlContextUsage[contextIdx]++;
            }

            std::cout << "[ML-FIRST SUCCESS] Model=" << result.model << " Rate=" << result.rateIdx
                      << " Conf=" << mlConfidence << " SNR=" << station->lastSnr << "dB"
                      << std::endl;
        }
        else
        {
            m_mlFailures++;
            mlRate = ruleRate;
            mlConfidence = 0.0;
            mlStatus = "FAILED";
            std::cout << "[ML FAILED] Using rule-based fallback: " << ruleRate << std::endl;
        }
    }

    // Stage 4: INTELLIGENT FUSION (The Magic Happens Here)
    uint32_t finalRate = FuseMLAndRuleBased(mlRate, ruleRate, mlConfidence, safety);

    // Stage 5: Final Safety and Tracking
    finalRate = std::min(finalRate, maxRateIndex);
    finalRate = std::max(finalRate, static_cast<uint32_t>(0));

    // ENHANCED ANTI-THRASHING with ML awareness
    if (finalRate != station->currentRateIndex)
    {
        // Track ML influence
        bool wasMLInfluenced =
            (mlConfidence >= CalculateAdaptiveConfidenceThreshold(station, safety.context));
        if (wasMLInfluenced)
        {
            station->lastMLInfluencedRate = finalRate;
            station->lastMLInfluenceTime = now;
        }

        // SMART anti-thrashing
        if (station->rateChangeCount > 8 && mlConfidence < 0.2)
        {
            // Too many changes with low confidence = thrashing
            uint32_t dampedRate = (station->currentRateIndex + finalRate) / 2;
            std::cout << "[ANTI-THRASH] " << station->rateChangeCount
                      << " changes, low ML conf -> Dampen " << finalRate << " to " << dampedRate
                      << std::endl;
            finalRate = dampedRate;
        }

        // Update tracking
        station->previousRateIndex = station->currentRateIndex;
        station->currentRateIndex = finalRate;
        station->lastRateChangeTime = now;
        station->rateChangeCount++;

        station->rateHistory.push_back(finalRate);
        if (station->rateHistory.size() > 20)
        {
            station->rateHistory.pop_front();
        }
    }

    // COMPREHENSIVE LOGGING
    std::string fusionType =
        (mlConfidence >= CalculateAdaptiveConfidenceThreshold(station, safety.context))
            ? "ML-LED"
            : "RULE-LED";
    std::cout << "[ML-FIRST DECISION] Call#" << s_callCounter << " | " << fusionType
              << " | SNR=" << station->lastSnr << "dB | Context=" << safety.contextStr
              << " | Rule=" << ruleRate << " | ML=" << mlRate << "(conf=" << mlConfidence << ")"
              << " | Final=" << finalRate << " | Status=" << mlStatus << std::endl;

    WifiMode mode = GetSupported(st, finalRate);
    uint64_t rate = mode.GetDataRate(allowedWidth);

    if (m_currentRate != rate)
    {
        std::cout << "[RATE CHANGE] " << m_currentRate << " -> " << rate << " (index " << finalRate
                  << ") | Strategy: " << m_oracleStrategy << std::endl;
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

// Enhanced rule-based rate with ML awareness
uint32_t
SmartWifiManagerRf::GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station,
                                             const SafetyAssessment& safety) const
{
    double snr = station->lastSnr;
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty())
    {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        shortSuccRatio = static_cast<double>(successes) / station->shortWindow.size();
    }

    uint32_t baseRate;
    if (snr >= 30)
        baseRate = 7;
    else if (snr >= 20)
        baseRate = 6;
    else if (snr >= 15)
        baseRate = 5;
    else if (snr >= 10)
        baseRate = 4;
    else if (snr >= 5)
        baseRate = 3;
    else if (snr >= 0)
        baseRate = 2;
    else if (snr >= -10)
        baseRate = 1;
    else
        baseRate = 0;

    uint32_t adjustedRate = baseRate;

    // Success-based adjustments (less conservative than before)
    if (shortSuccRatio > 0.9 && station->consecSuccess > 50)
    {
        adjustedRate = std::min(baseRate + 1, static_cast<uint32_t>(7));
    }
    else if (shortSuccRatio < 0.6 || station->consecFailure > 3)
    {
        adjustedRate = (baseRate > 0) ? baseRate - 1 : 0;
    }

    // ML-AWARE ADJUSTMENT: If ML has been performing well, be slightly more optimistic
    if (station->recentMLAccuracy > 0.4 && station->mlInferencesSuccessful > 8)
    {
        if (shortSuccRatio > 0.8 && adjustedRate < 6)
        {
            adjustedRate = std::min(adjustedRate + 1, static_cast<uint32_t>(6));
            std::cout << "[ML-AWARE RULE] Good ML track -> Optimistic rule boost to "
                      << adjustedRate << std::endl;
        }
    }

    return adjustedRate;
}

// Safety assessment (enhanced but not overly conservative)
SmartWifiManagerRf::SafetyAssessment
SmartWifiManagerRf::AssessNetworkSafety(SmartWifiManagerRfState* station)
{
    SafetyAssessment assessment;
    assessment.context = ClassifyNetworkContext(station);
    assessment.riskLevel = CalculateRiskLevel(station);
    assessment.recommendedSafeRate = GetContextSafeRate(station, assessment.context);

    // Less trigger-happy emergency conditions
    assessment.requiresEmergencyAction =
        (assessment.context == WifiContextType::EMERGENCY ||
         station->consecFailure >= m_failureThreshold || assessment.riskLevel > m_riskThreshold);

    assessment.confidenceInAssessment = 1.0 - assessment.riskLevel;
    assessment.contextStr = ContextTypeToString(assessment.context);
    station->lastContext = assessment.context;
    station->lastRiskLevel = assessment.riskLevel;
    return assessment;
}

double
SmartWifiManagerRf::CalculateRiskLevel(SmartWifiManagerRfState* station) const
{
    double risk = 0.0;
    risk += (station->consecFailure >= m_failureThreshold) ? 0.4 : 0.0; // Reduced
    risk += (station->snrVariance > 8.0) ? 0.2 : 0.0;                   // Less sensitive
    risk += (station->lastSnr < 0.0) ? 0.3 : 0.0;                       // Only very poor SNR
    risk += std::max(0.0, 1.0 - station->confidence) * 0.3;             // Reduced impact
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

// SNR reporting methods (same as before but with better logging)
void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lastRawSnr = rxSnr;
    double realisticSnr =
        ConvertNS3ToRealisticSnr(rxSnr, m_benchmarkDistance, m_currentInterferers);
    station->lastSnr = realisticSnr;

    station->snrHistory.push_back(realisticSnr);
    station->rawSnrHistory.push_back(rxSnr);
    if (station->snrHistory.size() > 20)
    {
        station->snrHistory.pop_front();
        station->rawSnrHistory.pop_front();
    }

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

    UpdateMetrics(st, true, realisticSnr);
}

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
    double realisticDataSnr =
        ConvertNS3ToRealisticSnr(dataSnr, m_benchmarkDistance, m_currentInterferers);
    station->lastSnr = realisticDataSnr;

    station->retryCount = 0;
    station->totalPackets++;
    station->successfulRetries++;

    UpdateMetrics(st, true, realisticDataSnr);
}

void
SmartWifiManagerRf::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->retryCount++;
    station->totalRetries++;
    station->lostPackets++;
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerRf::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    UpdateMetrics(st, false, static_cast<SmartWifiManagerRfState*>(st)->lastSnr);
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
}

void
SmartWifiManagerRf::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->lostPackets++;
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerRf::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->lostPackets++;
    UpdateMetrics(st, false, station->lastSnr);
}

// Enhanced metrics update
void
SmartWifiManagerRf::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();

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

    if (success)
    {
        station->consecSuccess++;
        station->consecFailure = 0;
        station->lastPacketSuccess = true;
        station->shortWindow.push_back(true);
        station->mediumWindow.push_back(true);
    }
    else
    {
        station->consecFailure++;
        station->consecSuccess = 0;
        station->lastPacketSuccess = false;
        station->shortWindow.push_back(false);
        station->mediumWindow.push_back(false);
    }

    if (station->shortWindow.size() > m_windowSize)
    {
        station->shortWindow.pop_front();
    }
    if (station->mediumWindow.size() > (m_windowSize * 2))
    {
        station->mediumWindow.pop_front();
    }

    double recentSuccessRate = 0.5;
    if (!station->shortWindow.empty())
    {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        recentSuccessRate = static_cast<double>(successes) / station->shortWindow.size();
    }

    station->confidence = 0.8 * station->confidence + 0.2 * recentSuccessRate;
    station->severity = 1.0 - station->confidence;
    station->lastUpdateTime = now;
}

// Helper functions (same as before)
double
SmartWifiManagerRf::GetRetrySuccessRatio(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    if (station->totalRetries == 0)
        return 0.5;
    return static_cast<double>(station->successfulRetries) / station->totalRetries;
}

uint32_t
SmartWifiManagerRf::GetRecentRateChanges(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    return std::min(station->rateChangeCount, static_cast<uint32_t>(20));
}

double
SmartWifiManagerRf::GetTimeSinceLastRateChange(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();
    double timeDiff = (now - station->lastRateChangeTime).GetMilliSeconds();
    return std::max(0.0, std::min(10000.0, timeDiff));
}

double
SmartWifiManagerRf::GetRateStabilityScore(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    if (station->rateChangeCount == 0)
        return 1.0;
    return std::max(0.0, 1.0 - (station->rateChangeCount / 20.0));
}

double
SmartWifiManagerRf::GetPacketLossRate(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    if (station->totalPackets == 0)
        return 0.0;
    return std::max(
        0.0,
        std::min(1.0, static_cast<double>(station->lostPackets) / station->totalPackets));
}

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
    double successFactor = station->confidence;
    return std::max(0.0, std::min(1.0, (stabilityFactor + successFactor) / 2.0));
}

double
SmartWifiManagerRf::GetOfferedLoad() const
{
    return 10.0;
}

double
SmartWifiManagerRf::GetMobilityMetric(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    double snrMobility = std::tanh(station->snrVariance / 10.0);
    station->mobilityMetric = snrMobility;
    return std::max(0.0, std::min(1.0, station->mobilityMetric));
}

void
SmartWifiManagerRf::DebugPrintCurrentConfig() const
{
    std::cout << "[ML-FIRST CONFIG] Distance: " << m_benchmarkDistance << "m" << std::endl;
    std::cout << "[ML-FIRST CONFIG] Interferers: " << m_currentInterferers << std::endl;
    std::cout << "[ML-FIRST CONFIG] Strategy: " << m_oracleStrategy << std::endl;
    std::cout << "[ML-FIRST CONFIG] Confidence Threshold: " << m_confidenceThreshold
              << " (adaptive)" << std::endl;
    std::cout << "[ML-FIRST CONFIG] ML Weight: " << m_mlGuidanceWeight << std::endl;
}

} // namespace ns3