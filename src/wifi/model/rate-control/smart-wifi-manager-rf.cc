/*
 * Enhanced Smart WiFi Manager Implementation - FIXED SNR ENGINE
 * Compatible with ahmedjk34's Enhanced ML Pipeline (98.1% CV accuracy)
 *
 * FIXED: Complete SNR calculation engine with consistent realistic conversion
 * FIXED: Unified SNR processing pipeline
 * FIXED: Proper distance-based SNR modeling
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-09-24
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

// FIXED: Global realistic SNR conversion function - MOVED TO GLOBAL SCOPE
double
ConvertNS3ToRealisticSnr(double ns3Value, double distance, uint32_t interferers)
{
    double realisticSnr;

    // Distance-based realistic SNR calculation
    if (distance <= 10.0)
    {
        realisticSnr = 35.0 - (distance * 1.5); // Close: 35dB to 20dB
    }
    else if (distance <= 30.0)
    {
        realisticSnr = 20.0 - ((distance - 10.0) * 1.0); // Medium: 20dB to 0dB
    }
    else if (distance <= 50.0)
    {
        realisticSnr = 0.0 - ((distance - 30.0) * 0.75); // Far: 0dB to -15dB
    }
    else
    {
        realisticSnr = -15.0 - ((distance - 50.0) * 0.5); // Very far: -15dB to -25dB
    }

    // Add interference degradation
    realisticSnr -= (interferers * 3.0);

    // Add realistic variation based on NS-3 input
    double variation = fmod(ns3Value, 20.0) - 10.0; // ±10dB variation
    realisticSnr += variation * 0.3;                // Scale down the variation

    // Bound to realistic WiFi SNR range
    realisticSnr = std::max(-30.0, std::min(45.0, realisticSnr));

    return realisticSnr;
}

// Enhanced TypeId with updated defaults for 28-feature pipeline
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
                          "Oracle strategy (oracle_balanced, oracle_conservative, "
                          "oracle_aggressive, rateIdx)",
                          StringValue("oracle_balanced"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_oracleStrategy),
                          MakeStringChecker())
            .AddAttribute("InferenceServerPort",
                          "Port number of Enhanced Python ML inference server",
                          UintegerValue(8765),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferenceServerPort),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("UseRealisticSnr",
                          "Use enhanced realistic SNR calculation with proper bounds",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_useRealisticSnr),
                          MakeBooleanChecker())
            .AddAttribute("MaxSnrDb",
                          "Maximum realistic SNR in dB",
                          DoubleValue(45.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_maxSnrDb),
                          MakeDoubleChecker<double>())
            .AddAttribute("MinSnrDb",
                          "Minimum realistic SNR in dB",
                          DoubleValue(-30.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_minSnrDb),
                          MakeDoubleChecker<double>())
            .AddAttribute("SnrOffset",
                          "SNR offset to apply to ns-3 values (dB)",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_snrOffset),
                          MakeDoubleChecker<double>())
            .AddAttribute("WindowSize",
                          "Window size for success ratio calculation",
                          UintegerValue(20),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_windowSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SnrAlpha",
                          "Alpha parameter for SNR exponential smoothing",
                          DoubleValue(0.1),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_snrAlpha),
                          MakeDoubleChecker<double>())
            .AddAttribute("InferencePeriod",
                          "Period between ML inferences (in transmissions)",
                          UintegerValue(50),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferencePeriod),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("FallbackRate",
                          "Fallback rate index on ML failure",
                          UintegerValue(3),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_fallbackRate),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EnableFallback",
                          "Enable enhanced fallback mechanism on ML failure",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableFallback),
                          MakeBooleanChecker())
            .AddAttribute("ConfidenceThreshold",
                          "Minimum ML confidence required to trust prediction",
                          DoubleValue(0.4),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_confidenceThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("RiskThreshold",
                          "Maximum risk allowed before forcing conservative rate",
                          DoubleValue(0.6),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_riskThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("FailureThreshold",
                          "Consecutive failures required to trigger emergency",
                          UintegerValue(3),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_failureThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MLGuidanceWeight",
                          "Weight of ML guidance in final decision (0.0-1.0)",
                          DoubleValue(0.7),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_mlGuidanceWeight),
                          MakeDoubleChecker<double>())
            .AddAttribute("MLCacheTime",
                          "Time to cache ML results in milliseconds",
                          UintegerValue(250),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_mlCacheTime),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EnableAdaptiveWeighting",
                          "Enable adaptive ML weighting based on confidence",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableAdaptiveWeighting),
                          MakeBooleanChecker())
            .AddAttribute("ConservativeBoost",
                          "Conservative rate boost factor for safety",
                          DoubleValue(1.2),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_conservativeBoost),
                          MakeDoubleChecker<double>())
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
      m_mlGuidanceWeight(0.7),
      m_enableAdaptiveWeighting(true),
      m_conservativeBoost(1.2),
      m_snrAlpha(0.1),
      m_enableDetailedLogging(true)
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

    std::cout << "[FIXED SNR ENGINE] SmartWifiManagerRf v3.0 initialized successfully" << std::endl;
    std::cout << "[FIXED SNR ENGINE] Model: " << m_modelName << std::endl;
    std::cout << "[FIXED SNR ENGINE] Oracle Strategy: " << m_oracleStrategy << std::endl;
    std::cout << "[FIXED SNR ENGINE] Realistic SNR enabled: " << m_useRealisticSnr << std::endl;
    std::cout << "[FIXED SNR ENGINE] SNR range: [" << m_minSnrDb << ", " << m_maxSnrDb << "] dB"
              << std::endl;
    std::cout << "[FIXED SNR ENGINE] Distance: " << m_benchmarkDistance
              << "m, Interferers: " << m_currentInterferers << std::endl;

    WifiRemoteStationManager::DoInitialize();
}

// FIXED: Enhanced station creation with consistent SNR initialization
WifiRemoteStation*
SmartWifiManagerRf::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    SmartWifiManagerRfState* station = new SmartWifiManagerRfState;

    // FIXED: Initialize with realistic SNR based on current distance
    double initialRealisticSnr =
        ConvertNS3ToRealisticSnr(100.0, m_benchmarkDistance, m_currentInterferers);

    // Core SNR metrics - FIXED to use realistic values consistently
    station->lastSnr = initialRealisticSnr; // REALISTIC SNR
    station->lastRawSnr = 0.0;              // Raw NS-3 SNR (will be updated)
    station->snrFast = initialRealisticSnr; // REALISTIC SNR
    station->snrSlow = initialRealisticSnr; // REALISTIC SNR
    station->snrTrendShort = 0.0;
    station->snrStabilityIndex = 1.0;
    station->snrPredictionConfidence = 0.8;
    station->snrVariance = 0.1;

    // Success tracking
    station->consecSuccess = 0;
    station->consecFailure = 0;

    // Network condition assessment
    station->severity = 0.0;
    station->confidence = 1.0;

    // Timing features
    station->T1 = 0;
    station->T2 = 0;
    station->T3 = 0;
    station->lastUpdateTime = Simulator::Now();
    station->lastInferenceTime = Seconds(0);
    station->lastRateChangeTime = Simulator::Now();

    // Enhanced tracking
    station->retryCount = 0;
    station->mobilityMetric = 0.0;
    station->lastPosition = Vector(0, 0, 0);
    station->currentRateIndex = std::min(m_fallbackRate, static_cast<uint32_t>(7));
    station->previousRateIndex = station->currentRateIndex;
    station->queueLength = 0;
    station->rateChangeCount = 0;

    // Context tracking
    station->lastContext = WifiContextType::UNKNOWN;
    station->lastRiskLevel = 0.0;
    station->decisionReason = 0;
    station->lastPacketSuccess = true;

    // Enhanced packet tracking
    station->totalPackets = 0;
    station->lostPackets = 0;
    station->totalRetries = 0;
    station->successfulRetries = 0;

    // ML interaction tracking
    station->mlInferencesReceived = 0;
    station->mlInferencesSuccessful = 0;
    station->avgMlConfidence = 0.0;
    station->preferredModel = m_modelName;

    std::cout << "[FIXED STATION CREATION] Initial realistic SNR: " << initialRealisticSnr
              << "dB for distance: " << m_benchmarkDistance
              << "m, interferers: " << m_currentInterferers << std::endl;

    return station;
}

// FIXED: Configuration methods with proper synchronization
void
SmartWifiManagerRf::SetBenchmarkDistance(double distance)
{
    if (distance <= 0.0 || distance > 200.0)
    {
        std::cout << "[ERROR] Invalid distance: " << distance
                  << "m. Keeping: " << m_benchmarkDistance << "m" << std::endl;
        return;
    }

    double oldDistance = m_benchmarkDistance;
    m_benchmarkDistance = distance;

    std::cout << "[FIXED DISTANCE SET] Updated from " << oldDistance << "m to "
              << m_benchmarkDistance << "m" << std::endl;
    NS_LOG_FUNCTION(this << distance);
}

void
SmartWifiManagerRf::SetModelName(const std::string& modelName)
{
    m_modelName = modelName;
    std::cout << "[INFO] Model name set to: " << m_modelName << std::endl;
}

void
SmartWifiManagerRf::SetOracleStrategy(const std::string& strategy)
{
    m_oracleStrategy = strategy;
    m_modelName = strategy; // Sync model name with strategy
    std::cout << "[INFO] Oracle strategy set to: " << m_oracleStrategy << std::endl;
}

void
SmartWifiManagerRf::SetCurrentInterferers(uint32_t interferers)
{
    m_currentInterferers = interferers;
    std::cout << "[INFO] Current interferers set to: " << m_currentInterferers << std::endl;
}

// FIXED: New synchronization method
void
SmartWifiManagerRf::UpdateFromBenchmarkGlobals(double distance, uint32_t interferers)
{
    if (std::abs(m_benchmarkDistance - distance) > 0.001)
    {
        std::cout << "[SYNC] Updating manager distance: " << m_benchmarkDistance << "m -> "
                  << distance << "m" << std::endl;
        m_benchmarkDistance = distance;
    }
    if (m_currentInterferers != interferers)
    {
        std::cout << "[SYNC] Updating manager interferers: " << m_currentInterferers << " -> "
                  << interferers << std::endl;
        m_currentInterferers = interferers;
    }
}

// FIXED: Realistic SNR conversion method
double
SmartWifiManagerRf::ConvertToRealisticSnr(double ns3Snr) const
{
    return ConvertNS3ToRealisticSnr(ns3Snr, m_benchmarkDistance, m_currentInterferers);
}

// FIXED: Distance-based SNR calculation
double
SmartWifiManagerRf::CalculateDistanceBasedSnr(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // FIXED: Use realistic conversion with current distance
    if (station->lastRawSnr != 0.0)
    {
        double realisticSnr = ConvertToRealisticSnr(station->lastRawSnr);
        std::cout << "[FIXED DISTANCE SNR] Raw=" << station->lastRawSnr
                  << "dB -> Realistic=" << realisticSnr
                  << "dB using distance=" << m_benchmarkDistance << "m" << std::endl;
        return realisticSnr;
    }

    // Fallback: Pure distance-based calculation
    double fallbackSnr =
        (m_benchmarkDistance <= 10.0)   ? 35.0 - (m_benchmarkDistance * 1.5)
        : (m_benchmarkDistance <= 30.0) ? 20.0 - ((m_benchmarkDistance - 10.0) * 1.0)
        : (m_benchmarkDistance <= 50.0) ? 0.0 - ((m_benchmarkDistance - 30.0) * 0.75)
                                        : -15.0 - ((m_benchmarkDistance - 50.0) * 0.5);

    fallbackSnr -= (m_currentInterferers * 3.0);
    fallbackSnr = std::max(-30.0, std::min(45.0, fallbackSnr));

    std::cout << "[FALLBACK DISTANCE SNR] Using distance=" << m_benchmarkDistance
              << "m, intf=" << m_currentInterferers << " -> SNR=" << fallbackSnr << "dB"
              << std::endl;

    return fallbackSnr;
}

double
SmartWifiManagerRf::ApplyRealisticSnrBounds(double snr) const
{
    return std::max(m_minSnrDb, std::min(m_maxSnrDb, snr + m_snrOffset));
}

// FIXED: Enhanced feature extraction for 28 safe features
std::vector<double>
SmartWifiManagerRf::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // Calculate success ratios
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

    // FIXED: 28 safe features using REALISTIC SNR consistently
    std::vector<double> features(28);

    features[0] = station->lastSnr;                             // REALISTIC SNR
    features[1] = station->snrFast;                             // REALISTIC SNR Fast
    features[2] = station->snrSlow;                             // REALISTIC SNR Slow
    features[3] = GetSnrTrendShort(st);                         // SNR trend
    features[4] = GetSnrStabilityIndex(st);                     // SNR stability
    features[5] = GetSnrPredictionConfidence(st);               // SNR prediction confidence
    features[6] = std::max(0.0, std::min(1.0, shortSuccRatio)); // Short success ratio
    features[7] = std::max(0.0, std::min(1.0, medSuccRatio));   // Medium success ratio
    features[8] =
        std::min(100.0, static_cast<double>(station->consecSuccess)); // Consecutive successes
    features[9] =
        std::min(100.0, static_cast<double>(station->consecFailure)); // Consecutive failures
    features[10] = GetPacketLossRate(st);                             // Packet loss rate
    features[11] = GetRetrySuccessRatio(st);                          // Retry success ratio
    features[12] = static_cast<double>(GetRecentRateChanges(st));     // Recent rate changes
    features[13] = GetTimeSinceLastRateChange(st);                    // Time since last rate change
    features[14] = GetRateStabilityScore(st);                         // Rate stability score
    features[15] = std::max(0.0, std::min(1.0, station->severity));   // Severity
    features[16] = std::max(0.0, std::min(1.0, station->confidence)); // Confidence
    features[17] = static_cast<double>(station->T1);                  // T1
    features[18] = static_cast<double>(station->T2);                  // T2
    features[19] = static_cast<double>(station->T3);                  // T3
    features[20] = static_cast<double>(station->decisionReason);      // Decision reason
    features[21] = station->lastPacketSuccess ? 1.0 : 0.0;            // Packet success
    features[22] = GetOfferedLoad();                                  // Offered load
    features[23] = static_cast<double>(station->queueLength);         // Queue length
    features[24] = static_cast<double>(station->retryCount);          // Retry count
    features[25] = static_cast<double>(GetChannelWidth(st));          // Channel width
    features[26] = GetMobilityMetric(st);                             // Mobility metric
    features[27] = std::max(0.0, std::min(100.0, station->snrVariance)); // SNR variance

    std::cout << "[FIXED FEATURES] 28 Features with REALISTIC SNR: lastSnr=" << features[0]
              << "dB snrFast=" << features[1] << "dB snrSlow=" << features[2] << "dB" << std::endl;

    return features;
}

// FIXED: Enhanced ML inference with proper validation
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

    // FIXED: Validate 28 features
    if (features.size() != 28)
    {
        result.error = "Invalid feature count: expected 28, got " + std::to_string(features.size());
        std::cout << "[ERROR FIXED ML] " << result.error << std::endl;
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
    timeout.tv_usec = 200000; // 200ms timeout
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

    // Send request with 28 features + model name
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

    // Receive response
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

                // Get confidence
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

// Helper functions for 28 features (unchanged, but verified to work with realistic SNR)
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
        recent += station->snrHistory[i]; // Using REALISTIC SNR history
    }
    for (size_t i = 0; i < halfSize; ++i)
    {
        older += station->snrHistory[i]; // Using REALISTIC SNR history
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
    return 10.0; // Static for simulation
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
SmartWifiManagerRf::LogFeatureVector(const std::vector<double>& features,
                                     const std::string& context) const
{
    if (features.size() != 28)
    {
        std::cout << "[ERROR FIXED LOG] " << context << ": Expected 28 features, got "
                  << features.size() << std::endl;
        return;
    }

    std::cout << "[FIXED DEBUG " << context << "] 28 Features with REALISTIC SNR: ";
    for (size_t i = 0; i < features.size(); ++i)
    {
        std::cout << "[" << i << "]=" << std::setprecision(4) << features[i] << " ";
        if ((i + 1) % 7 == 0)
            std::cout << std::endl << "    ";
    }
    std::cout << std::endl;
}

// FIXED: SNR reporting methods with consistent realistic conversion
void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // FIXED: Store raw NS-3 SNR first
    station->lastRawSnr = rxSnr;

    // FIXED: Convert to realistic SNR using current manager settings
    double realisticSnr = ConvertToRealisticSnr(rxSnr);
    station->lastSnr = realisticSnr;

    std::cout << "[FIXED RX SNR] Raw NS-3=" << rxSnr << "dB -> Realistic=" << realisticSnr
              << "dB (dist=" << m_benchmarkDistance << "m, intf=" << m_currentInterferers << ")"
              << std::endl;

    // FIXED: Update both histories consistently
    station->snrHistory.push_back(realisticSnr);
    station->rawSnrHistory.push_back(rxSnr);
    if (station->snrHistory.size() > 20)
    {
        station->snrHistory.pop_front();
        station->rawSnrHistory.pop_front();
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

    // FIXED: Store raw values first
    station->lastRawSnr = dataSnr;

    // FIXED: Convert using current manager settings
    double realisticDataSnr = ConvertToRealisticSnr(dataSnr);
    station->lastSnr = realisticDataSnr;

    // FIXED: Reset retry count and update counters
    station->retryCount = 0;
    station->totalPackets++;
    station->successfulRetries++;

    std::cout << "[FIXED DATA OK] Raw NS-3=" << dataSnr << "dB -> Realistic=" << realisticDataSnr
              << "dB (dist=" << m_benchmarkDistance << "m)" << std::endl;

    UpdateMetrics(st, true, realisticDataSnr);
}

void
SmartWifiManagerRf::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    UpdateMetrics(st, false, station->lastSnr); // Use realistic SNR
}

void
SmartWifiManagerRf::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->retryCount++;
    station->totalRetries++;
    station->lostPackets++;
    UpdateMetrics(st, false, station->lastSnr); // Use realistic SNR
}

void
SmartWifiManagerRf::DoReportRtsOk(WifiRemoteStation* st,
                                  double ctsSnr,
                                  WifiMode ctsMode,
                                  double rtsSnr)
{
    NS_LOG_FUNCTION(this << st << ctsSnr << ctsMode << rtsSnr);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // FIXED: Convert RTS SNR to realistic
    station->lastRawSnr = rtsSnr;
    double realisticRtsSnr = ConvertToRealisticSnr(rtsSnr);
    station->lastSnr = realisticRtsSnr;

    std::cout << "[FIXED RTS OK] Raw NS-3=" << rtsSnr << "dB -> Realistic=" << realisticRtsSnr
              << "dB (dist=" << m_benchmarkDistance << "m)" << std::endl;
}

void
SmartWifiManagerRf::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->lostPackets++;
    UpdateMetrics(st, false, station->lastSnr); // Use realistic SNR
}

void
SmartWifiManagerRf::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->lostPackets++;
    UpdateMetrics(st, false, station->lastSnr); // Use realistic SNR
}

// FIXED: Main rate decision method with consistent realistic SNR usage
WifiTxVector
SmartWifiManagerRf::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    uint32_t maxRateIndex = GetNSupported(st) - 1;
    maxRateIndex = std::min(maxRateIndex, static_cast<uint32_t>(7));

    // Stage 1: Safety/Context Assessment using REALISTIC SNR
    SafetyAssessment safety = AssessNetworkSafety(station);

    // Stage 2: Rule-Based Decision using REALISTIC SNR
    uint32_t primaryRate = GetEnhancedRuleBasedRate(station, safety);

    // Stage 3: ML Guidance
    uint32_t mlGuidance = primaryRate;
    double mlConfidence = 0.0;
    std::string mlStatus = "NO_ATTEMPT";

    static uint64_t s_callCounter = 0;
    ++s_callCounter;

    Time now = Simulator::Now();
    bool canUseCachedMl =
        (now - m_lastMlTime) < MilliSeconds(m_mlCacheTime) && m_lastMlTime > Seconds(0);
    bool needNewMlInference = !safety.requiresEmergencyAction &&
                              safety.riskLevel < m_riskThreshold && !canUseCachedMl &&
                              (s_callCounter % m_inferencePeriod) == 0;

    if (canUseCachedMl)
    {
        mlGuidance = m_lastMlRate;
        mlConfidence = m_lastMlConfidence;
        mlStatus = "CACHED";
        m_mlCacheHits++;
    }
    else if (needNewMlInference)
    {
        mlStatus = "ATTEMPTING";

        std::vector<double> features = ExtractFeatures(st); // Uses REALISTIC SNR
        InferenceResult result = RunMLInference(features);

        if (result.success)
        {
            m_mlInferences++;
            station->mlInferencesReceived++;
            station->mlInferencesSuccessful++;

            mlGuidance = std::min(result.rateIdx, maxRateIndex);
            mlConfidence = result.confidence;

            // Update cache
            m_lastMlRate = mlGuidance;
            m_lastMlTime = now;
            m_lastMlConfidence = mlConfidence;
            m_lastMlModel = result.model;

            mlStatus = "SUCCESS";

            std::cout << "[FIXED ML SUCCESS] Model: " << result.model
                      << " Prediction: " << result.rateIdx << " -> " << mlGuidance
                      << " Confidence: " << mlConfidence
                      << " using REALISTIC SNR=" << station->lastSnr << "dB" << std::endl;
        }
        else
        {
            m_mlFailures++;
            mlGuidance = primaryRate;
            mlConfidence = 0.0;
            mlStatus = "FAILED";
        }
    }
    else
    {
        mlStatus = "SKIPPED";
    }

    // Stage 4: Intelligent Fusion
    uint32_t finalRate = FuseMLAndRuleBased(mlGuidance, primaryRate, mlConfidence, safety);

    // Stage 5: Final bounds and tracking
    finalRate = std::min(finalRate, maxRateIndex);
    finalRate = std::max(finalRate, static_cast<uint32_t>(0));

    // Track rate changes
    if (finalRate != station->currentRateIndex)
    {
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

    std::cout << "[FIXED RATE DECISION] Call#" << s_callCounter
              << " | Realistic SNR=" << station->lastSnr << "dB"
              << " | Context=" << safety.contextStr << " | Risk=" << safety.riskLevel
              << " | RuleRate=" << primaryRate << " | MLRate=" << mlGuidance
              << "(conf=" << mlConfidence << ")"
              << " | FinalRate=" << finalRate << " | Status=" << mlStatus << std::endl;

    WifiMode mode = GetSupported(st, finalRate);
    uint64_t rate = mode.GetDataRate(allowedWidth);

    if (m_currentRate != rate)
    {
        std::cout << "[FIXED RATE CHANGE] " << m_currentRate << " -> " << rate << " (index "
                  << finalRate << ") | Strategy: " << m_oracleStrategy << std::endl;
        m_currentRate = rate;
    }
    // CONTINUATION OF smart-wifi-manager-rf.cc

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

// FIXED: Enhanced rule-based rate selection using REALISTIC SNR
uint32_t
SmartWifiManagerRf::GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station,
                                             const SafetyAssessment& safety) const
{
    // FIXED: Use REALISTIC SNR, not raw NS-3
    double snr = station->lastSnr; // This contains realistic SNR (-30 to +45 dB)
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty())
    {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        shortSuccRatio = static_cast<double>(successes) / station->shortWindow.size();
    }

    // FIXED: Use realistic SNR thresholds for WiFi
    uint32_t baseRate;
    if (snr >= 30)
        baseRate = 7; // Excellent signal (realistic)
    else if (snr >= 20)
        baseRate = 6; // Very good signal
    else if (snr >= 15)
        baseRate = 5; // Good signal
    else if (snr >= 10)
        baseRate = 4; // Moderate signal
    else if (snr >= 5)
        baseRate = 3; // Acceptable signal
    else if (snr >= 0)
        baseRate = 2; // Weak signal
    else if (snr >= -10)
        baseRate = 1; // Very weak signal
    else
        baseRate = 0; // Poor signal

    // FIXED: Apply success-based adjustments
    uint32_t adjustedRate = baseRate;
    std::string ruleReason = "REALISTIC_SNR_BASED";

    if (shortSuccRatio > 0.95 && station->consecSuccess > 100)
    {
        adjustedRate = std::min(baseRate + 2, static_cast<uint32_t>(7)); // Reduced boost
        ruleReason = "EXCELLENT_SUCCESS_BOOST";
    }
    else if (shortSuccRatio > 0.90 && station->consecSuccess > 20)
    {
        adjustedRate = std::min(baseRate + 1, static_cast<uint32_t>(7));
        ruleReason = "GOOD_SUCCESS_BOOST";
    }
    else if (shortSuccRatio < 0.7 || station->consecFailure > 2)
    {
        adjustedRate = (baseRate > 0) ? baseRate - 1 : 0;
        ruleReason = "POOR_PERFORMANCE_REDUCTION";
    }

    std::cout << "[FIXED RULE LOGIC] Realistic SNR=" << snr << "dB, SuccRatio=" << shortSuccRatio
              << ", ConsecSucc=" << station->consecSuccess << " | BaseRate=" << baseRate
              << " -> AdjustedRate=" << adjustedRate << " (" << ruleReason << ")" << std::endl;

    return adjustedRate;
}

// FIXED: Intelligent fusion method
uint32_t
SmartWifiManagerRf::FuseMLAndRuleBased(uint32_t mlRate,
                                       uint32_t ruleRate,
                                       double mlConfidence,
                                       const SafetyAssessment& safety) const
{
    if (safety.requiresEmergencyAction)
    {
        return safety.recommendedSafeRate;
    }

    if (mlConfidence > m_confidenceThreshold)
    {
        if (m_enableAdaptiveWeighting)
        {
            // Adaptive weighting based on confidence
            double adaptiveWeight =
                std::min(0.9, m_mlGuidanceWeight * (mlConfidence / m_confidenceThreshold));
            double ruleWeight = 1.0 - adaptiveWeight;

            double blendedRate = (adaptiveWeight * mlRate) + (ruleWeight * ruleRate);
            uint32_t finalRate = static_cast<uint32_t>(std::round(blendedRate));

            // Apply conservative boost if enabled
            if (m_conservativeBoost > 1.0 && safety.riskLevel > 0.3)
            {
                finalRate = static_cast<uint32_t>(finalRate / m_conservativeBoost);
            }

            return std::min(finalRate, ruleRate + 2); // Safety clamp
        }
        else
        {
            // Fixed weighting
            double blendedRate =
                (m_mlGuidanceWeight * mlRate) + ((1.0 - m_mlGuidanceWeight) * ruleRate);
            return static_cast<uint32_t>(std::round(blendedRate));
        }
    }
    else
    {
        // Low confidence: use rule-based with slight ML influence if cached
        if (mlConfidence > 0.2)
        {
            return static_cast<uint32_t>((0.8 * ruleRate) + (0.2 * mlRate));
        }
        return ruleRate;
    }
}

// FIXED: Enhanced safety assessment using REALISTIC SNR
SmartWifiManagerRf::SafetyAssessment
SmartWifiManagerRf::AssessNetworkSafety(SmartWifiManagerRfState* station)
{
    SafetyAssessment assessment;
    assessment.context = ClassifyNetworkContext(station);
    assessment.riskLevel = CalculateRiskLevel(station);
    assessment.recommendedSafeRate = GetContextSafeRate(station, assessment.context);
    assessment.requiresEmergencyAction =
        (assessment.context == WifiContextType::EMERGENCY ||
         station->consecFailure >= m_failureThreshold || assessment.riskLevel > m_riskThreshold);
    assessment.confidenceInAssessment = 1.0 - assessment.riskLevel;
    assessment.contextStr = ContextTypeToString(assessment.context);
    station->lastContext = assessment.context;
    station->lastRiskLevel = assessment.riskLevel;
    return assessment;
}

// FIXED: Context classification using REALISTIC SNR
WifiContextType
SmartWifiManagerRf::ClassifyNetworkContext(SmartWifiManagerRfState* station) const
{
    // FIXED: Use realistic SNR for context classification
    double snr = station->lastSnr; // Contains realistic SNR
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty())
    {
        shortSuccRatio =
            static_cast<double>(
                std::count(station->shortWindow.begin(), station->shortWindow.end(), true)) /
            station->shortWindow.size();
    }

    WifiContextType result;

    // FIXED: Use realistic WiFi SNR ranges for context
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
        result = WifiContextType::GOOD_UNSTABLE; // Default case
    }

    std::cout << "[FIXED CONTEXT] Realistic SNR=" << snr << "dB, SuccRatio=" << shortSuccRatio
              << " -> Context=" << ContextTypeToString(result) << std::endl;

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
SmartWifiManagerRf::CalculateRiskLevel(SmartWifiManagerRfState* station) const
{
    double risk = 0.0;
    risk += (station->consecFailure >= m_failureThreshold) ? 0.5 : 0.0;
    risk += (station->snrVariance > 5.0) ? 0.25 : 0.0;
    risk += (station->lastSnr < 5.0) ? 0.25 : 0.0; // Using realistic SNR
    risk += std::max(0.0, 1.0 - station->confidence);
    risk = std::min(1.0, risk);
    return risk;
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

uint32_t
SmartWifiManagerRf::GetRuleBasedRate(SmartWifiManagerRfState* station) const
{
    return GetContextSafeRate(station, ClassifyNetworkContext(station));
}

// FIXED: UpdateMetrics with proper realistic SNR handling
void
SmartWifiManagerRf::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();

    // FIXED: Ensure SNR is realistic (should already be converted by caller)
    if (snr >= -30.0 && snr <= 45.0)
    { // Realistic WiFi SNR bounds
        // Update exponential moving averages
        if (station->snrFast == 0.0 && station->snrSlow == 0.0)
        {
            station->snrFast = snr;
            station->snrSlow = snr;
            station->snrVariance = 0.1;
            std::cout << "[FIXED METRICS INIT] Initialized SNR averages to " << snr
                      << "dB (realistic)" << std::endl;
        }
        else
        {
            double oldFast = station->snrFast;
            station->snrFast = m_snrAlpha * snr + (1 - m_snrAlpha) * station->snrFast;
            station->snrSlow = (m_snrAlpha / 10) * snr + (1 - m_snrAlpha / 10) * station->snrSlow;

            // Update variance
            double diff = snr - oldFast;
            station->snrVariance = 0.9 * station->snrVariance + 0.1 * (diff * diff);
        }

        std::cout << "[FIXED METRICS UPDATE] SNR=" << snr
                  << "dB (realistic), Fast=" << station->snrFast << "dB, Slow=" << station->snrSlow
                  << "dB, Variance=" << station->snrVariance << std::endl;
    }
    else
    {
        std::cout << "[ERROR METRICS] SNR out of realistic range: " << snr
                  << "dB, expected [-30, 45] dB" << std::endl;
    }

    // FIXED: Success/failure tracking
    if (success)
    {
        station->consecSuccess++;
        station->consecFailure = 0;
        station->lastPacketSuccess = true;

        // Update short-term success window
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

    // FIXED: Maintain window sizes
    if (station->shortWindow.size() > m_windowSize)
    {
        station->shortWindow.pop_front();
    }
    if (station->mediumWindow.size() > (m_windowSize * 2))
    {
        station->mediumWindow.pop_front();
    }

    // FIXED: Update confidence and severity based on recent performance
    double recentSuccessRate = 0.5;
    if (!station->shortWindow.empty())
    {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        recentSuccessRate = static_cast<double>(successes) / station->shortWindow.size();
    }

    station->confidence = 0.8 * station->confidence + 0.2 * recentSuccessRate;
    station->severity = 1.0 - station->confidence; // Inverse relationship

    station->lastUpdateTime = now;

    std::cout << "[FIXED METRICS FINAL] Success=" << success
              << ", ConsecSucc=" << station->consecSuccess
              << ", ConsecFail=" << station->consecFailure << ", Confidence=" << station->confidence
              << ", Severity=" << station->severity << std::endl;
}

void
SmartWifiManagerRf::LogContextAndDecision(const SafetyAssessment& safety,
                                          uint32_t mlRate,
                                          uint32_t ruleRate,
                                          uint32_t finalRate) const
{
    std::cout << "[FIXED CONTEXT] Context=" << safety.contextStr << " Risk=" << safety.riskLevel
              << " Emergency=" << safety.requiresEmergencyAction
              << " RecommendedSafeRate=" << safety.recommendedSafeRate << " MLRate=" << mlRate
              << " RuleRate=" << ruleRate << " FinalRate=" << finalRate << std::endl;
}

void
SmartWifiManagerRf::DebugPrintCurrentConfig() const
{
    std::cout << "[MANAGER CONFIG] Distance: " << m_benchmarkDistance << "m" << std::endl;
    std::cout << "[MANAGER CONFIG] Interferers: " << m_currentInterferers << std::endl;
    std::cout << "[MANAGER CONFIG] Strategy: " << m_oracleStrategy << std::endl;
    std::cout << "[MANAGER CONFIG] Model: " << m_modelName << std::endl;
    std::cout << "[MANAGER CONFIG] SNR Range: [" << m_minSnrDb << ", " << m_maxSnrDb << "] dB"
              << std::endl;
}

} // namespace ns3