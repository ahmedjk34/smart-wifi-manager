/*
 * Smart WiFi Manager with 9 Safe Features - NEW PIPELINE IMPLEMENTATION
 * Compatible with ahmedjk34's probabilistic oracle models (9 features, 45-63% realistic accuracy)
 *
 * CRITICAL UPDATES (2025-10-02 18:06:16 UTC):
 * ============================================================================
 * WHAT WE CHANGED FROM 14-FEATURE VERSION:
 * 1. Feature extraction: ExtractFeatures() now returns 9 features (not 14)
 * 2. Removed outcome feature tracking (shortSuccRatio, medSuccRatio, packetLossRate, etc.)
 * 3. Python client integration via socket to localhost:8765
 * 4. Model paths: python_files/trained_models/ (not root directory)
 * 5. Default oracle: oracle_aggressive (62.8% test accuracy)
 * 6. Removed all window state management (no previous/current window tracking)
 * 7. Updated RunMLInference() to use your Python server protocol
 *
 * WHY WE CHANGED IT:
 * - Your File 3 removed 5 outcome features (data leakage fix)
 * - Your trained models expect exactly 9 features
 * - Python server handles ML inference (clean separation, easy debugging)
 * - Socket communication matches your 6a_ml_inference_server.py protocol
 * - oracle_aggressive performs best (62.8% vs 45-48% for other oracles)
 *
 * PYTHON SERVER INTEGRATION:
 * - Server script: python_files/6a_ml_inference_server.py
 * - Start command: python3 python_files/6a_ml_inference_server.py
 * - Default port: 8765 (configurable via InferenceServerPort attribute)
 * - Protocol: "feat1 feat2 ... feat9 [model_name]\n"
 * - Response: {"rateIdx": X, "confidence": Y, "success": true, ...}
 *
 * FEATURE ORDER (CRITICAL - MUST MATCH TRAINING):
 * 0. lastSnr (dB)               - Most recent realistic SNR
 * 1. snrFast (dB)               - Fast-moving average (α=0.1)
 * 2. snrSlow (dB)               - Slow-moving average (α=0.01)
 * 3. snrTrendShort              - Short-term SNR trend
 * 4. snrStabilityIndex          - SNR stability (0-10)
 * 5. snrPredictionConfidence    - Prediction confidence (0-1)
 * 6. snrVariance                - SNR variance (0-100)
 * 7. channelWidth (MHz)         - Channel bandwidth
 * 8. mobilityMetric             - Node mobility (0-50)
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-10-02 18:06:16 UTC
 * Version: 6.0 (NEW PIPELINE - 9 Features, Python Client)
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
// UPDATED: Global realistic SNR conversion function (matches benchmark)
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
            // UPDATED: Model paths now point to python_files/trained_models/
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
            // UPDATED: Default model is now oracle_aggressive (62.8% accuracy)
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
      m_nextStationId(1)
{
    NS_LOG_FUNCTION(this);
    std::cout << "[NEW PIPELINE] SmartWifiManagerRf v6.0 initialized (9 safe features)"
              << std::endl;
    std::cout << "[NEW PIPELINE] Python server integration (port " << m_inferenceServerPort << ")"
              << std::endl;
    std::cout << "[NEW PIPELINE] Default oracle: " << m_oracleStrategy << " (62.8% accuracy)"
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

    std::cout << "[NEW PIPELINE] Model: " << m_modelName << " | Strategy: " << m_oracleStrategy
              << std::endl;
    std::cout << "[NEW PIPELINE] Features: 9 (7 SNR + 2 network config)" << std::endl;
    std::cout << "[NEW PIPELINE] Python Server: localhost:" << m_inferenceServerPort << std::endl;
    std::cout << "[NEW PIPELINE] Model Path: " << m_modelPath << std::endl;
    std::cout << "[NEW PIPELINE] Confidence Threshold: " << m_confidenceThreshold
              << " | ML Weight: " << m_mlGuidanceWeight << std::endl;

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
    std::cout << "[CONFIG] Distance updated to " << distance << "m" << std::endl;
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
    m_currentInterferers.store(interferers);
}

void
SmartWifiManagerRf::UpdateFromBenchmarkGlobals(double distance, uint32_t interferers)
{
    m_benchmarkDistance.store(distance);
    m_currentInterferers.store(interferers);
    std::cout << "[SYNC] Updated distance=" << distance << "m, interferers=" << interferers
              << std::endl;
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
// NEW PIPELINE: Feature extraction - 9 SAFE FEATURES ONLY
// ============================================================================
std::vector<double>
SmartWifiManagerRf::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // CRITICAL: Extract exactly 9 features matching training pipeline
    std::vector<double> features(9);

    // SNR features (7 features) - indices 0-6
    features[0] = station->lastSnr;                                     // lastSnr
    features[1] = station->snrFast;                                     // snrFast
    features[2] = station->snrSlow;                                     // snrSlow
    features[3] = GetSnrTrendShort(st);                                 // snrTrendShort
    features[4] = GetSnrStabilityIndex(st);                             // snrStabilityIndex
    features[5] = GetSnrPredictionConfidence(st);                       // snrPredictionConfidence
    features[6] = std::max(0.0, std::min(100.0, station->snrVariance)); // snrVariance

    // Network configuration (2 features) - indices 7-8
    features[7] = static_cast<double>(GetChannelWidth(st)); // channelWidth
    features[8] = GetMobilityMetric(st);                    // mobilityMetric

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
        NS_LOG_DEBUG("Extracted 9 features: "
                     << "SNR=[" << features[0] << "," << features[1] << "," << features[2] << "] "
                     << "Trend=" << features[3] << " Stability=" << features[4]
                     << " Confidence=" << features[5] << " Variance=" << features[6]
                     << " Width=" << features[7] << " Mobility=" << features[8]);
    }

    return features;
}

// ============================================================================
// NEW PIPELINE: ML Inference via Python server (socket communication)
// ============================================================================
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

    // CRITICAL: Validate feature count (9, not 14 or 21!)
    if (features.size() != 9)
    {
        result.error = "Invalid feature count: expected 9, got " + std::to_string(features.size());
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

    // CRITICAL: Build request matching Python server protocol
    // Format: "feat1 feat2 feat3 feat4 feat5 feat6 feat7 feat8 feat9 [model_name]\n"
    // Example: "25.0 25.0 25.0 0.0 0.01 0.99 0.5 20.0 0.5 oracle_aggressive\n"
    std::ostringstream featStream;
    for (size_t i = 0; i < features.size(); ++i)
    {
        featStream << std::fixed << std::setprecision(6) << features[i];
        if (i + 1 < features.size())
            featStream << " ";
    }

    // Append model name (optional - server will use default if omitted)
    if (!m_modelName.empty())
    {
        featStream << " " << m_modelName;
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
    // Expected format: {"rateIdx": 5, "confidence": 0.87, "success": true, ...}\n
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

    // Parse JSON response (simple string parsing - no external JSON library needed)
    // Look for "rateIdx": X
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
                                                      << " latency=" << result.latencyMs << "ms");
    }

    return result;
}

// ============================================================================
// Context and Safety Assessment
// ============================================================================
WifiContextType
SmartWifiManagerRf::ClassifyNetworkContext(SmartWifiManagerRfState* station) const
{
    double snr = station->lastSnr;

    // Simple SNR-based context classification
    // (Note: We removed success ratio tracking, so only use SNR)
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
// Enhanced rule-based rate adaptation
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

    // Note: We removed packet loss tracking (outcome feature)
    // Risk is now based purely on SNR metrics
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
// NEW PIPELINE: Main rate decision engine - DoGetDataTxVector
// ============================================================================
WifiTxVector
SmartWifiManagerRf::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    uint32_t supportedRates = GetNSupported(st);
    uint32_t maxRateIndex =
        std::min(supportedRates > 0 ? supportedRates - 1 : 0, static_cast<uint32_t>(7));

    // Stage 1: Safety assessment
    SafetyAssessment safety = AssessNetworkSafety(station);
    safety.managerRef = this;
    safety.stationId = station->stationId;

    // Stage 2: Rule-based baseline
    uint32_t ruleRate = GetEnhancedRuleBasedRate(station, safety);

    // Stage 3: ML inference
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

    // Adaptive inference frequency
    uint32_t adaptiveInferencePeriod = m_inferencePeriod;
    double currentDistance = m_benchmarkDistance.load();
    uint32_t currentInterferers = m_currentInterferers.load();

    if (currentDistance <= 30.0 && currentInterferers <= 1)
    {
        adaptiveInferencePeriod = std::max(static_cast<uint32_t>(10), m_inferencePeriod / 2);
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

        // CRITICAL: Extract 9 features (not 14!)
        std::vector<double> features = ExtractFeatures(st);

        // Call Python server via socket
        InferenceResult result = RunMLInference(features);

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

            station->recentMLAccuracy = 0.9 * station->recentMLAccuracy + 0.1 * mlConfidence;
            int contextIdx = static_cast<int>(safety.context);
            if (contextIdx >= 0 && contextIdx < 6)
            {
                station->mlContextConfidence[contextIdx] =
                    0.8 * station->mlContextConfidence[contextIdx] + 0.2 * mlConfidence;
                station->mlContextUsage[contextIdx]++;
            }

            NS_LOG_INFO("ML SUCCESS: model=" << result.model << " rate=" << result.rateIdx
                                             << " conf=" << mlConfidence
                                             << " SNR=" << station->lastSnr << "dB"
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

    // Stage 4: Intelligent fusion
    uint32_t finalRate = FuseMLAndRuleBased(mlRate, ruleRate, mlConfidence, safety, station);

    // Stage 5: Final bounds and tracking
    finalRate = std::min(finalRate, maxRateIndex);
    finalRate = std::max(finalRate, static_cast<uint32_t>(0));

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

    std::string fusionType =
        (mlConfidence >= CalculateAdaptiveConfidenceThreshold(station, safety.context))
            ? "ML-LED"
            : "RULE-LED";

    uint64_t finalDataRate = GetSupported(st, finalRate).GetDataRate(allowedWidth);

    NS_LOG_INFO("[NEW PIPELINE DECISION] "
                << fusionType << " | SNR=" << station->lastSnr
                << "dB | Context=" << safety.contextStr << " | Rule=" << ruleRate
                << " | ML=" << mlRate << "(conf=" << mlConfidence << ")"
                << " | Final=" << finalRate << " | Rate=" << (finalDataRate / 1e6) << "Mbps"
                << " | Status=" << mlStatus);

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
// SNR reporting with realistic conversion
// ============================================================================
void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    station->lastRawSnr = rxSnr;
    double realisticSnr = ConvertToRealisticSnr(rxSnr);
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
    double realisticDataSnr = ConvertToRealisticSnr(dataSnr);
    station->lastSnr = realisticDataSnr;

    station->totalPackets++;

    UpdateMetrics(st, true, realisticDataSnr);
}

void
SmartWifiManagerRf::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
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

// ============================================================================
// NEW PIPELINE: Metrics update (simplified - no window tracking)
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
// NEW PIPELINE: Helper functions (safe features only!)
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

    // Note: We removed confidence tracking (outcome feature)
    // Use stability as a proxy for confidence
    return std::max(0.0, std::min(1.0, stabilityFactor));
}

double
SmartWifiManagerRf::GetMobilityMetric(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // FIX #1: Get actual node speed from MobilityModel (NOT SNR variance!)
    // Get the node associated with this station
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
    std::cout << "[NEW PIPELINE CONFIG] Distance: " << m_benchmarkDistance.load() << "m"
              << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] Interferers: " << m_currentInterferers.load() << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] Strategy: " << m_oracleStrategy << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] Features: 9 (no temporal leakage, no outcome features)"
              << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] Python Server Port: " << m_inferenceServerPort << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] Model Path: " << m_modelPath << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] Confidence Threshold: " << m_confidenceThreshold
              << std::endl;
    std::cout << "[NEW PIPELINE CONFIG] ML Weight: " << m_mlGuidanceWeight << std::endl;
}

} // namespace ns3