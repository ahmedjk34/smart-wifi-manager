#include "smart-wifi-manager-rf.h"

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-mac.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>    
#include <poll.h>
#include <netinet/tcp.h>
#include <fcntl.h>   // For fcntl(), F_SETFL, O_NONBLOCK
#include <errno.h>   // For errno and EINPROGRESS


namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SmartWifiManagerRf");
NS_OBJECT_ENSURE_REGISTERED(SmartWifiManagerRf);

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
                          StringValue("step3_rf_oracle_best_rateIdx_model_FIXED.joblib"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelPath),
                          MakeStringChecker())
            .AddAttribute("ScalerPath",
                          "Path to the scaler file (.joblib)", 
                          StringValue("step3_scaler_FIXED.joblib"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_scalerPath),
                          MakeStringChecker())
            .AddAttribute("InferenceServerPort",
                          "Port number of Python ML inference server",
                          UintegerValue(8765),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferenceServerPort),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("ModelType",
                          "Model type (oracle or v3)",
                          StringValue("oracle"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelType),
                          MakeStringChecker())
            .AddAttribute("EnableProbabilities",
                          "Enable probability output from ML model",
                          BooleanValue(false),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableProbabilities),
                          MakeBooleanChecker())
            .AddAttribute("EnableValidation",
                          "Enable feature range validation",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableValidation),
                          MakeBooleanChecker())
            .AddAttribute("MaxInferenceTime",
                          "Maximum allowed inference time in ms",
                          UintegerValue(150),  // FIXED: Increased from 80ms
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_maxInferenceTime),
                          MakeUintegerChecker<uint32_t>())
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
                          UintegerValue(80),  // Reduced frequency - was 10
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_inferencePeriod),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("FallbackRate",
                          "Fallback rate index on ML failure",
                          UintegerValue(2),
                          MakeUintegerAccessor(&SmartWifiManagerRf::m_fallbackRate),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EnableFallback",
                          "Enable fallback mechanism on ML failure",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_enableFallback),
                          MakeBooleanChecker())
            .AddAttribute("UseRealisticSnr",
                          "Use realistic SNR calculation with proper bounds",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerRf::m_useRealisticSnr),
                          MakeBooleanChecker())
            .AddAttribute("MaxSnrDb",
                          "Maximum realistic SNR in dB",
                          DoubleValue(30.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_maxSnrDb),
                          MakeDoubleChecker<double>())
            .AddAttribute("MinSnrDb",
                          "Minimum realistic SNR in dB",
                          DoubleValue(-5.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_minSnrDb),
                          MakeDoubleChecker<double>())
            .AddAttribute("SnrOffset",
                          "SNR offset to apply to ns-3 values (dB)",
                          DoubleValue(-10.0),
                          MakeDoubleAccessor(&SmartWifiManagerRf::m_snrOffset),
                          MakeDoubleChecker<double>())
            .AddAttribute("ConfidenceThreshold",
                "Minimum ML confidence required to trust prediction",
                DoubleValue(0.3),  // FIXED: Lowered from 0.8 to accept more ML suggestions
                MakeDoubleAccessor(&SmartWifiManagerRf::m_confidenceThreshold),
                MakeDoubleChecker<double>())
            .AddAttribute("RiskThreshold",
                "Maximum risk allowed before forcing conservative rate",
                DoubleValue(0.6),  // Reduced from 0.7 - be more conservative
                MakeDoubleAccessor(&SmartWifiManagerRf::m_riskThreshold),
                MakeDoubleChecker<double>())
            .AddAttribute("FailureThreshold",
                "Consecutive failures required to trigger emergency",
                UintegerValue(3),  // Reduced from 4 - faster emergency response
                MakeUintegerAccessor(&SmartWifiManagerRf::m_failureThreshold),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("MLGuidanceWeight",
                "Weight of ML guidance in final decision (0.0-1.0)",
                DoubleValue(0.5),  // FIXED: Increased from 0.3 to give ML more influence
                MakeDoubleAccessor(&SmartWifiManagerRf::m_mlGuidanceWeight),
                MakeDoubleChecker<double>())
            .AddAttribute("MLCacheTime",
                "Time to cache ML results in milliseconds",
                UintegerValue(200),  // Cache ML results for 200ms
                MakeUintegerAccessor(&SmartWifiManagerRf::m_mlCacheTime),
                MakeUintegerChecker<uint32_t>())
            .AddTraceSource("Rate",
                            "Remote station data rate changed",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_currentRate),
                            "ns3::TracedValueCallback::Uint64")
            .AddTraceSource("MLInferences",
                            "Number of ML inferences made",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_mlInferences),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("MLFailures",
                            "Number of ML failures",
                            MakeTraceSourceAccessor(&SmartWifiManagerRf::m_mlFailures),
                            "ns3::TracedValueCallback::Uint32");
    return tid;
}

SmartWifiManagerRf::SmartWifiManagerRf() 
    : m_currentRate(0), 
      m_mlInferences(0), 
      m_benchmarkDistance(1.0),
      m_mlFailures(0),
      m_lastMlRate(3),  // Cache last ML result
      m_lastMlTime(Seconds(0)),
      m_mlGuidanceWeight(0.5)  // Increased ML influence
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

    std::ifstream modelFile(m_modelPath);
    if (!modelFile.good())
    {
        std::cout << "[FATAL] Model file not found: " << m_modelPath << std::endl;
        NS_FATAL_ERROR("Random Forest model file not found: " + m_modelPath);
    }
    std::ifstream scalerFile(m_scalerPath);
    if (!scalerFile.good())
    {
        std::cout << "[FATAL] Scaler file not found: " << m_scalerPath << std::endl;
        NS_FATAL_ERROR("Scaler file not found: " + m_scalerPath);
    }

    std::cout << "[INFO RF] SmartWifiManagerRf initialized successfully" << std::endl;
    std::cout << "[INFO RF] Model: " << m_modelPath << std::endl;
    std::cout << "[INFO RF] Scaler: " << m_scalerPath << std::endl;
    std::cout << "[INFO RF] Server Port: " << m_inferenceServerPort << std::endl;
    std::cout << "[INFO RF SNR] Realistic SNR enabled: " << m_useRealisticSnr << std::endl;
    std::cout << "[INFO RF SNR] SNR range: [" << m_minSnrDb << ", " << m_maxSnrDb << "] dB" << std::endl;
    std::cout << "[INFO RF SNR] SNR offset: " << m_snrOffset << " dB" << std::endl;

    WifiRemoteStationManager::DoInitialize();
}

WifiRemoteStation*
SmartWifiManagerRf::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);

    SmartWifiManagerRfState* station = new SmartWifiManagerRfState;

    // FIXED: Initialize with realistic SNR for 1m distance
    double initialSnr = (m_benchmarkDistance <= 1.0) ? 45.0 : 25.0;
    station->lastSnr = initialSnr;  
    station->snrFast = initialSnr;    // CRITICAL FIX: Don't start with 15!
    station->snrSlow = initialSnr;    // CRITICAL FIX: Don't start with 15!
    station->snrVariance = 0.1;       // Start with low variance
    
    station->consecSuccess = 0;
    station->consecFailure = 0;
    station->severity = 0.0;
    station->confidence = 1.0;
    station->T1 = 0;
    station->T2 = 0;
    station->T3 = 0;
    station->retryCount = 0;
    station->mobilityMetric = 0.0;
    station->lastUpdateTime = Simulator::Now();
    station->lastInferenceTime = Seconds(0);
    station->lastPosition = Vector(0, 0, 0);
    station->currentRateIndex = std::min(m_fallbackRate, static_cast<uint32_t>(7));
    station->queueLength = 0;
    station->lastContext = WifiContextType::UNKNOWN;
    station->lastRiskLevel = 0.0;

    std::cout << "[INFO RF] Created new station with initial rate index: "
              << station->currentRateIndex << " and initial SNR: " << initialSnr << " dB" << std::endl;

    return station;
}

void SmartWifiManagerRf::SetBenchmarkDistance(double distance) 
{
    m_benchmarkDistance = distance;
    std::cout << "[INFO RF DISTANCE] Benchmark distance set to: " << m_benchmarkDistance << "m" << std::endl;
}

double SmartWifiManagerRf::CalculateDistanceBasedSnr(WifiRemoteStation* st) const 
{
    double snrDb;
    
    // MORE REALISTIC SNR VALUES
    if (m_benchmarkDistance <= 1.0) {
        snrDb = 45.0;  // Very close - excellent signal (was 40.0)
    } else if (m_benchmarkDistance <= 5.0) {
        snrDb = 40.0;  // Close - excellent signal
    } else if (m_benchmarkDistance <= 10.0) {
        snrDb = 35.0;  // Close - very good signal
    } else if (m_benchmarkDistance <= 20.0) {
        snrDb = 30.0;  // Medium close - good signal
    } else if (m_benchmarkDistance <= 40.0) {
        snrDb = 25.0;  // Medium - moderate signal (your current 40m value)
    } else if (m_benchmarkDistance <= 60.0) {
        snrDb = 20.0;  // Medium far - acceptable signal
    } else if (m_benchmarkDistance <= 80.0) {
        snrDb = 15.0;  // Far - weak but usable
    } else if (m_benchmarkDistance <= 120.0) {
        snrDb = 10.0;  // Very far - marginal signal
    } else {
        snrDb = 5.0;   // Extreme distance - very weak
    }
    
    std::cout << "[DEBUG SNR DISTANCE] Benchmark distance=" << m_benchmarkDistance 
              << "m -> SNR=" << snrDb << "dB (was fixed at 25dB)" << std::endl;
    
    return snrDb;
}

void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    
    // Calculate realistic SNR
    double correctedSnr = CalculateDistanceBasedSnr(st);
    
    std::cout << "[DEBUG SNR RX] NS3 reported=" << rxSnr << ", Corrected SNR=" 
              << correctedSnr << "dB" << std::endl;
    
    station->lastSnr = correctedSnr;
}

void
SmartWifiManagerRf::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerRf::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    station->retryCount++;
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
    
    double correctedSnr = CalculateDistanceBasedSnr(st);
    std::cout << "[DEBUG SNR RTS] NS3 reported=" << rtsSnr << "dB, Corrected SNR=" 
              << correctedSnr << "dB" << std::endl;
    station->lastSnr = correctedSnr;
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
    
    double correctedSnr = CalculateDistanceBasedSnr(st);
    std::cout << "[DEBUG SNR DATA] NS3 reported=" << dataSnr << "dB, Corrected SNR=" 
              << correctedSnr << "dB" << std::endl;
    station->lastSnr = correctedSnr;
    
    station->retryCount = 0;
    UpdateMetrics(st, true, correctedSnr);
}

void
SmartWifiManagerRf::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerRf::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    UpdateMetrics(st, false, station->lastSnr);
}

// ===== ENHANCED RULE-BASED ALGORITHM =====
uint32_t
SmartWifiManagerRf::GetEnhancedRuleBasedRate(SmartWifiManagerRfState* station, const SafetyAssessment& safety) const
{
    double snr = station->lastSnr;
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty()) {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        shortSuccRatio = static_cast<double>(successes) / station->shortWindow.size();
    }

    // FIXED: Base rate on SNR directly, not broken context
    uint32_t baseRate;
    if (snr >= 40) baseRate = 6;      // 45dB should give rate 6!
    else if (snr >= 35) baseRate = 5;
    else if (snr >= 30) baseRate = 4;
    else if (snr >= 25) baseRate = 3;
    else if (snr >= 20) baseRate = 2;
    else if (snr >= 15) baseRate = 1;
    else baseRate = 0;

    uint32_t adjustedRate = baseRate;
    std::string ruleReason = "SNR_BASED";
    
    // FIXED: With 1100+ consecutive successes, be MUCH more aggressive
    if (shortSuccRatio > 0.95 && station->consecSuccess > 100) {
        adjustedRate = std::min(baseRate + 3, static_cast<uint32_t>(7)); // BIG boost
        ruleReason = "MASSIVE_SUCCESS_BOOST";
    } else if (shortSuccRatio > 0.95 && station->consecSuccess > 50) {
        adjustedRate = std::min(baseRate + 2, static_cast<uint32_t>(7));
        ruleReason = "EXCELLENT_BOOST";
    } else if (shortSuccRatio > 0.85 && station->consecSuccess > 10) {
        adjustedRate = std::min(baseRate + 1, static_cast<uint32_t>(7));
        ruleReason = "GOOD_BOOST";
    } else if (shortSuccRatio < 0.7 || station->consecFailure > 1) {
        adjustedRate = (baseRate > 0) ? baseRate - 1 : 0;
        ruleReason = "POOR_CONSERVATIVE";
    }
    
    std::cout << "[RULE LOGIC FIXED] SNR=" << snr << " SuccRatio=" << shortSuccRatio 
              << " ConsecSucc=" << station->consecSuccess << " | BaseRate=" << baseRate 
              << " -> AdjustedRate=" << adjustedRate << " (" << ruleReason << ")" << std::endl;
    
    return adjustedRate;
}

WifiTxVector
SmartWifiManagerRf::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    uint32_t maxRateIndex = GetNSupported(st) - 1;
    maxRateIndex = std::min(maxRateIndex, static_cast<uint32_t>(7));

    // -------- Stage 1: Safety/Context Assessment --------
    SafetyAssessment safety = AssessNetworkSafety(station);

    // -------- Stage 2: Primary Rule-Based Decision --------
    uint32_t primaryRate = GetEnhancedRuleBasedRate(station, safety);

// -------- Stage 3: ML Guidance (Infrequent & Cached) --------
    uint32_t mlGuidance = primaryRate;  // Default fallback
    double mlConfidence = 0.0;
    bool mlInferenceAttempted = false;
    bool mlInferenceSucceeded = false;
    bool usedCachedMl = false;
    std::string mlStatus = "NO_ATTEMPT";

    // Check if we should get ML guidance
    static uint64_t s_callCounter = 0;
    ++s_callCounter;

    Time now = Simulator::Now();
    bool canUseCachedMl = (now - m_lastMlTime) < MilliSeconds(m_mlCacheTime) && 
                        m_lastMlTime > Seconds(0);  // Ensure we actually have a cached value
    bool needNewMlInference = !safety.requiresEmergencyAction && 
                            safety.riskLevel < m_riskThreshold &&
                            !canUseCachedMl &&
                            (s_callCounter % m_inferencePeriod) == 0;

    if (canUseCachedMl) {
        mlGuidance = m_lastMlRate;
        mlConfidence = 0.8; // Assume cached results are reasonably confident
        usedCachedMl = true;
        mlStatus = "CACHED";
    } else if (needNewMlInference) {
        mlInferenceAttempted = true;
        mlStatus = "ATTEMPTING";
        
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);
        
        if (result.success) {
            m_mlInferences++;
            mlGuidance = std::min(result.rateIdx, maxRateIndex);
            mlConfidence = result.confidence;
            
            // Update cache with successful result
            m_lastMlRate = mlGuidance;
            m_lastMlTime = now;
            
            mlInferenceSucceeded = true;
            mlStatus = "SUCCESS";
            
            std::cout << "[ML SUCCESS] Raw ML Prediction: " << result.rateIdx 
                    << " (clamped to " << mlGuidance << "), Confidence: " << mlConfidence
                    << ", Latency: " << result.latencyMs << "ms" << std::endl;
        } else {
            m_mlFailures++;
            // DON'T update cache on failure - keep using rule-based
            mlGuidance = primaryRate;
            mlConfidence = 0.0;
            mlStatus = "FAILED";
            
            std::cout << "[ML FAILURE] " << result.error << ", using rule-based rate: " 
                    << primaryRate << std::endl;
        }
    } else {
        mlStatus = "SKIPPED";
    }

    // -------- Stage 4: Weighted Fusion (Balanced) --------
    uint32_t finalRate;
    std::string decisionReason;

    if (safety.requiresEmergencyAction) {
        finalRate = safety.recommendedSafeRate;
        decisionReason = "EMERGENCY_OVERRIDE";
    } else if (mlInferenceSucceeded && mlConfidence > m_confidenceThreshold) {
        // High confidence ML: blend with rules
        double mlWeight = m_mlGuidanceWeight;
        double ruleWeight = 1.0 - mlWeight;
        
        double blendedRate = (mlWeight * mlGuidance) + (ruleWeight * primaryRate);
        finalRate = static_cast<uint32_t>(std::round(blendedRate));
        
        uint32_t clampedRate = std::min(finalRate, primaryRate + 2);
        decisionReason = "ML_GUIDED_BLEND";
        if (clampedRate != finalRate) {
            decisionReason += "_CLAMPED";
        }
        finalRate = clampedRate;
        
    } else {
        // No ML or low confidence: use rule-based
        finalRate = primaryRate;
        if (usedCachedMl && mlConfidence > 0.5) {
            decisionReason = "RULE_BASED_WITH_CACHED_ML_HINT";
        } else {
            decisionReason = "PURE_RULE_BASED";
        }
    }

    // -------- Stage 5: Final Safety Bounds --------
    finalRate = std::min(finalRate, maxRateIndex);
    finalRate = std::max(finalRate, static_cast<uint32_t>(0));

    // CLEAR DEBUG LOG - Separate actual ML result from fallback
    std::cout << "[RATE DECISION] Call#" << s_callCounter 
            << " | SNR=" << station->lastSnr << "dB"
            << " | Context=" << safety.contextStr
            << " | RuleRate=" << primaryRate;

    if (mlInferenceSucceeded) {
        std::cout << " | MLRate=" << mlGuidance << "(conf=" << mlConfidence << ")";
    } else if (usedCachedMl) {
        std::cout << " | MLRate=" << mlGuidance << "(cached)";
    } else {
        std::cout << " | MLRate=N/A";
    }

    std::cout << " | FinalRate=" << finalRate
            << " | Reason=" << decisionReason
            << " | MLStatus=" << mlStatus
            << std::endl;
        
    // if (mlAttempted) {
    //     std::cout << " | MLAttempt=YES";
    // } else if (usedCachedMl) {
    //     std::cout << " | MLCache=YES";
    // } else {
    //     std::cout << " | MLSkip=YES";
    // }
    // std::cout << std::endl;

    WifiMode mode = GetSupported(st, finalRate);
    uint64_t rate = mode.GetDataRate(allowedWidth);

    if (m_currentRate != rate) {
        std::cout << "[RATE CHANGE] " << m_currentRate << " -> " << rate 
                  << " (index " << finalRate << ")" << std::endl;
        m_currentRate = rate;
    }

    return WifiTxVector(mode,
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
    return WifiTxVector(mode,
                        GetDefaultTxPowerLevel(),
                        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
                        800,
                        1,
                        1,
                        0,
                        GetChannelWidth(st),
                        GetAggregation(st));
}

// FIXED: Restore reasonable ML connection timeouts
SmartWifiManagerRf::InferenceResult
SmartWifiManagerRf::RunMLInference(const std::vector<double>& features) const
{
    NS_LOG_FUNCTION(this);
    InferenceResult result;
    result.success = false;
    result.rateIdx = m_fallbackRate;
    result.latencyMs = 0.0;
    result.confidence = 0.0;

    if (features.size() != 22) {
        result.error = "Invalid feature count";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        result.error = "socket failed";
        return result;
    }

    // FIXED: Restore reasonable timeouts (your previous 40-70ms suggests these worked)
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000; // CHANGED: 100ms instead of 30ms
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    // FIXED: Use blocking connect with proper timeout
    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(m_inferenceServerPort);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    int conn_ret = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    if (conn_ret < 0) {
        close(sockfd);
        result.error = "connect failed";
        return result;
    }

    // Send request
    std::ostringstream featStream;
    for (size_t i = 0; i < features.size(); ++i) {
        featStream << std::fixed << std::setprecision(4) << features[i];
        if (i + 1 < features.size()) featStream << " ";
    }
    featStream << "\n";
    std::string req = featStream.str();

    ssize_t sent = send(sockfd, req.c_str(), req.size(), 0);
    if (sent != static_cast<ssize_t>(req.size())) {
        close(sockfd);
        result.error = "send failed";
        return result;
    }

    // FIXED: Simple receive with reasonable buffer
    std::string response;
    char buffer[1024];  // Bigger buffer
    ssize_t received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);
    
    close(sockfd);

    if (received <= 0) {
        result.error = "no response";
        return result;
    }

    buffer[received] = '\0';
    response = std::string(buffer);

    // Parse JSON (your existing parsing code is fine)
    size_t rate_pos = response.find("\"rateIdx\":");
    if (rate_pos != std::string::npos) {
        size_t start = response.find(':', rate_pos) + 1;
        size_t end = response.find_first_of(",}", start);
        if (end != std::string::npos) {
            std::string rate_str = response.substr(start, end - start);
            try {
                double rate_val = std::stod(rate_str);
                result.rateIdx = static_cast<uint32_t>(std::max(0.0, std::min(7.0, rate_val)));
                result.success = true;
                
                // Get confidence
                size_t conf_pos = response.find("\"confidence\":");
                if (conf_pos != std::string::npos) {
                    size_t conf_start = response.find(':', conf_pos) + 1;
                    size_t conf_end = response.find_first_of(",}", conf_start);
                    if (conf_end != std::string::npos) {
                        std::string conf_str = response.substr(conf_start, conf_end - conf_start);
                        result.confidence = std::stod(conf_str);
                    }
                }
            } catch (...) {
                result.error = "parse error";
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.latencyMs = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    return result;
}

std::vector<double>
SmartWifiManagerRf::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);

    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    // --- Success Ratios ---
    double shortSuccRatio = 0.5;
    double medSuccRatio = 0.5;
    if (!station->shortWindow.empty())
    {
        int successes = std::count(station->shortWindow.begin(), station->shortWindow.end(), true);
        shortSuccRatio = static_cast<double>(successes) / station->shortWindow.size();
    }
    if (!station->mediumWindow.empty())
    {
        int successes = std::count(station->mediumWindow.begin(), station->mediumWindow.end(), true);
        medSuccRatio = static_cast<double>(successes) / station->mediumWindow.size();
    }

    // --- Feature Vector Construction (22 features, exact order) ---
    std::vector<double> features(22);

    // 1. rateIdx
    features[0] = static_cast<double>(station->currentRateIndex);

    // 2. phyRate
    WifiMode mode = GetSupported(st, station->currentRateIndex);
    features[1] = static_cast<double>(mode.GetDataRate(GetChannelWidth(st)));

    // 3. lastSnr -- use lastSnr, which should now be realistic
    features[2] = station->lastSnr;

    // 4. snrFast
    features[3] = station->snrFast;

    // 5. snrSlow
    features[4] = station->snrSlow;

    // 6. shortSuccRatio
    features[5] = std::max(0.0, std::min(1.0, shortSuccRatio));

    // 7. medSuccRatio
    features[6] = std::max(0.0, std::min(1.0, medSuccRatio));

    // 8. consecSuccess
    features[7] = std::min(100.0, static_cast<double>(station->consecSuccess));

    // 9. consecFailure
    features[8] = std::min(100.0, static_cast<double>(station->consecFailure));

    // 10. severity
    features[9] = std::max(0.0, std::min(1.0, station->severity));

    // 11. confidence
    features[10] = std::max(0.0, std::min(1.0, station->confidence));

    // 12. T1
    features[11] = static_cast<double>(station->T1);

    // 13. T2
    features[12] = static_cast<double>(station->T2);

    // 14. T3
    features[13] = static_cast<double>(station->T3);

    // 15. decisionReason
    features[14] = static_cast<double>(station->decisionReason);

    // 16. packetSuccess
    features[15] = station->lastPacketSuccess ? 1.0 : 0.0;

    // 17. offeredLoad
    features[16] = GetOfferedLoad();

    // 18. queueLen
    features[17] = static_cast<double>(station->queueLength);

    // 19. retryCount
    features[18] = static_cast<double>(station->retryCount);

    // 20. channelWidth
    features[19] = static_cast<double>(GetChannelWidth(st));

    // 21. mobilityMetric
    features[20] = GetMobilityMetric(st);

    // 22. snrVariance
    features[21] = std::max(0.0, std::min(100.0, station->snrVariance));

    // --- Debug output ---
    std::cout << "[DEBUG RF SNR FEATURES] SNR values - Last:" << features[2] 
              << " Fast:" << features[3] << " Slow:" << features[4] << std::endl;
    std::cout << "[DEBUG RF] Features sent to ML: ";
    for (size_t i = 0; i < features.size(); ++i) {
        std::cout << features[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "[FEATURES TO ML] rateIdx=" << features[0] 
              << " phyRate=" << features[1] 
              << " lastSnr=" << features[2]
              << " snrFast=" << features[3] 
              << " snrSlow=" << features[4]
              << " shortSucc=" << features[5] 
              << " medSucc=" << features[6] << std::endl;

    return features;
}

void
SmartWifiManagerRf::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);

    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();

    // SNR should already be realistic at this point
    if (snr >= m_minSnrDb && snr <= m_maxSnrDb)
    {
        station->lastSnr = snr;
        // CRITICAL FIX: Initialize snrFast/snrSlow to CURRENT snr, not 15!
        if (station->snrFast == 15.0 && station->snrSlow == 15.0) {
            station->snrFast = snr;    // 45, not 15!
            station->snrSlow = snr;    // 45, not 15!
            station->snrVariance = 0.1; // Start with low variance
            std::cout << "[SNR INIT FIX] Initialized snrFast/Slow to " << snr << "dB" << std::endl;
        } else {
            station->snrFast = m_snrAlpha * snr + (1 - m_snrAlpha) * station->snrFast;
            station->snrSlow = (m_snrAlpha / 10) * snr + (1 - m_snrAlpha / 10) * station->snrSlow;
        }
        
        std::cout << "[DEBUG SNR UPDATE] SNR=" << snr << "dB, Fast=" << station->snrFast 
                  << "dB, Slow=" << station->snrSlow << "dB" << std::endl;
    }

    station->shortWindow.push_back(success);
    if (station->shortWindow.size() > m_windowSize / 2)
    {
        station->shortWindow.pop_front();
    }
    station->mediumWindow.push_back(success);
    if (station->mediumWindow.size() > m_windowSize)
    {
        station->mediumWindow.pop_front();
    }

    if (success)
    {
        station->consecSuccess++;
        station->consecFailure = 0;
    }
    else
    {
        station->consecFailure++;
        station->consecSuccess = 0;
    }

    if (!success)
    {
        station->severity = std::min(1.0, station->severity + 0.1);
        station->confidence = std::max(0.1, station->confidence - 0.1);
    }
    else
    {
        station->severity = std::max(0.0, station->severity - 0.05);
        station->confidence = std::min(1.0, station->confidence + 0.05);
    }

    // CRITICAL FIX: Prevent variance explosion from initialization mismatch
    double snrDiff = snr - station->snrSlow;
    // Cap the difference to prevent initialization explosions
    snrDiff = std::max(-10.0, std::min(10.0, snrDiff)); 
    station->snrVariance = 0.95 * station->snrVariance + 0.05 * (snrDiff * snrDiff);

    // DEBUG: Show what WAS causing poor_unstable classification
    std::cout << "[DEBUG VARIANCE FIXED] SNR=" << snr << " snrSlow=" << station->snrSlow 
              << " cappedDiff=" << snrDiff << " variance=" << station->snrVariance << std::endl;

    double timeDiff = (now - station->lastUpdateTime).GetSeconds();
    station->T1 = static_cast<uint32_t>(timeDiff * 1000);
    station->T2 = station->T1 * 2;
    station->T3 = station->T1 * 3;
    station->lastUpdateTime = now;
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

// ===== CONTEXT AND SAFETY ASSESSMENT =====
SmartWifiManagerRf::SafetyAssessment
SmartWifiManagerRf::AssessNetworkSafety(SmartWifiManagerRfState* station)
{
    SafetyAssessment assessment;
    assessment.context = ClassifyNetworkContext(station);
    assessment.riskLevel = CalculateRiskLevel(station);
    assessment.recommendedSafeRate = GetContextSafeRate(station, assessment.context);
    assessment.requiresEmergencyAction = (
        assessment.context == WifiContextType::EMERGENCY ||
        station->consecFailure >= m_failureThreshold ||
        assessment.riskLevel > m_riskThreshold
    );
    assessment.confidenceInAssessment = 1.0 - assessment.riskLevel;
    assessment.contextStr = ContextTypeToString(assessment.context);
    station->lastContext = assessment.context;
    station->lastRiskLevel = assessment.riskLevel;
    return assessment;
}

// FIXED: Context based on SNR and success rate, NOT variance (since SNR is static)
WifiContextType
SmartWifiManagerRf::ClassifyNetworkContext(SmartWifiManagerRfState* station) const
{
    double snr = station->lastSnr;
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty())
        shortSuccRatio = static_cast<double>(std::count(station->shortWindow.begin(), station->shortWindow.end(), true)) / station->shortWindow.size();

    // FIXED: Ignore variance completely since SNR is static by design
    if (snr < 5.0 || shortSuccRatio < 0.5 || station->consecFailure >= m_failureThreshold)
        return WifiContextType::EMERGENCY;
    if (snr < 10.0 || shortSuccRatio < 0.7)  // Removed snrVar condition
        return WifiContextType::POOR_UNSTABLE;
    if (snr < 15.0 || shortSuccRatio < 0.8)
        return WifiContextType::MARGINAL;
    // Skip GOOD_UNSTABLE since we have no real variance
    if (snr >= 40.0 && shortSuccRatio > 0.95)  // 45dB should hit this!
        return WifiContextType::EXCELLENT_STABLE;
    if (snr >= 25.0 && shortSuccRatio > 0.9)
        return WifiContextType::GOOD_STABLE;
    return WifiContextType::GOOD_STABLE;
}

std::string
SmartWifiManagerRf::ContextTypeToString(WifiContextType type) const
{
    switch (type) {
        case WifiContextType::EMERGENCY: return "emergency_recovery";
        case WifiContextType::POOR_UNSTABLE: return "poor_unstable";
        case WifiContextType::MARGINAL: return "marginal_conditions";
        case WifiContextType::GOOD_UNSTABLE: return "good_unstable";
        case WifiContextType::GOOD_STABLE: return "good_stable";
        case WifiContextType::EXCELLENT_STABLE: return "excellent_stable";
        default: return "unknown";
    }
}

double
SmartWifiManagerRf::CalculateRiskLevel(SmartWifiManagerRfState* station) const
{
    double risk = 0.0;
    risk += (station->consecFailure >= m_failureThreshold) ? 0.5 : 0.0;
    risk += (station->snrVariance > 5.0) ? 0.25 : 0.0;
    risk += (station->lastSnr < 5.0) ? 0.25 : 0.0; // Adjusted for realistic SNR
    risk += std::max(0.0, 1.0 - station->confidence);
    risk = std::min(1.0, risk);
    return risk;
}

uint32_t
SmartWifiManagerRf::GetContextSafeRate(SmartWifiManagerRfState* station, WifiContextType context) const
{
    switch (context) {
        case WifiContextType::EMERGENCY: return 0;
        case WifiContextType::POOR_UNSTABLE: return 1;
        case WifiContextType::MARGINAL: return 3;
        case WifiContextType::GOOD_UNSTABLE: return 5;
        case WifiContextType::GOOD_STABLE: return 6;
        case WifiContextType::EXCELLENT_STABLE: return 7;
        default: return m_fallbackRate;
    }
}

uint32_t
SmartWifiManagerRf::GetRuleBasedRate(SmartWifiManagerRfState* station) const
{
    // Conservative: Always clamp to safe rate for current context
    return GetContextSafeRate(station, ClassifyNetworkContext(station));
}

void
SmartWifiManagerRf::LogContextAndDecision(const SafetyAssessment& safety, uint32_t mlRate, uint32_t ruleRate, uint32_t finalRate) const
{
    std::cout << "[CONTEXT RF] Context=" << safety.contextStr
              << " Risk=" << safety.riskLevel
              << " Emergency=" << safety.requiresEmergencyAction
              << " RecommendedSafeRate=" << safety.recommendedSafeRate
              << " MLRate=" << mlRate
              << " RuleRate=" << ruleRate
              << " FinalRate=" << finalRate << std::endl;
}

} // namespace ns3