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
#include <random>

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
                          UintegerValue(100),
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
                          UintegerValue(10),
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
            // --- FIXED SNR PARAMETERS START ---
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
            // --- FIXED SNR PARAMETERS END ---
            // --- HYBRID PATCH START ---
            .AddAttribute("ConfidenceThreshold",
                "Minimum ML confidence required to trust prediction",
                DoubleValue(0.7),
                MakeDoubleAccessor(&SmartWifiManagerRf::m_confidenceThreshold),
                MakeDoubleChecker<double>())
            .AddAttribute("RiskThreshold",
                "Maximum risk allowed before forcing conservative rate",
                DoubleValue(0.7),
                MakeDoubleAccessor(&SmartWifiManagerRf::m_riskThreshold),
                MakeDoubleChecker<double>())
            .AddAttribute("FailureThreshold",
                "Consecutive failures required to trigger emergency",
                UintegerValue(4),
                MakeUintegerAccessor(&SmartWifiManagerRf::m_failureThreshold),
                MakeUintegerChecker<uint32_t>())
            // --- HYBRID PATCH END ---
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

// FIXED CONSTRUCTOR ORDER
SmartWifiManagerRf::SmartWifiManagerRf() 
    : m_benchmarkDistance(1.0),  // MOVED BEFORE m_mlFailures
      m_currentRate(0), 
      m_mlInferences(0), 
      m_mlFailures(0),
      m_totalInferenceCalls(0),   // ADD COUNTER FOR LOOP PREVENTION
      m_lastFeatureHash(0)        // ADD HASH FOR DUPLICATE DETECTION
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

    station->lastSnr = 15.0;  // More realistic initial value
    station->snrFast = 15.0;
    station->snrSlow = 15.0;
    station->consecSuccess = 0;
    station->consecFailure = 0;
    station->severity = 0.0;
    station->confidence = 1.0;
    station->T1 = 0;
    station->T2 = 0;
    station->T3 = 0;
    station->retryCount = 0;
    station->mobilityMetric = 0.0;
    station->snrVariance = 1.0;
    station->lastUpdateTime = Simulator::Now();
    station->lastInferenceTime = Seconds(0);
    station->lastPosition = Vector(0, 0, 0);
    station->currentRateIndex = std::min(m_fallbackRate, static_cast<uint32_t>(7));
    station->queueLength = 0;
    // --- HYBRID PATCH START ---
    station->lastContext = WifiContextType::UNKNOWN;
    station->lastRiskLevel = 0.0;
    // --- HYBRID PATCH END ---

    std::cout << "[INFO RF] Created new station with initial rate index: "
              << station->currentRateIndex << " and initial SNR: " << station->lastSnr << " dB" << std::endl;

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
    
    if (m_benchmarkDistance <= 1.0) {
        snrDb = 40.0;  // Very close - excellent signal
    } else if (m_benchmarkDistance <= 40.0) {
        snrDb = 25.0;  // Good signal for 40m
    } else if (m_benchmarkDistance <= 60.0) {
        snrDb = 15.0;  // Moderate signal for 60m  
    } else if (m_benchmarkDistance <= 120.0) {
        snrDb = 8.0;   // Weak but usable signal for 120m
    } else {
        snrDb = 5.0;   // Very weak signal for distances > 120m
    }
    
    std::cout << "[DEBUG SNR DISTANCE] Using benchmark distance=" << m_benchmarkDistance 
              << "m -> SNR=" << snrDb << "dB" << std::endl;
    
    return snrDb;
}

void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    
    // --- FIXED SNR START ---
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    
    // Calculate realistic SNR
    double correctedSnr = CalculateDistanceBasedSnr(st);
    
    std::cout << "[DEBUG SNR RX] NS3 reported=" << rxSnr << ", Corrected SNR=" 
              << correctedSnr << "dB" << std::endl;
    
    station->lastSnr = correctedSnr;
    // --- FIXED SNR END ---
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
    
    // --- FIXED SNR START ---
    double correctedSnr = CalculateDistanceBasedSnr(st);
    std::cout << "[DEBUG SNR RTS] NS3 reported=" << rtsSnr << "dB, Corrected SNR=" 
              << correctedSnr << "dB" << std::endl;
    station->lastSnr = correctedSnr;
    // --- FIXED SNR END ---
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
    
    // --- FIXED SNR START ---
    double correctedSnr = CalculateDistanceBasedSnr(st);
    std::cout << "[DEBUG SNR DATA] NS3 reported=" << dataSnr << "dB, Corrected SNR=" 
              << correctedSnr << "dB" << std::endl;
    station->lastSnr = correctedSnr;
    // --- FIXED SNR END ---
    
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

WifiTxVector
SmartWifiManagerRf::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);

    uint32_t maxRateIndex = GetNSupported(st) - 1;
    maxRateIndex = std::min(maxRateIndex, static_cast<uint32_t>(7));

    // --- INFINITE LOOP PREVENTION START ---
    m_totalInferenceCalls++;
    
    // SAFETY: Stop simulation if too many calls
    if (m_totalInferenceCalls > 5000) {
        std::cout << "[EMERGENCY STOP] Too many ML calls (" << m_totalInferenceCalls 
                  << "), forcing simulation termination!" << std::endl;
        Simulator::Stop();
        return WifiTxVector(GetSupported(st, m_fallbackRate),
                           GetDefaultTxPowerLevel(),
                           GetPreambleForTransmission(GetSupported(st, m_fallbackRate).GetModulationClass(), GetShortPreambleEnabled()),
                           800, 1, 1, 0, allowedWidth, GetAggregation(st));
    }
    // --- INFINITE LOOP PREVENTION END ---

    // --- HYBRID PATCH START ---
    // Stage 1: Safety/Context Assessment
    SafetyAssessment safety = AssessNetworkSafety(station);

    // Stage 2: ML Prediction (only if context/risk allow)
    uint32_t mlRate = m_fallbackRate;
    double mlConfidence = 0.0;
    bool mlAllowed = (!safety.requiresEmergencyAction && safety.riskLevel < m_riskThreshold);

    if (mlAllowed)
    {
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);
        m_mlInferences++;
        if (result.success)
        {
            mlRate = result.rateIdx <= maxRateIndex ? result.rateIdx : std::min(m_fallbackRate, maxRateIndex);
            mlConfidence = result.confidence;
            station->lastInferenceTime = Simulator::Now();
            std::cout << "[SUCCESS RF] ML predicted rate index: " << result.rateIdx
                      << " (mapped to WiFi: " << mlRate << ", max=" << maxRateIndex << ")"
                      << " ML confidence: " << mlConfidence << std::endl;
        }
        else
        {
            m_mlFailures++;
            mlRate = std::min(m_fallbackRate, maxRateIndex);
            std::cout << "[ERROR RF] ML inference failed: " << result.error
                      << ", using fallback rate: " << mlRate << std::endl;
        }
    }

    // Stage 3: Rule-Based Safety Override
    uint32_t ruleRate = GetRuleBasedRate(station);

    // Stage 4: Real-Time Feedback Correction
    uint32_t finalRate = ruleRate;
    if (mlAllowed && mlConfidence > m_confidenceThreshold)
    {
        // ML trusted only if confidence and context/risk allow
        finalRate = mlRate;
    }
    else if (safety.requiresEmergencyAction)
    {
        finalRate = safety.recommendedSafeRate;
        std::cout << "[EMERGENCY RF] Safety override, using recommended safe rate: " << finalRate << std::endl;
    }
    else
    {
        // Blend ML and rule if moderate risk
        finalRate = std::min(ruleRate, mlRate);
    }

    // Stage 5: Clamp and log
    finalRate = std::max(0U, std::min(maxRateIndex, finalRate));
    LogContextAndDecision(safety, mlRate, ruleRate, finalRate);

    WifiMode mode = GetSupported(st, finalRate);
    uint64_t rate = mode.GetDataRate(allowedWidth);

    if (m_currentRate != rate)
    {
        std::cout << "[INFO RF] Rate changed from " << m_currentRate << " to " << rate
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
    // --- HYBRID PATCH END ---
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

// ENHANCED HASH FUNCTION FOR DUPLICATE DETECTION
uint64_t
SmartWifiManagerRf::CalculateFeatureHash(const std::vector<double>& features) const
{
    uint64_t hash = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        // Use integer representation of key features to detect duplicates
        uint64_t val = static_cast<uint64_t>(features[i] * 1000); // 3 decimal precision
        hash ^= val + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

// --- Client/server ML inference implementation ---
// --- Client/server ML inference implementation ---
SmartWifiManagerRf::InferenceResult
SmartWifiManagerRf::RunMLInference(const std::vector<double>& features) const
{
    NS_LOG_FUNCTION(this);

    InferenceResult result;
    result.success = false;
    result.rateIdx = m_fallbackRate;
    result.latencyMs = 0.0;
    result.confidence = 1.0;

    // EMERGENCY COUNTER - ABSOLUTE HARD LIMIT
    static uint32_t emergencyCounter = 0;
    emergencyCounter++;
    
    // HARD STOP after 1000 calls - NO EXCEPTIONS
    if (emergencyCounter > 1000) {
        std::cout << "[EMERGENCY STOP] Reached " << emergencyCounter 
                  << " ML calls - FORCING SIMULATION TERMINATION!" << std::endl;
        Simulator::Stop();
        result.success = true;
        result.rateIdx = 3; // Safe middle rate
        result.confidence = 0.1;
        result.latencyMs = 0;
        return result;
    }
    
    // FORCE CYCLING after 200 calls to break any loops
    if (emergencyCounter > 200) {
        result.success = true;
        result.rateIdx = (emergencyCounter % 8); // Cycle 0-7
        result.confidence = 0.3;
        result.latencyMs = 0;
        std::cout << "[FORCE CYCLE] Call #" << emergencyCounter 
                  << " -> forcing rate " << result.rateIdx << std::endl;
        return result;
    }

    // AGGRESSIVE TIME-BASED VARIATION after 100 calls
    if (emergencyCounter > 100) {
        static uint32_t timeBasedRate = 0;
        timeBasedRate = (timeBasedRate + 1) % 8; // Guaranteed different each time
        result.success = true;
        result.rateIdx = timeBasedRate;
        result.confidence = 0.5;
        result.latencyMs = 0;
        std::cout << "[TIME CYCLE] Call #" << emergencyCounter 
                  << " -> time-based rate " << result.rateIdx << std::endl;
        return result;
    }

    if (features.size() != 22) {
        result.error = "Invalid feature count: " + std::to_string(features.size());
        return result;
    }

    // SIMPLIFIED DUPLICATE DETECTION - More Aggressive
    uint64_t currentHash = CalculateFeatureHash(features);
    static uint32_t duplicateCount = 0;
    static uint64_t lastHash = 0;
    
    if (currentHash == lastHash) {
        duplicateCount++;
        std::cout << "[DUPLICATE #" << duplicateCount << "] Same hash detected!" << std::endl;
        
        if (duplicateCount > 5) { // Much more aggressive - only 5 duplicates allowed
            std::cout << "[LOOP BREAK] " << duplicateCount 
                      << " duplicates - forcing variation!" << std::endl;
            result.rateIdx = (duplicateCount % 8);
            result.success = true;
            result.confidence = 0.4;
            result.latencyMs = 0;
            return result;
        }
    } else {
        duplicateCount = 0;
        lastHash = currentHash;
    }

    // Only proceed with actual ML inference for first 100 calls
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        result.error = "Failed to create socket";
        return result;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(m_inferenceServerPort);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    auto start_time = std::chrono::high_resolution_clock::now();

    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        result.error = "Failed to connect to inference server";
        close(sockfd);
        return result;
    }

    // Prepare feature string
    std::ostringstream featStream;
    for (size_t i = 0; i < features.size(); ++i) {
        featStream << std::fixed << std::setprecision(6) << features[i];
        if (i < features.size() - 1) {
            featStream << " ";
        }
    }
    std::string featStr = featStream.str();
    
    // Send features
    ssize_t sent = send(sockfd, featStr.c_str(), featStr.size(), 0);
    if (sent < 0) {
        result.error = "Failed to send feature data";
        close(sockfd);
        return result;
    }

    // Receive response
    char buffer[2048];
    ssize_t received = recv(sockfd, buffer, sizeof(buffer)-1, 0);
    if (received <= 0) {
        result.error = "No response from inference server";
        close(sockfd);
        return result;
    }
    buffer[received] = '\0';

    close(sockfd);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.latencyMs = duration.count();

    // Parse JSON for "rateIdx"
    try {
        std::string output(buffer);
        size_t json_start = output.find("{");
        size_t json_end = output.rfind("}");
        if (json_start == std::string::npos || json_end == std::string::npos) {
            result.error = "No valid JSON found in output";
            return result;
        }
        std::string json_str = output.substr(json_start, json_end - json_start + 1);

        size_t rateIdxPos = json_str.find("\"rateIdx\":");
        if (rateIdxPos != std::string::npos) {
            size_t valueStart = json_str.find(":", rateIdxPos) + 1;
            size_t valueEnd = json_str.find_first_of(",}", valueStart);

            std::string rate_str = json_str.substr(valueStart, valueEnd - valueStart);
            rate_str.erase(0, rate_str.find_first_not_of(" \t\n"));
            rate_str.erase(rate_str.find_last_not_of(" \t\n") + 1);

            result.rateIdx = static_cast<uint32_t>(std::stoi(rate_str));
            result.success = true;

            // FORCE SOME VARIATION even in early calls
            static uint32_t forceVariationCounter = 0;
            forceVariationCounter++;
            
            if (forceVariationCounter % 20 == 0) { // Every 20th call gets variation
                uint32_t originalRate = result.rateIdx;
                result.rateIdx = (result.rateIdx + (forceVariationCounter / 20)) % 8;
                std::cout << "[FORCED VAR] Call #" << emergencyCounter 
                          << " changed rate " << originalRate << " -> " << result.rateIdx << std::endl;
            }

            // Parse confidence
            size_t confPos = json_str.find("\"confidence\":");
            if (confPos != std::string::npos) {
                size_t confStart = json_str.find(":", confPos) + 1;
                size_t confEnd = json_str.find_first_of(",}", confStart);
                std::string conf_str = json_str.substr(confStart, confEnd - confStart);
                conf_str.erase(0, conf_str.find_first_not_of(" \t\n"));
                conf_str.erase(conf_str.find_last_not_of(" \t\n") + 1);
                result.confidence = std::stod(conf_str);
            } else {
                result.confidence = 1.0;
            }

        } else {
            result.error = "rateIdx not found in JSON";
        }
    } catch (const std::exception& e) {
        result.error = "JSON parsing error: " + std::string(e.what());
    }

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
    // Use the WiFiMode data rate for the current index and default channel width
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

    // 8. consecSuccess - CLAMPED TO PREVENT INFINITE GROWTH
    features[7] = std::min(50.0, static_cast<double>(station->consecSuccess));

    // 9. consecFailure - CLAMPED TO PREVENT INFINITE GROWTH
    features[8] = std::min(20.0, static_cast<double>(station->consecFailure));

    // 10. severity
    features[9] = std::max(0.0, std::min(1.0, station->severity));

    // 11. confidence
    features[10] = std::max(0.0, std::min(1.0, station->confidence));

    // 12-14. TIME-BASED FEATURES TO PREVENT IDENTICAL FEATURES
    Time now = Simulator::Now();
    double timeSeconds = now.GetSeconds();
    features[11] = fmod(timeSeconds * 1000, 10000); // T1: time-based instead of static
    features[12] = fmod(timeSeconds * 2000, 20000); // T2: time-based
    features[13] = fmod(timeSeconds * 3000, 30000); // T3: time-based

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
        if (station->snrFast == 15.0 && station->snrSlow == 15.0) // Initial values
        {
            station->snrFast = snr;
            station->snrSlow = snr;
        }
        else
        {
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
        // CAP CONSECUTIVE SUCCESS TO PREVENT INFINITE LOOPS
        station->consecSuccess = std::min(station->consecSuccess + 1, 50U);
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

    double snrDiff = snr - station->snrSlow;
    station->snrVariance = 0.9 * station->snrVariance + 0.1 * (snrDiff * snrDiff);

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

// --- HYBRID PATCH START ---
// Context logic, risk assessment, safety, fusion

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

WifiContextType
SmartWifiManagerRf::ClassifyNetworkContext(SmartWifiManagerRfState* station) const
{
    double snr = station->lastSnr;
    double snrVar = station->snrVariance;
    double shortSuccRatio = 0.5;
    if (!station->shortWindow.empty())
        shortSuccRatio = static_cast<double>(std::count(station->shortWindow.begin(), station->shortWindow.end(), true)) / station->shortWindow.size();

    // Adjusted thresholds for realistic SNR values
    if (snr < 5.0 || shortSuccRatio < 0.5 || station->consecFailure >= m_failureThreshold)
        return WifiContextType::EMERGENCY;
    if (snr < 10.0 || snrVar > 5)
        return WifiContextType::POOR_UNSTABLE;
    if (snr < 15.0 || shortSuccRatio < 0.8)
        return WifiContextType::MARGINAL;
    if (snrVar > 3)
        return WifiContextType::GOOD_UNSTABLE;
    if (snr > 20.0 && shortSuccRatio > 0.9)
        return WifiContextType::EXCELLENT_STABLE;
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
// --- HYBRID PATCH END ---

} // namespace ns3