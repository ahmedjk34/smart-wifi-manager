/*
 * Copyright (c) 2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "smart-wifi-manager-xgb.h"

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
#include "ns3/wifi-net-device.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SmartWifiManagerXgb");
NS_OBJECT_ENSURE_REGISTERED(SmartWifiManagerXgb);

TypeId
SmartWifiManagerXgb::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SmartWifiManagerXgb")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<SmartWifiManagerXgb>()
            .AddAttribute("ModelPath",
                          "Path to the XGBoost model file (.joblib)",
                          StringValue("/home/ahmedjk34/ns-allinone-3.41/ns-3.41/step3_xgb_oracle_best_rateIdx_model_FIXED.joblib"),
                          MakeStringAccessor(&SmartWifiManagerXgb::m_modelPath),
                          MakeStringChecker())
            .AddAttribute("ScalerPath",
                          "Path to the scaler file (.joblib)", 
                          StringValue("/home/ahmedjk34/ns-allinone-3.41/ns-3.41/step3_scaler_FIXED.joblib"),
                          MakeStringAccessor(&SmartWifiManagerXgb::m_scalerPath),
                          MakeStringChecker())
            .AddAttribute("PythonScript",
              "Path to Python inference script",
              StringValue("/home/ahmedjk34/ns-allinone-3.41/ns-3.41/python_files/ml_client.py"),
              MakeStringAccessor(&SmartWifiManagerXgb::m_pythonScript),
              MakeStringChecker())
            .AddAttribute("EnableProbabilities",
                          "Enable probability output from ML model",
                          BooleanValue(false),
                          MakeBooleanAccessor(&SmartWifiManagerXgb::m_enableProbabilities),
                          MakeBooleanChecker())
            .AddAttribute("EnableValidation",
                          "Enable feature range validation",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerXgb::m_enableValidation),
                          MakeBooleanChecker())
            .AddAttribute("MaxInferenceTime",
                          "Maximum allowed inference time in ms",
                          UintegerValue(50),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_maxInferenceTime),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("WindowSize",
                          "Window size for success ratio calculation",
                          UintegerValue(50),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_windowSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SnrAlpha",
                          "Alpha parameter for SNR exponential smoothing",
                          DoubleValue(0.1),
                          MakeDoubleAccessor(&SmartWifiManagerXgb::m_snrAlpha),
                          MakeDoubleChecker<double>())
            .AddAttribute("InferencePeriod",
                          "Period between ML inferences (in packets)",
                          UintegerValue(5),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_inferencePeriod),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MinInferenceInterval",
                          "Minimum time between inferences (ms)",
                          UintegerValue(10),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_minInferenceInterval),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("FallbackRate",
                          "Fallback rate index on ML failure",
                          UintegerValue(3),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_fallbackRate),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EnableFallback",
                          "Enable fallback mechanism on ML failure",
                          BooleanValue(true),
                          MakeBooleanAccessor(&SmartWifiManagerXgb::m_enableFallback),
                          MakeBooleanChecker())
            .AddTraceSource("Rate",
                            "Remote station data rate changed",
                            MakeTraceSourceAccessor(&SmartWifiManagerXgb::m_currentRate),
                            "ns3::TracedValueCallback::Uint64")
            .AddTraceSource("MLInferences",
                            "Number of ML inferences made",
                            MakeTraceSourceAccessor(&SmartWifiManagerXgb::m_mlInferences),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("MLFailures",
                            "Number of ML failures",
                            MakeTraceSourceAccessor(&SmartWifiManagerXgb::m_mlFailures),
                            "ns3::TracedValueCallback::Uint32");
    return tid;
}

SmartWifiManagerXgb::SmartWifiManagerXgb()
    : m_lastGlobalInference(Seconds(0)),
      m_globalInferenceCount(0),
      m_currentRate(0),
      m_mlInferences(0),
      m_mlFailures(0)
{
    NS_LOG_FUNCTION(this);
}

SmartWifiManagerXgb::~SmartWifiManagerXgb()
{
    NS_LOG_FUNCTION(this);
}

void
SmartWifiManagerXgb::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    
    // Verify that we're not using HT/VHT modes
    if (GetHtSupported() || GetVhtSupported() || GetHeSupported())
    {
        NS_FATAL_ERROR("SmartWifiManagerXgb does not support HT/VHT/HE modes");
    }
    
    // Verify model files exist
    std::ifstream modelFile(m_modelPath);
    if (!modelFile.good())
    {
        std::cout << "[FATAL] Model file not found: " << m_modelPath << std::endl;
        NS_FATAL_ERROR("XGBoost model file not found: " + m_modelPath);
    }
    
    std::ifstream scalerFile(m_scalerPath);
    if (!scalerFile.good())
    {
        std::cout << "[FATAL] Scaler file not found: " << m_scalerPath << std::endl;
        NS_FATAL_ERROR("Scaler file not found: " + m_scalerPath);
    }
    
    std::ifstream pythonFile(m_pythonScript);
    if (!pythonFile.good())
    {
        std::cout << "[FATAL] Python script not found: " << m_pythonScript << std::endl;
        NS_FATAL_ERROR("Python inference script not found: " + m_pythonScript);
    }
    
    std::cout << "[INFO XGB] SmartWifiManagerXgb initialized successfully" << std::endl;
    std::cout << "[INFO XGB] Model: " << m_modelPath << std::endl;
    std::cout << "[INFO XGB] Scaler: " << m_scalerPath << std::endl;
    std::cout << "[INFO XGB] Script: " << m_pythonScript << std::endl;
    
    WifiRemoteStationManager::DoInitialize();
}

WifiRemoteStation*
SmartWifiManagerXgb::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    
    SmartWifiManagerXgbState* station = new SmartWifiManagerXgbState;
    
    // Initialize SNR with reasonable values
    station->lastSnr = 25.0;  // Start with good SNR
    station->snrFast = 25.0;
    station->snrSlow = 25.0;
    station->snrVariance = 1.0;
    
    // Initialize counters
    station->consecSuccess = 0;
    station->consecFailure = 0;
    station->totalTransmissions = 0;
    
    // Initialize performance metrics
    station->severity = 0.0;
    station->confidence = 1.0;
    
    // Initialize timing counters
    station->T1 = 0;
    station->T2 = 0;
    station->T3 = 0;
    station->retryCount = 0;
    
    // Initialize mobility
    station->mobilityMetric = 0.0;
    station->positionVariance = 0.0;
    station->lastPosition = Vector(0, 0, 0);
    
    // Initialize timing
    Time now = Simulator::Now();
    station->lastUpdateTime = now;
    station->lastInferenceTime = Seconds(0);
    station->lastSuccessTime = now;
    station->lastFailureTime = now;
    
    // Initialize rate management
    station->currentRateIndex = std::min(m_fallbackRate, static_cast<uint32_t>(7)); // Safe initial rate
    station->packetsSinceInference = 0;
    
    // Initialize traffic tracking
    station->bytesTransmitted = 0;
    station->packetsTransmitted = 0;
    station->currentOfferedLoad = 0.0;
    
    // Initialize queue tracking
    station->estimatedQueueLength = 0;
    station->lastQueueUpdate = now;
    
    std::cout << "[INFO XGB] Created new station with initial rate index: " 
              << station->currentRateIndex << std::endl;
    
    return station;
}

void
SmartWifiManagerXgb::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    // RX OK doesn't affect rate adaptation directly in this implementation
}

void
SmartWifiManagerXgb::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerXgb::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    station->retryCount++;
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerXgb::DoReportRtsOk(WifiRemoteStation* st,
                                   double ctsSnr,
                                   WifiMode ctsMode,
                                   double rtsSnr)
{
    NS_LOG_FUNCTION(this << st << ctsSnr << ctsMode << rtsSnr);
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    station->lastSnr = rtsSnr;
    UpdateSnrVariance(st, rtsSnr);
}

void
SmartWifiManagerXgb::DoReportDataOk(WifiRemoteStation* st,
                                    double ackSnr,
                                    WifiMode ackMode,
                                    double dataSnr,
                                    uint16_t dataChannelWidth,
                                    uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << dataNss);
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    station->lastSnr = dataSnr;
    station->retryCount = 0; // Reset retry count on success
    
    // Estimate packet size from data rate and channel width
    uint32_t estimatedPacketSize = 1000; // Default estimate
    UpdateMetrics(st, true, dataSnr, estimatedPacketSize);
    UpdateSnrVariance(st, dataSnr);
}

void
SmartWifiManagerXgb::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    UpdateMetrics(st, false, station->lastSnr);
}

void
SmartWifiManagerXgb::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    UpdateMetrics(st, false, station->lastSnr);
}

WifiTxVector
SmartWifiManagerXgb::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    
    // Get maximum available rate index for 802.11g (0-7)
    uint32_t maxRateIndex = GetNSupported(st) - 1;
    maxRateIndex = std::min(maxRateIndex, static_cast<uint32_t>(7)); // 802.11g max
    
    // Ensure current rate index is valid
    if (station->currentRateIndex > maxRateIndex)
    {
        station->currentRateIndex = std::min(m_fallbackRate, maxRateIndex);
        std::cout << "[WARN XGB] Reset invalid rate index to: " << station->currentRateIndex << std::endl;
    }
    
    // Check if ML inference should be performed
    if (ShouldPerformInference(st))
    {
        std::cout << "[INFO XGB] Starting ML inference for station (packet #" 
                  << station->totalTransmissions << ")" << std::endl;
        
        // Extract features and run ML inference
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);
        
        m_mlInferences++;
        
        if (result.success)
        {
            // Map ML model rate index (0-11) to 802.11g rate index (0-7)
            uint32_t mappedRateIdx = MapMlRateToWifiRate(result.rateIdx, maxRateIndex);
            
            station->currentRateIndex = mappedRateIdx;
            station->lastInferenceTime = Simulator::Now();
            station->packetsSinceInference = 0;
            
            std::cout << "[SUCCESS XGB] ML predicted rate " << result.rateIdx 
                      << " mapped to WiFi rate " << mappedRateIdx 
                      << " (max=" << maxRateIndex << ", latency=" << result.latencyMs << "ms)" << std::endl;
        }
        else
        {
            m_mlFailures++;
            if (m_enableFallback)
            {
                station->currentRateIndex = std::min(m_fallbackRate, maxRateIndex);
            }
            std::cout << "[ERROR XGB] ML inference failed: " << result.error 
                      << ", using fallback rate: " << station->currentRateIndex << std::endl;
        }
    }
    
    // Final safety check
    if (station->currentRateIndex > maxRateIndex)
    {
        station->currentRateIndex = maxRateIndex;
        std::cout << "[EMERGENCY XGB] Rate index corrected to: " << station->currentRateIndex << std::endl;
    }
    
    WifiMode mode = GetSupported(st, station->currentRateIndex);
    uint64_t rate = mode.GetDataRate(allowedWidth);
    
    // Update traced value and increment packet counter
    if (m_currentRate != rate)
    {
        std::cout << "[INFO XGB] Rate changed from " << m_currentRate << " to " << rate 
                  << " (index " << station->currentRateIndex << ")" << std::endl;
        m_currentRate = rate;
    }
    
    station->packetsSinceInference++;
    
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
SmartWifiManagerXgb::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    
    // Use lowest rate for RTS
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
SmartWifiManagerXgb::InferenceResult
SmartWifiManagerXgb::RunMLInference(const std::vector<double>& features) const
{
    NS_LOG_FUNCTION(this);
    
    InferenceResult result;
    result.success = false;
    result.rateIdx = m_fallbackRate;
    result.latencyMs = 0.0;
    
    if (features.size() != 18)
    {
        result.error = "Invalid feature count: " + std::to_string(features.size());
        return result;
    }
    
    // Build command for the fast client
    std::ostringstream cmd;
    cmd << "cd /home/ahmedjk34/ns-allinone-3.41/ns-3.41 && "
        << "/home/ahmedjk34/myenv/bin/python3 " << m_pythonScript
        << " --output-format json --features";
    
    // Add features with proper precision
    for (const auto& feature : features)
    {
        cmd << " " << std::fixed << std::setprecision(3) << feature;
    }
    
    // Redirect stderr to suppress warnings
    cmd << " 2>/dev/null";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute command
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe)
    {
        result.error = "Failed to execute Python command";
        return result;
    }
    
    std::string output;
    char buffer[1024];
    
    // Read output
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        output += buffer;
    }
    
    int status = pclose(pipe);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.latencyMs = duration.count();
    
    if (status != 0)
    {
        result.error = "Python script failed with status: " + std::to_string(status);
        return result;
    }
    
    // Parse JSON output
    try
    {
        size_t json_start = output.find("{");
        size_t json_end = output.rfind("}");
        
        if (json_start == std::string::npos || json_end == std::string::npos)
        {
            result.error = "No valid JSON found in output";
            return result;
        }
        
        std::string json_str = output.substr(json_start, json_end - json_start + 1);
        
        // Extract rateIdx
        size_t rateIdxPos = json_str.find("\"rateIdx\":");
        if (rateIdxPos != std::string::npos)
        {
            size_t valueStart = json_str.find(":", rateIdxPos) + 1;
            size_t valueEnd = json_str.find_first_of(",}", valueStart);
            
            std::string rate_str = json_str.substr(valueStart, valueEnd - valueStart);
            rate_str.erase(0, rate_str.find_first_not_of(" \t\n"));
            rate_str.erase(rate_str.find_last_not_of(" \t\n") + 1);
            
            result.rateIdx = static_cast<uint32_t>(std::stoi(rate_str));
            
            if (result.rateIdx <= 7) // Valid for 802.11g
            {
                result.success = true;
            }
            else
            {
                result.error = "Invalid rate index: " + std::to_string(result.rateIdx);
            }
        }
        else
        {
            result.error = "rateIdx not found in JSON";
        }
    }
    catch (const std::exception& e)
    {
        result.error = "JSON parsing error: " + std::string(e.what());
    }
    
    return result;
}

std::vector<double>
SmartWifiManagerXgb::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    std::vector<double> features(18);
    Time now = Simulator::Now();
    
    // Calculate success ratios with bounds checking
    double shortSuccRatio = 0.5; // Default to moderate success
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
    
    // Calculate timing metrics robustly
    double timeSinceLastTx = (now - station->lastUpdateTime).GetMilliSeconds();
    double timeSinceLastSuccess = (now - station->lastSuccessTime).GetMilliSeconds();
    double timeSinceLastFailure = (now - station->lastFailureTime).GetMilliSeconds();
    
    // Feature extraction with validation
    // [lastSnr, snrFast, snrSlow, shortSuccRatio, medSuccRatio,
    //  consecSuccess, consecFailure, severity, confidence, T1, T2, T3,
    //  offeredLoad, queueLen, retryCount, channelWidth, mobilityMetric, snrVariance]
    
    features[0] = std::max(-20.0, std::min(80.0, station->lastSnr));           // SNR bounds
    features[1] = std::max(-20.0, std::min(80.0, station->snrFast));
    features[2] = std::max(-20.0, std::min(80.0, station->snrSlow));
    features[3] = std::max(0.0, std::min(1.0, shortSuccRatio));                // Ratio bounds
    features[4] = std::max(0.0, std::min(1.0, medSuccRatio));
    features[5] = std::min(100.0, static_cast<double>(station->consecSuccess)); // Reasonable bounds
    features[6] = std::min(100.0, static_cast<double>(station->consecFailure));
    features[7] = std::max(0.0, std::min(1.0, station->severity));
    features[8] = std::max(0.0, std::min(1.0, station->confidence));
    features[9] = std::min(10000.0, timeSinceLastTx);                          // Time bounds (ms)
    features[10] = std::min(10000.0, timeSinceLastSuccess);
    features[11] = std::min(10000.0, timeSinceLastFailure);
    features[12] = GetOfferedLoad(st);                                         // Mbps
    features[13] = static_cast<double>(GetQueueLength(st));                    // Queue length
    features[14] = std::min(10.0, static_cast<double>(station->retryCount));   // Retry bounds
    features[15] = static_cast<double>(GetChannelWidth(st));                   // Channel width
    features[16] = GetMobilityMetric(st);                                      // Mobility 0-1
    features[17] = std::max(0.0, std::min(100.0, station->snrVariance));       // SNR variance bounds
    
    // Debug output for first few inferences
    if (m_globalInferenceCount < 5)
    {
        std::cout << "[DEBUG XGB] Features: ";
        for (size_t i = 0; i < features.size(); ++i)
        {
            std::cout << features[i];
            if (i < features.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    return features;
}

void
SmartWifiManagerXgb::UpdateMetrics(WifiRemoteStation* st, bool success, double snr, uint32_t dataSize)
{
    NS_LOG_FUNCTION(this << st << success << snr << dataSize);
    
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    Time now = Simulator::Now();
    
    // Update basic counters
    station->totalTransmissions++;
    
    // Update SNR with bounds checking
    if (snr > -50.0 && snr < 100.0) // Reasonable SNR bounds
    {
        station->lastSnr = snr;
        
        // Initialize SNR averages if first valid measurement
        if (station->snrFast == 25.0 && station->totalTransmissions == 1)
        {
            station->snrFast = snr;
            station->snrSlow = snr;
        }
        else
        {
            station->snrFast = m_snrAlpha * snr + (1 - m_snrAlpha) * station->snrFast;
            station->snrSlow = (m_snrAlpha / 10) * snr + (1 - m_snrAlpha / 10) * station->snrSlow;
        }
    }
    
    // Update success windows
    station->shortWindow.push_back(success);
    if (station->shortWindow.size() > m_windowSize / 5) // Keep short window small
    {
        station->shortWindow.pop_front();
    }
    
    station->mediumWindow.push_back(success);
    if (station->mediumWindow.size() > m_windowSize)
    {
        station->mediumWindow.pop_front();
    }
    
    // Update consecutive counters
    if (success)
    {
        station->consecSuccess++;
        station->consecFailure = 0;
        station->lastSuccessTime = now;
        
        // Update traffic stats
        if (dataSize > 0)
        {
            station->bytesTransmitted += dataSize;
            station->packetsTransmitted++;
        }
    }
    else
    {
        station->consecFailure++;
        station->consecSuccess = 0;
        station->lastFailureTime = now;
    }
    
    // Update severity and confidence with bounds
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
    
    // Update offered load calculation
    double timeDiff = (now - station->lastUpdateTime).GetSeconds();
    if (timeDiff > 0.001 && station->packetsTransmitted > 0) // At least 1ms
    {
        double recentThroughput = (station->bytesTransmitted * 8.0) / (timeDiff * 1e6); // Mbps
        station->currentOfferedLoad = 0.9 * station->currentOfferedLoad + 0.1 * recentThroughput;
    }
    
    station->lastUpdateTime = now;
}

double
SmartWifiManagerXgb::GetOfferedLoad(WifiRemoteStation* st) const
{
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    
    // Return calculated offered load, bounded to reasonable values
    double offeredLoad = std::max(0.1, std::min(100.0, station->currentOfferedLoad));
    
    // If no recent activity, estimate based on packet rate
    if (offeredLoad < 0.5 && station->totalTransmissions > 0)
    {
        Time now = Simulator::Now();
        double simTime = now.GetSeconds();
        if (simTime > 1.0) // At least 1 second of simulation
        {
            double avgPacketRate = station->totalTransmissions / simTime;
            offeredLoad = std::max(offeredLoad, avgPacketRate * 1.0); // Rough estimate: 1 Mbps per pkt/s
        }
    }
    
    return offeredLoad;
}

double
SmartWifiManagerXgb::GetMobilityMetric(WifiRemoteStation* st) const
{
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    
    // Since we can't access GetNode() from WifiRemoteStation, use SNR-based mobility estimation
    // In a real implementation, you would need access to the node through other means
    
    // Use SNR variance as primary mobility indicator
    double snrMobility = std::tanh(station->snrVariance / 10.0);
    
    // Use consecutive failure patterns as secondary indicator
    double failurePattern = 0.0;
    if (station->totalTransmissions > 10)
    {
        failurePattern = std::min(1.0, static_cast<double>(station->consecFailure) / 10.0);
    }
    
    // Combine metrics
    station->mobilityMetric = 0.8 * snrMobility + 0.2 * failurePattern;
    
    return std::max(0.0, std::min(1.0, station->mobilityMetric));
}

uint32_t
SmartWifiManagerXgb::GetQueueLength(WifiRemoteStation* st) const
{
    // Since we can't easily access the MAC queue, estimate based on performance metrics
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    
    // Estimate based on retry count and consecutive failures
    uint32_t estimated = station->retryCount + (station->consecFailure / 2);
    
    // Factor in severity (higher severity suggests more queuing)
    estimated += static_cast<uint32_t>(station->severity * 5);
    
    return std::min(estimated, static_cast<uint32_t>(20)); // Reasonable upper bound
}

uint32_t
SmartWifiManagerXgb::MapMlRateToWifiRate(uint32_t mlRateIdx, uint32_t maxRateIdx) const
{
    // ML model was trained on 12 rate indices (0-11)
    // 802.11g has 8 rates (0-7)
    // Map intelligently based on performance characteristics
    
    // Mapping table: ML rate -> 802.11g rate
    static const uint32_t rateMapping[12] = {
        0, // ML 0 -> WiFi 0 (1 Mbps)
        1, // ML 1 -> WiFi 1 (2 Mbps)
        2, // ML 2 -> WiFi 2 (5.5 Mbps)
        3, // ML 3 -> WiFi 3 (11 Mbps)
        4, // ML 4 -> WiFi 4 (6 Mbps)
        5, // ML 5 -> WiFi 5 (9 Mbps)
        6, // ML 6 -> WiFi 6 (12 Mbps)
        7, // ML 7 -> WiFi 7 (18 Mbps)
        7, // ML 8 -> WiFi 7 (cap at highest)
        7, // ML 9 -> WiFi 7 (cap at highest)
        7, // ML 10 -> WiFi 7 (cap at highest)
        7  // ML 11 -> WiFi 7 (cap at highest)
    };
    
    if (mlRateIdx >= 12)
    {
        return std::min(m_fallbackRate, maxRateIdx);
    }
    
    uint32_t mappedRate = rateMapping[mlRateIdx];
    return std::min(mappedRate, maxRateIdx);
}

bool
SmartWifiManagerXgb::ShouldPerformInference(WifiRemoteStation* st) const
{
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    Time now = Simulator::Now();
    
    // Check packet-based period
    if (station->packetsSinceInference < m_inferencePeriod)
    {
        return false;
    }
    
    // Check minimum time interval
    if ((now - station->lastInferenceTime).GetMilliSeconds() < m_minInferenceInterval)
    {
        return false;
    }
    
    // Global rate limiting to prevent system overload
    if ((now - m_lastGlobalInference).GetMilliSeconds() < (m_minInferenceInterval / 2))
    {
        return false;
    }
    
    // Allow inference
    m_lastGlobalInference = now;
    m_globalInferenceCount++;
    return true;
}

void
SmartWifiManagerXgb::UpdateSnrVariance(WifiRemoteStation* st, double snr)
{
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    
    // Add to SNR history
    station->snrHistory.push_back(snr);
    if (station->snrHistory.size() > 20) // Keep last 20 measurements
    {
        station->snrHistory.pop_front();
    }
    
    // Calculate variance if we have enough samples
    if (station->snrHistory.size() >= 3)
    {
        double sum = 0.0;
        for (double s : station->snrHistory)
        {
            sum += s;
        }
        double mean = sum / station->snrHistory.size();
        
        double variance = 0.0;
        for (double s : station->snrHistory)
        {
            variance += (s - mean) * (s - mean);
        }
        variance /= station->snrHistory.size();
        
        station->snrVariance = variance;
    }
}

} // namespace ns3