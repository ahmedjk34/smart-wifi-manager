/*
 * Copyright (c) 2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

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
                          StringValue("step3_rf_oracle_best_rateldx_model.joblib"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_modelPath),
                          MakeStringChecker())
            .AddAttribute("ScalerPath",
                          "Path to the scaler file (.joblib)", 
                          StringValue("step3_scaler.joblib"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_scalerPath),
                          MakeStringChecker())
            .AddAttribute("PythonScript",
                          "Path to Python inference script",
                          StringValue("python_files/ml_rate_inference.py"),
                          MakeStringAccessor(&SmartWifiManagerRf::m_pythonScript),
                          MakeStringChecker())
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
      m_mlFailures(0)
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
    
    // Verify that we're not using HT/VHT modes
    if (GetHtSupported() || GetVhtSupported() || GetHeSupported())
    {
        NS_FATAL_ERROR("SmartWifiManagerRf does not support HT/VHT/HE modes");
    }
    
    WifiRemoteStationManager::DoInitialize();
}

WifiRemoteStation*
SmartWifiManagerRf::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    
    SmartWifiManagerRfState* station = new SmartWifiManagerRfState;
    
    // Initialize state
    station->lastSnr = 0.0;
    station->snrFast = 0.0;
    station->snrSlow = 0.0;
    station->consecSuccess = 0;
    station->consecFailure = 0;
    station->severity = 0.0;
    station->confidence = 1.0;
    station->T1 = 0;
    station->T2 = 0;
    station->T3 = 0;
    station->retryCount = 0;
    station->mobilityMetric = 0.0;
    station->snrVariance = 0.0;
    station->lastUpdateTime = Simulator::Now();
    station->lastInferenceTime = Seconds(0);
    station->lastPosition = Vector(0, 0, 0);
    station->currentRateIndex = m_fallbackRate;
    station->queueLength = 0;
    
    return station;
}

void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    // RX OK doesn't affect rate adaptation directly in this implementation
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
    station->lastSnr = rtsSnr;
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
    station->lastSnr = dataSnr;
    station->retryCount = 0; // Reset retry count on success
    UpdateMetrics(st, true, dataSnr);
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
    
    // Get maximum available rate index
    uint32_t maxRateIndex = GetNSupported(st) - 1;
    
    // Ensure current rate index is valid
    if (station->currentRateIndex > maxRateIndex)
    {
        station->currentRateIndex = std::min(m_fallbackRate, maxRateIndex);
        std::cout << "[WARN RF] Reset invalid rate index to fallback: " << station->currentRateIndex << std::endl;
    }
    
    // Check if it's time for ML inference (but limit frequency to prevent hanging)
    Time now = Simulator::Now();
    static Time lastGlobalInference = Seconds(0);
    
    // Only do ML inference every 100ms globally to prevent overload
    if ((now - lastGlobalInference).GetMilliSeconds() > 100 &&
        (now - station->lastInferenceTime).GetMilliSeconds() > m_inferencePeriod)
    {
        lastGlobalInference = now;
        
        std::cout << "[INFO RF] Starting ML inference at " << now.GetSeconds() << "s" << std::endl;
        
        // Extract features and run ML inference
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);
        
        m_mlInferences++;
        
        if (result.success)
        {
            // Validate rate index bounds
            if (result.rateIdx <= maxRateIndex)
            {
                station->currentRateIndex = result.rateIdx;
                station->lastInferenceTime = now;
                std::cout << "[SUCCESS RF] ML predicted rate index: " << result.rateIdx 
                          << " (max available: " << maxRateIndex << ")" << std::endl;
            }
            else
            {
                // Rate index out of bounds, use fallback
                m_mlFailures++;
                station->currentRateIndex = std::min(m_fallbackRate, maxRateIndex);
                std::cout << "[WARN RF] ML predicted invalid rate index " << result.rateIdx 
                          << " (max=" << maxRateIndex << "), using fallback: " << station->currentRateIndex << std::endl;
            }
        }
        else
        {
            m_mlFailures++;
            if (m_enableFallback)
            {
                station->currentRateIndex = std::min(m_fallbackRate, maxRateIndex);
            }
            std::cout << "[ERROR RF] ML inference failed: " << result.error 
                      << ", using fallback rate: " << station->currentRateIndex << std::endl;
        }
    }
    
    // Final safety check
    if (station->currentRateIndex > maxRateIndex)
    {
        station->currentRateIndex = maxRateIndex;
        std::cout << "[EMERGENCY RF] Rate index correction to: " << station->currentRateIndex << std::endl;
    }
    
    WifiMode mode = GetSupported(st, station->currentRateIndex);
    uint64_t rate = mode.GetDataRate(allowedWidth);
    
    // Update traced value
    if (m_currentRate != rate)
    {
        std::cout << "[INFO RF] Rate changed from " << m_currentRate << " to " << rate 
                  << " (index " << station->currentRateIndex << ")" << std::endl;
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

SmartWifiManagerRf::InferenceResult
SmartWifiManagerRf::RunMLInference(const std::vector<double>& features) const
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
    
    // Build command with timeout and consistent format
    std::ostringstream cmd;
    cmd << "timeout 3s /home/ahmedjk34/myenv/bin/python3 " << m_pythonScript
        << " --model " << m_modelPath
        << " --scaler " << m_scalerPath
        << " --output-format json"  // Use JSON consistently
        << " --validate-range"
        // << " --model-type rf"       // Specify Random Forest model type
        << " --features";
    
    for (const auto& feature : features)
    {
        cmd << " " << feature;
    }
    
    if (m_enableProbabilities)
    {
        cmd << " --probabilities";
    }
    
    // Redirect stderr to stdout to capture all output
    cmd << " 2>&1";
    
    // Debug output
    std::cout << "[DEBUG RF] Executing: " << cmd.str() << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute command
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe)
    {
        result.error = "Failed to execute Python command";
        std::cout << "[ERROR RF] " << result.error << std::endl;
        return result;
    }
    
    std::string output;
    char buffer[1024];
    
    // Read with timeout simulation (basic approach)
    int chars_read = 0;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        output += buffer;
        chars_read++;
        
        // Simple protection against infinite loops
        if (chars_read > 100) {
            std::cout << "[WARNING RF] Too much output, breaking" << std::endl;
            break;
        }
    }
    
    int status = pclose(pipe);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "[DEBUG RF] Command completed in " << duration.count() 
              << "ms with status: " << status << std::endl;
    std::cout << "[DEBUG RF] Raw output: '" << output << "'" << std::endl;
    
    if (status != 0)
    {
        result.error = "Python script failed with status: " + std::to_string(status) + ", output: " + output;
        std::cout << "[ERROR RF] " << result.error << std::endl;
        return result;
    }
    
    // Try to parse JSON output
    try
    {
        // Look for JSON structure
        size_t json_start = output.find("{");
        size_t json_end = output.rfind("}");
        
        if (json_start == std::string::npos || json_end == std::string::npos)
        {
            result.error = "No JSON found in output: " + output;
            std::cout << "[ERROR RF] " << result.error << std::endl;
            return result;
        }
        
        std::string json_str = output.substr(json_start, json_end - json_start + 1);
        std::cout << "[DEBUG RF] Extracted JSON: " << json_str << std::endl;
        
        // Parse JSON manually (simple approach)
        size_t rateIdxPos = json_str.find("\"rateIdx\":");
        
        if (rateIdxPos != std::string::npos)
        {
            // Extract rateIdx
            size_t start = json_str.find(":", rateIdxPos) + 1;
            size_t end = json_str.find(",", start);
            if (end == std::string::npos) end = json_str.find("}", start);
            
            std::string rate_str = json_str.substr(start, end - start);
            // Remove whitespace
            rate_str.erase(0, rate_str.find_first_not_of(" \t"));
            rate_str.erase(rate_str.find_last_not_of(" \t") + 1);
            
            result.rateIdx = std::stoi(rate_str);
            result.latencyMs = duration.count();
            result.success = true;
            
            std::cout << "[SUCCESS RF] Parsed rate index: " << result.rateIdx << std::endl;
        }
        else
        {
            result.error = "Could not find rateIdx in JSON: " + json_str;
            std::cout << "[ERROR RF] " << result.error << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        result.error = "JSON parsing failed: " + std::string(e.what()) + ", output: " + output;
        std::cout << "[ERROR RF] " << result.error << std::endl;
    }
    
    return result;
}

std::vector<double>
SmartWifiManagerRf::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    std::vector<double> features(18);
    
    // Calculate success ratios
    double shortSuccRatio = 0.0;
    double medSuccRatio = 0.0;
    
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
    
    // Feature order: [lastSnr, snrFast, snrSlow, shortSuccRatio, medSuccRatio,
    //                 consecSuccess, consecFailure, severity, confidence, T1, T2, T3,
    //                 offeredLoad, queueLen, retryCount, channelWidth, mobilityMetric, snrVariance]
    
    features[0] = station->lastSnr;
    features[1] = station->snrFast;
    features[2] = station->snrSlow;
    features[3] = shortSuccRatio;
    features[4] = medSuccRatio;
    features[5] = static_cast<double>(station->consecSuccess);
    features[6] = static_cast<double>(station->consecFailure);
    features[7] = station->severity;
    features[8] = station->confidence;
    features[9] = static_cast<double>(station->T1);
    features[10] = static_cast<double>(station->T2);
    features[11] = static_cast<double>(station->T3);
    features[12] = GetOfferedLoad();
    features[13] = static_cast<double>(station->queueLength);
    features[14] = static_cast<double>(station->retryCount);
    features[15] = static_cast<double>(GetChannelWidth(st));
    features[16] = GetMobilityMetric(st);
    features[17] = station->snrVariance;
    
    return features;
}

void
SmartWifiManagerRf::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);
    
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();
    
    // Update SNR smoothing
    if (station->snrFast == 0.0)
    {
        station->snrFast = snr;
        station->snrSlow = snr;
    }
    else
    {
        station->snrFast = m_snrAlpha * snr + (1 - m_snrAlpha) * station->snrFast;
        station->snrSlow = (m_snrAlpha / 10) * snr + (1 - m_snrAlpha / 10) * station->snrSlow;
    }
    
    // Update SNR variance
    double snrDiff = snr - station->snrSlow;
    station->snrVariance = 0.9 * station->snrVariance + 0.1 * (snrDiff * snrDiff);
    
    // Update success windows
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
    
    // Update consecutive counters
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
    
    // Update severity and confidence
    if (!success)
    {
        station->severity = std::min(1.0, station->severity + 0.1);
    }
    else
    {
        station->severity = std::max(0.0, station->severity - 0.05);
    }
    
    station->confidence = std::max(0.1, std::min(1.0, 
        station->confidence + (success ? 0.05 : -0.1)));
    
    // Update timing counters (simplified)
    double timeDiff = (now - station->lastUpdateTime).GetSeconds();
    station->T1 = static_cast<uint32_t>(timeDiff * 1000); // ms
    station->T2 = station->T1 * 2;
    station->T3 = station->T1 * 3;
    
    station->lastUpdateTime = now;
}

double
SmartWifiManagerRf::GetOfferedLoad() const
{
    // Simplified offered load estimation
    // In a real implementation, this would track actual traffic
    return 10.0; // Mbps
}

double
SmartWifiManagerRf::GetMobilityMetric(WifiRemoteStation* st) const
{
    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    
    // Simple mobility metric based on SNR variance as a proxy for mobility
    // Higher SNR variance often indicates mobility due to changing channel conditions
    double normalizedVariance = std::tanh(station->snrVariance / 10.0);
    station->mobilityMetric = normalizedVariance;
    
    return station->mobilityMetric;
}

} // namespace ns3