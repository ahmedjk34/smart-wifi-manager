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

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <cmath>

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
                          StringValue("step3_xgb_oracle_best_rateldx_model.joblib"),
                          MakeStringAccessor(&SmartWifiManagerXgb::m_modelPath),
                          MakeStringChecker())
            .AddAttribute("ScalerPath",
                          "Path to the scaler file (.joblib)", 
                          StringValue("step3_scaler.joblib"),
                          MakeStringAccessor(&SmartWifiManagerXgb::m_scalerPath),
                          MakeStringChecker())
            .AddAttribute("PythonScript",
                          "Path to Python inference script",
                          StringValue("python_files/ml_rate_inference.py"),
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
                          UintegerValue(100),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_maxInferenceTime),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("WindowSize",
                          "Window size for success ratio calculation",
                          UintegerValue(20),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_windowSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SnrAlpha",
                          "Alpha parameter for SNR exponential smoothing",
                          DoubleValue(0.1),
                          MakeDoubleAccessor(&SmartWifiManagerXgb::m_snrAlpha),
                          MakeDoubleChecker<double>())
            .AddAttribute("InferencePeriod",
                          "Period between ML inferences (in transmissions)",
                          UintegerValue(10),
                          MakeUintegerAccessor(&SmartWifiManagerXgb::m_inferencePeriod),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("FallbackRate",
                          "Fallback rate index on ML failure",
                          UintegerValue(2),
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
    : m_currentRate(0),
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
    
    WifiRemoteStationManager::DoInitialize();
}

WifiRemoteStation*
SmartWifiManagerXgb::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    
    SmartWifiManagerXgbState* station = new SmartWifiManagerXgbState;
    
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
    UpdateMetrics(st, true, dataSnr);
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
    
    // Check if it's time for ML inference
    Time now = Simulator::Now();
    if ((now - station->lastInferenceTime).GetMilliSeconds() > m_inferencePeriod ||
        station->lastInferenceTime == Seconds(0))
    {
        // Extract features and run ML inference
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);
        
        m_mlInferences++;
        
        if (result.success && result.rateIdx < GetNSupported(st))
        {
            station->currentRateIndex = result.rateIdx;
            station->lastInferenceTime = now;
            NS_LOG_DEBUG("ML predicted rate index: " << result.rateIdx 
                        << " latency: " << result.latencyMs << "ms");
        }
        else
        {
            m_mlFailures++;
            if (m_enableFallback)
            {
                station->currentRateIndex = std::min(m_fallbackRate, static_cast<uint32_t>(GetNSupported(st) - 1));
            }
            NS_LOG_WARN("ML inference failed: " << result.error 
                       << " using fallback rate: " << station->currentRateIndex);
        }
    }
    
    WifiMode mode = GetSupported(st, station->currentRateIndex);
    uint64_t rate = mode.GetDataRate(allowedWidth);
    
    if (m_currentRate != rate)
    {
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
    
    // Build command
    std::ostringstream cmd;
    cmd << "python3 " << m_pythonScript
        << " --model " << m_modelPath
        << " --scaler " << m_scalerPath
        << " --features";
    
    for (const auto& feature : features)
    {
        cmd << " " << feature;
    }
    
    cmd << " --output-format json";
    
    if (m_enableProbabilities)
    {
        cmd << " --probabilities";
    }
    
    if (m_enableValidation)
    {
        cmd << " --validate-range";
    }
    
    // Execute command
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe)
    {
        result.error = "Failed to execute Python command";
        return result;
    }
    
    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        output += buffer;
    }
    
    int status = pclose(pipe);
    
    if (status != 0)
    {
        result.error = "Python script failed with status: " + std::to_string(status);
        return result;
    }
    
    // Parse JSON output (simplified parsing)
    size_t rateIdxPos = output.find("\"rateIdx\":");
    size_t latencyPos = output.find("\"latencyMs\":");
    
    if (rateIdxPos != std::string::npos && latencyPos != std::string::npos)
    {
        // Extract rateIdx
        size_t start = output.find(":", rateIdxPos) + 1;
        size_t end = output.find(",", start);
        if (end == std::string::npos) end = output.find("}", start);
        
        try
        {
            result.rateIdx = std::stoi(output.substr(start, end - start));
        }
        catch (const std::exception& e)
        {
            result.error = "Failed to parse rateIdx from JSON";
            return result;
        }
        
        // Extract latency
        start = output.find(":", latencyPos) + 1;
        end = output.find(",", start);
        if (end == std::string::npos) end = output.find("}", start);
        
        try
        {
            result.latencyMs = std::stod(output.substr(start, end - start));
        }
        catch (const std::exception& e)
        {
            result.latencyMs = 0.0; // Non-critical
        }
        
        result.success = true;
    }
    else
    {
        result.error = "Failed to parse JSON output";
    }
    
    return result;
}

std::vector<double>
SmartWifiManagerXgb::ExtractFeatures(WifiRemoteStation* st) const
{
    NS_LOG_FUNCTION(this << st);
    
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
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
SmartWifiManagerXgb::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);
    
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
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
SmartWifiManagerXgb::GetOfferedLoad() const
{
    // Simplified offered load estimation
    // In a real implementation, this would track actual traffic
    return 10.0; // Mbps
}

double
SmartWifiManagerXgb::GetMobilityMetric(WifiRemoteStation* st) const
{
    SmartWifiManagerXgbState* station = static_cast<SmartWifiManagerXgbState*>(st);
    
    // Simple mobility metric based on SNR variance as a proxy for mobility
    // Higher SNR variance often indicates mobility due to changing channel conditions
    double normalizedVariance = std::tanh(station->snrVariance / 10.0);
    station->mobilityMetric = normalizedVariance;
    
    return station->mobilityMetric;
}

} // namespace ns3