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
#include <fstream>
#include <iomanip>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

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
    if (GetHtSupported() || GetVhtSupported() || GetHeSupported())
    {
        NS_FATAL_ERROR("SmartWifiManagerRf does not support HT/VHT/HE modes");
    }

    // Verify model/scaler exist
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

    WifiRemoteStationManager::DoInitialize();
}

WifiRemoteStation*
SmartWifiManagerRf::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);

    SmartWifiManagerRfState* station = new SmartWifiManagerRfState;

    station->lastSnr = 25.0;
    station->snrFast = 25.0;
    station->snrSlow = 25.0;
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

    std::cout << "[INFO RF] Created new station with initial rate index: "
              << station->currentRateIndex << std::endl;

    return station;
}

void
SmartWifiManagerRf::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
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
    station->retryCount = 0;
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

    uint32_t maxRateIndex = GetNSupported(st) - 1;
    maxRateIndex = std::min(maxRateIndex, static_cast<uint32_t>(7));

    if (station->currentRateIndex > maxRateIndex)
    {
        station->currentRateIndex = std::min(m_fallbackRate, maxRateIndex);
        std::cout << "[WARN RF] Reset invalid rate index to fallback: " << station->currentRateIndex << std::endl;
    }

    if ((Simulator::Now() - station->lastInferenceTime).GetMilliSeconds() > m_inferencePeriod)
    {
        std::vector<double> features = ExtractFeatures(st);
        InferenceResult result = RunMLInference(features);

        m_mlInferences++;

        if (result.success)
        {
            uint32_t rateIdx = result.rateIdx <= maxRateIndex ? result.rateIdx : std::min(m_fallbackRate, maxRateIndex);
            station->currentRateIndex = rateIdx;
            station->lastInferenceTime = Simulator::Now();
            std::cout << "[SUCCESS RF] ML predicted rate index: " << result.rateIdx
                      << " (mapped to WiFi: " << rateIdx << ", max=" << maxRateIndex << ")" << std::endl;
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

    if (station->currentRateIndex > maxRateIndex)
    {
        station->currentRateIndex = maxRateIndex;
        std::cout << "[EMERGENCY RF] Rate index correction to: " << station->currentRateIndex << std::endl;
    }

    WifiMode mode = GetSupported(st, station->currentRateIndex);
    uint64_t rate = mode.GetDataRate(allowedWidth);

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

// --- Client/server ML inference implementation ---
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

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        result.error = "Failed to create socket";
        return result;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(m_inferenceServerPort);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // Localhost

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
    std::vector<double> features(18);

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

    features[0] = std::max(-20.0, std::min(80.0, station->lastSnr));
    features[1] = std::max(-20.0, std::min(80.0, station->snrFast));
    features[2] = std::max(-20.0, std::min(80.0, station->snrSlow));
    features[3] = std::max(0.0, std::min(1.0, shortSuccRatio));
    features[4] = std::max(0.0, std::min(1.0, medSuccRatio));
    features[5] = std::min(100.0, static_cast<double>(station->consecSuccess));
    features[6] = std::min(100.0, static_cast<double>(station->consecFailure));
    features[7] = std::max(0.0, std::min(1.0, station->severity));
    features[8] = std::max(0.0, std::min(1.0, station->confidence));
    features[9] = static_cast<double>(station->T1);
    features[10] = static_cast<double>(station->T2);
    features[11] = static_cast<double>(station->T3);
    features[12] = GetOfferedLoad();
    features[13] = static_cast<double>(station->queueLength);
    features[14] = static_cast<double>(station->retryCount);
    features[15] = 20.0;
    features[16] = GetMobilityMetric(st);
    features[17] = std::max(0.0, std::min(100.0, station->snrVariance));

    
    // --- DEBUG PRINT: Print features before sending to ML server ---
    std::cout << "[DEBUG RF] Features sent to ML: ";
    for (size_t i = 0; i < features.size(); ++i) {
        std::cout << features[i] << " ";
    }
    std::cout << std::endl;
    // --------------------------------------------------------------


    return features;
}

void
SmartWifiManagerRf::UpdateMetrics(WifiRemoteStation* st, bool success, double snr)
{
    NS_LOG_FUNCTION(this << st << success << snr);

    SmartWifiManagerRfState* station = static_cast<SmartWifiManagerRfState*>(st);
    Time now = Simulator::Now();

    if (snr > -50.0 && snr < 100.0)
    {
        station->lastSnr = snr;
        if (station->snrFast == 25.0)
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

} // namespace ns3