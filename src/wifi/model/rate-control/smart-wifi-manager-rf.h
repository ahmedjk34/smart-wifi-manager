/*
 * Copyright (c) 2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef SMART_WIFI_MANAGER_RF_H
#define SMART_WIFI_MANAGER_RF_H

#include "ns3/traced-value.h"
#include "ns3/wifi-remote-station-manager.h"
#include "ns3/vector.h"
#include "ns3/node.h"
#include "ns3/mobility-model.h"
#include <string>
#include <chrono>
#include <deque>

namespace ns3
{

/**
 * \brief Smart Rate control algorithm using Random Forest ML models
 * \ingroup wifi
 *
 * This class implements the Smart rate control algorithm using Random Forest models.
 * It uses Python inference script to predict optimal rate indices based on
 * network conditions and performance metrics.
 * This RAA does not support HT modes and will error
 * exit if the user tries to configure this RAA with a Wi-Fi MAC
 * that supports 802.11n or higher.
 */
class SmartWifiManagerRf : public WifiRemoteStationManager
{
  public:
    static TypeId GetTypeId();
    SmartWifiManagerRf();
    ~SmartWifiManagerRf() override;

    struct InferenceResult
    {
        uint32_t rateIdx;
        double latencyMs;
        bool success;
        std::string error;
    };

  private:
    void DoInitialize() override;
    WifiRemoteStation* DoCreateStation() const override;
    void DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode) override;
    void DoReportRtsFailed(WifiRemoteStation* station) override;
    void DoReportDataFailed(WifiRemoteStation* station) override;
    void DoReportRtsOk(WifiRemoteStation* station,
                       double ctsSnr,
                       WifiMode ctsMode,
                       double rtsSnr) override;
    void DoReportDataOk(WifiRemoteStation* station,
                        double ackSnr,
                        WifiMode ackMode,
                        double dataSnr,
                        uint16_t dataChannelWidth,
                        uint8_t dataNss) override;
    void DoReportFinalRtsFailed(WifiRemoteStation* station) override;
    void DoReportFinalDataFailed(WifiRemoteStation* station) override;
    WifiTxVector DoGetDataTxVector(WifiRemoteStation* station, uint16_t allowedWidth) override;
    WifiTxVector DoGetRtsTxVector(WifiRemoteStation* station) override;

    InferenceResult RunMLInference(const std::vector<double>& features) const;
    std::vector<double> ExtractFeatures(WifiRemoteStation* station) const;
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr);
    double GetOfferedLoad() const;
    double GetMobilityMetric(WifiRemoteStation* station) const;

    std::string m_modelPath;
    std::string m_scalerPath;
    std::string m_pythonScript;
    std::string m_modelType;
    bool m_enableProbabilities;
    bool m_enableValidation;
    uint32_t m_maxInferenceTime;
    uint32_t m_windowSize;
    double m_snrAlpha;
    uint32_t m_inferencePeriod;
    uint32_t m_fallbackRate;
    bool m_enableFallback;

    // --- Add this missing member for the server port ---
    uint16_t m_inferenceServerPort;

    TracedValue<uint64_t> m_currentRate;
    TracedValue<uint32_t> m_mlInferences;
    TracedValue<uint32_t> m_mlFailures;
};

/**
 * \brief SmartWifiManagerRf station state
 */
struct SmartWifiManagerRfState : public WifiRemoteStation
{
    double lastSnr;
    double snrFast;
    double snrSlow;
    std::deque<bool> shortWindow;
    std::deque<bool> mediumWindow;
    uint32_t consecSuccess;
    uint32_t consecFailure;
    double severity;
    double confidence;
    uint32_t T1, T2, T3;
    uint32_t retryCount;
    double mobilityMetric;
    double snrVariance;
    Time lastUpdateTime;
    Time lastInferenceTime;
    Vector lastPosition;
    uint32_t currentRateIndex;
    uint32_t queueLength;
};

} // namespace ns3

#endif /* SMART_WIFI_MANAGER_RF_H */