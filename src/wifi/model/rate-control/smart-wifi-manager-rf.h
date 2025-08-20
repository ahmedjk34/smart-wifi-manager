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
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    SmartWifiManagerRf();
    ~SmartWifiManagerRf() override;

    /**
     * \brief Structure to hold ML inference results
     */
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

    /**
     * \brief Execute Python ML inference
     * \param features The 18 feature values for inference
     * \return InferenceResult containing prediction and metadata
     */
    InferenceResult RunMLInference(const std::vector<double>& features) const;

    /**
     * \brief Extract features for ML inference
     * \param station The wifi remote station
     * \return Vector of 18 feature values
     */
    std::vector<double> ExtractFeatures(WifiRemoteStation* station) const;

    /**
     * \brief Update performance metrics for a station
     * \param station The wifi remote station
     * \param success Whether the transmission was successful
     * \param snr The signal-to-noise ratio
     */
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr);

    /**
     * \brief Get current offered load estimate
     * \return Estimated offered load in Mbps
     */
    double GetOfferedLoad() const;

    /**
     * \brief Get mobility metric for station
     * \param station The wifi remote station
     * \return Mobility metric (0-1 scale)
     */
    double GetMobilityMetric(WifiRemoteStation* station) const;

    // Configuration attributes
    std::string m_modelPath;          //!< Path to Random Forest model file
    std::string m_scalerPath;         //!< Path to scaler file
    std::string m_pythonScript;       //!< Path to Python inference script
    std::string m_modelType;          //!< Model type (oracle or v3)
    bool m_enableProbabilities;       //!< Whether to request probabilities
    bool m_enableValidation;          //!< Whether to enable range validation
    uint32_t m_maxInferenceTime;      //!< Max inference time in ms
    
    // Performance tracking
    uint32_t m_windowSize;            //!< Window size for success ratio calculation
    double m_snrAlpha;                //!< Alpha for SNR smoothing
    uint32_t m_inferencePeriod;       //!< Period between ML inferences
    
    // Fallback parameters
    uint32_t m_fallbackRate;          //!< Fallback rate index on ML failure
    bool m_enableFallback;            //!< Whether to use fallback on ML failure
    
    TracedValue<uint64_t> m_currentRate;    //!< Current data rate
    TracedValue<uint32_t> m_mlInferences;   //!< Number of ML inferences made
    TracedValue<uint32_t> m_mlFailures;     //!< Number of ML failures
};

/**
 * \brief SmartWifiManagerRf station state
 */
struct SmartWifiManagerRfState : public WifiRemoteStation
{
    double lastSnr;                   //!< Last reported SNR
    double snrFast;                   //!< Fast SNR average
    double snrSlow;                   //!< Slow SNR average
    
    std::deque<bool> shortWindow;     //!< Short-term success window
    std::deque<bool> mediumWindow;    //!< Medium-term success window
    
    uint32_t consecSuccess;           //!< Consecutive successes
    uint32_t consecFailure;           //!< Consecutive failures
    
    double severity;                  //!< Failure severity
    double confidence;                //!< Confidence in current rate
    
    uint32_t T1, T2, T3;             //!< Timing counters
    uint32_t retryCount;              //!< Current retry count
    
    double mobilityMetric;            //!< Mobility estimation
    double snrVariance;               //!< SNR variance
    
    Time lastUpdateTime;              //!< Last metrics update time
    Time lastInferenceTime;           //!< Last ML inference time
    Vector lastPosition;              //!< Last known position
    
    uint32_t currentRateIndex;        //!< Current rate index
    uint32_t queueLength;             //!< Estimated queue length
};

} // namespace ns3

#endif /* SMART_WIFI_MANAGER_RF_H */