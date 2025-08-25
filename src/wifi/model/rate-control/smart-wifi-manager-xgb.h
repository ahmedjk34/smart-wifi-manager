/*
 * Copyright (c) 2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef SMART_WIFI_MANAGER_XGB_H
#define SMART_WIFI_MANAGER_XGB_H

#include "ns3/traced-value.h"
#include "ns3/wifi-remote-station-manager.h"
#include "ns3/vector.h"
#include "ns3/node.h"
#include "ns3/mobility-model.h"
#include <string>
#include <chrono>
#include <deque>
#include <map>

namespace ns3
{

/**
 * \brief Smart Rate control algorithm using XGBoost ML models
 * \ingroup wifi
 *
 * This class implements the Smart rate control algorithm using XGBoost models.
 * It uses Python inference script to predict optimal rate indices based on
 * network conditions and performance metrics.
 * This RAA does not support HT modes and will error
 * exit if the user tries to configure this RAA with a Wi-Fi MAC
 * that supports 802.11n or higher.
 */
class SmartWifiManagerXgb : public WifiRemoteStationManager
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    SmartWifiManagerXgb();
    ~SmartWifiManagerXgb() override;

    /**
     * \brief Structure to hold ML inference results
     */
    struct InferenceResult
    {
        uint32_t rateIdx;
        double latencyMs;
        bool success;
        std::string error;
        std::vector<double> probabilities;
    };

    /**
     * \brief Structure to track traffic statistics
     */
    struct TrafficStats
    {
        uint64_t totalBytes;
        uint32_t totalPackets;
        Time lastUpdate;
        double currentThroughput;
        
        TrafficStats() : totalBytes(0), totalPackets(0), lastUpdate(Seconds(0)), currentThroughput(0.0) {}
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
     * \param dataSize Size of transmitted data
     */
    void UpdateMetrics(WifiRemoteStation* station, bool success, double snr, uint32_t dataSize = 0);

    /**
     * \brief Get current offered load estimate
     * \param station The wifi remote station
     * \return Estimated offered load in Mbps
     */
    double GetOfferedLoad(WifiRemoteStation* station) const;

    /**
     * \brief Get mobility metric for station
     * \param station The wifi remote station
     * \return Mobility metric (0-1 scale)
     */
    double GetMobilityMetric(WifiRemoteStation* station) const;

    /**
     * \brief Get queue length estimate
     * \param station The wifi remote station
     * \return Estimated queue length
     */
    uint32_t GetQueueLength(WifiRemoteStation* station) const;

    /**
     * \brief Map ML model rate index to 802.11g rate index
     * \param mlRateIdx Rate index from ML model (0-11)
     * \return Valid 802.11g rate index (0-7)
     */
    uint32_t MapMlRateToWifiRate(uint32_t mlRateIdx, uint32_t maxRateIdx) const;

    /**
     * \brief Check if ML inference should be performed
     * \param station The wifi remote station
     * \return True if inference should be done
     */
    bool ShouldPerformInference(WifiRemoteStation* station) const;

    /**
     * \brief Update SNR variance calculation
     * \param station The wifi remote station
     * \param snr Current SNR value
     */
    void UpdateSnrVariance(WifiRemoteStation* station, double snr);

    // Configuration attributes
    std::string m_modelPath;          //!< Path to XGBoost model file
    std::string m_scalerPath;         //!< Path to scaler file
    std::string m_pythonScript;       //!< Path to Python inference script
    bool m_enableProbabilities;       //!< Whether to request probabilities
    bool m_enableValidation;          //!< Whether to enable range validation
    uint32_t m_maxInferenceTime;      //!< Max inference time in ms
    
    // Performance tracking
    uint32_t m_windowSize;            //!< Window size for success ratio calculation
    double m_snrAlpha;                //!< Alpha for SNR smoothing
    uint32_t m_inferencePeriod;       //!< Period between ML inferences (in packets)
    uint32_t m_minInferenceInterval;  //!< Minimum time between inferences (ms)
    
    // Fallback parameters
    uint32_t m_fallbackRate;          //!< Fallback rate index on ML failure
    bool m_enableFallback;            //!< Whether to use fallback on ML failure
    
    // Global tracking - REORDERED FOR PROPER INITIALIZATION
    mutable Time m_lastGlobalInference;                           //!< Last global inference time
    mutable uint32_t m_globalInferenceCount;                     //!< Total inferences performed
    mutable std::map<Mac48Address, TrafficStats> m_trafficStats;  //!< Per-station traffic stats
    
    TracedValue<uint64_t> m_currentRate;    //!< Current data rate
    TracedValue<uint32_t> m_mlInferences;   //!< Number of ML inferences made
    TracedValue<uint32_t> m_mlFailures;     //!< Number of ML failures
};

/**
 * \brief SmartWifiManagerXgb station state
 */
struct SmartWifiManagerXgbState : public WifiRemoteStation
{
    // SNR tracking
    double lastSnr;                   //!< Last reported SNR
    double snrFast;                   //!< Fast SNR average (alpha=0.1)
    double snrSlow;                   //!< Slow SNR average (alpha=0.01)
    double snrVariance;               //!< SNR variance
    std::deque<double> snrHistory;    //!< SNR history for variance calculation
    
    // Success tracking
    std::deque<bool> shortWindow;     //!< Short-term success window (10 packets)
    std::deque<bool> mediumWindow;    //!< Medium-term success window (50 packets)
    
    uint32_t consecSuccess;           //!< Consecutive successes
    uint32_t consecFailure;           //!< Consecutive failures
    uint32_t totalTransmissions;      //!< Total transmission attempts
    
    // Performance metrics
    double severity;                  //!< Failure severity (0-1)
    double confidence;                //!< Confidence in current rate (0-1)
    
    // Timing counters (in ms)
    uint32_t T1;                     //!< Time since last transmission
    uint32_t T2;                     //!< Time since last success
    uint32_t T3;                     //!< Time since last failure
    
    uint32_t retryCount;              //!< Current retry count
    
    // Mobility tracking
    double mobilityMetric;            //!< Mobility estimation (0-1)
    Vector lastPosition;              //!< Last known position
    double positionVariance;          //!< Position variance
    std::deque<Vector> positionHistory; //!< Position history
    
    // Timing
    Time lastUpdateTime;              //!< Last metrics update time
    Time lastInferenceTime;           //!< Last ML inference time
    Time lastSuccessTime;             //!< Last successful transmission
    Time lastFailureTime;             //!< Last failed transmission
    
    // Rate management
    uint32_t currentRateIndex;        //!< Current rate index
    uint32_t packetsSinceInference;   //!< Packets since last inference
    
    // Traffic tracking
    uint64_t bytesTransmitted;        //!< Total bytes transmitted
    uint32_t packetsTransmitted;      //!< Total packets transmitted
    double currentOfferedLoad;        //!< Current offered load estimate
    
    // Queue tracking
    uint32_t estimatedQueueLength;    //!< Estimated queue length
    Time lastQueueUpdate;             //!< Last queue length update
};

} // namespace ns3

#endif /* SMART_WIFI_MANAGER_XGB_H */