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
#include "ns3/node-container.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include <string>
#include <chrono>
#include <deque>
// #include "ns3/ptr.h"
// #include "ns3/wifi-phy.h"

namespace ns3
{

// --- HYBRID PATCH START ---
// Context classification
enum class WifiContextType {
    EMERGENCY,
    POOR_UNSTABLE,
    MARGINAL,
    GOOD_UNSTABLE,
    GOOD_STABLE,
    EXCELLENT_STABLE,
    UNKNOWN
};
// --- HYBRID PATCH END ---

/**
 * \brief Smart Rate control algorithm using Random Forest ML models
 * \ingroup wifi
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
        // --- HYBRID PATCH START ---
        double confidence; // ML confidence (future use)
        // --- HYBRID PATCH END ---
    };

    // --- HYBRID PATCH START ---
    struct SafetyAssessment
    {
        WifiContextType context;
        double riskLevel;
        uint32_t recommendedSafeRate;
        bool requiresEmergencyAction;
        double confidenceInAssessment;
        std::string contextStr;
    };
    // --- HYBRID PATCH END ---

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
    
    // SNR calculation method
    double CalculateCurrentSnr(WifiRemoteStation* station) const;

        /**
     * Calculate realistic SNR from received power
     * \param rxPowerDbm received power in dBm
     * \return corrected SNR in dB
     */
    // double CalculateRealisticSnr(double rxPowerDbm) const;
    
    /**
     * Convert power from Watts to dBm
     * \param rxPowerWatt received power in Watts
     * \return power in dBm
     */
    double ConvertRxPowerWattToDbm(double rxPowerWatt) const;
    // --- SNR FIX MEMBERS END ---


        /**
     * Calculate distance-based SNR using path loss model
     * \param st the remote station
     * \return calculated SNR in dB
     */
    double CalculateDistanceBasedSnr(WifiRemoteStation* st) const;
    
    /**
     * Calculate realistic SNR from ns-3 reported value
     * \param ns3SnrValue value reported by ns-3
     * \param st the remote station
     * \return corrected SNR in dB
     */
    double CalculateRealisticSnr(double ns3SnrValue) const;
    // --- CORRECTED SNR FIX MEMBERS END ---
    // --- HYBRID PATCH START ---
    // Context/risk assessment and fusion
    SafetyAssessment AssessNetworkSafety(struct SmartWifiManagerRfState* station);
    WifiContextType ClassifyNetworkContext(struct SmartWifiManagerRfState* station) const;
    std::string ContextTypeToString(WifiContextType type) const;
    double CalculateRiskLevel(struct SmartWifiManagerRfState* station) const;
    uint32_t GetContextSafeRate(struct SmartWifiManagerRfState* station, WifiContextType context) const;
    uint32_t GetRuleBasedRate(struct SmartWifiManagerRfState* station) const;
    void LogContextAndDecision(const SafetyAssessment& safety, uint32_t mlRate, uint32_t ruleRate, uint32_t finalRate) const;
    // --- HYBRID PATCH END ---

    //fixing the snr issues [not calcualted correctly from PHY layer]
    bool m_useDistanceBasedSnr;       //!< Use distance-based SNR calculation
    double m_txPowerDbm;              //!< Transmit power in dBm
    double m_noiseFigureDb;           //!< Noise figure in dB
    double m_frequencyGHz;            //!< Operating frequency in GHz
    double m_pathLossExponent;        //!< Path loss exponent
    double m_maxSnrDb;                //!< Maximum realistic SNR in dB
    double m_minSnrDb;                //!< Minimum realistic SNR in dB
    double m_thermalNoiseFloorDbm;    //!< Calculated thermal noise floor in dBm
        double m_snrOffset;               //!< SNR offset to apply to ns-3 values (dB)
            bool m_useRealisticSnr;           //!< Use realistic SNR calculation


    

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
    uint16_t m_inferenceServerPort;

    // --- HYBRID PATCH START ---
    // Tunable fusion thresholds
    double m_confidenceThreshold;
    double m_riskThreshold;
    uint32_t m_failureThreshold;
    // --- HYBRID PATCH END ---

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
    // --- HYBRID PATCH START ---
    WifiContextType lastContext;
    double lastRiskLevel;
    // --- HYBRID PATCH END ---

    uint32_t decisionReason = 0;          
    bool lastPacketSuccess = true;        
};

} // namespace ns3

#endif /* SMART_WIFI_MANAGER_RF_H */