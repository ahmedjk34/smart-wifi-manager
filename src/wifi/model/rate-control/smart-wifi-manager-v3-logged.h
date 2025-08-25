/*
 * SmartWifiManagerV3Logged
 *
 * Phase 1: Enhanced Data Collection & Feature Engineering
 * - Feedback-oriented features added to logger
 * - Stratified probabilistic logging implemented
 * - All new code marked with PHASE1 NEW CODE comments
 */

#ifndef SMART_WIFI_MANAGER_V3_LOGGED_H
#define SMART_WIFI_MANAGER_V3_LOGGED_H

#include "ns3/wifi-remote-station-manager.h"
#include "ns3/traced-value.h"
#include "ns3/node.h"
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <cmath>

namespace ns3
{

class SmartWifiManagerV3Logged : public WifiRemoteStationManager
{
public:
  static TypeId GetTypeId (void);
  SmartWifiManagerV3Logged ();
  ~SmartWifiManagerV3Logged () override;
  TracedCallback<std::string, bool> m_packetResultTrace;
  TracedCallback<uint64_t, uint64_t> m_rateChange;


protected:
  void DoInitialize () override;
  WifiRemoteStation *DoCreateStation () const override;
  void DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode) override;
  void DoReportRtsOk (WifiRemoteStation *station,
                      double ctsSnr,
                      WifiMode ctsMode,
                      double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station,
                       double ackSnr,
                       WifiMode ackMode,
                       double dataSnr,
                       uint16_t dataChannelWidth,
                       uint8_t dataNss) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station, uint16_t allowedWidth) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;

private:
  struct SmartWifiRemoteStationV3Logged : public WifiRemoteStation
  {
    uint32_t m_success{0};
    uint32_t m_failed{0};
    uint8_t  m_rate{0};
    double   m_lastSnr{0.0};
    std::vector<bool> m_histShort;
    std::vector<bool> m_histMed;
    uint32_t m_histShortIdx{0};
    uint32_t m_histMedIdx{0};
    uint32_t m_histShortFill{0};
    uint32_t m_histMedFill{0};
    uint32_t m_consecSuccess{0};
    uint32_t m_consecFailure{0};
    double   m_snrFast{0.0};
    double   m_snrSlow{0.0};
    bool     m_snrInit{false};
    double   m_T1{10.0};
    double   m_T2{15.0};
    double   m_T3{25.0};
    std::vector<double> m_snrSamples;
    uint32_t m_txSinceThresholdUpdate{0};
    double   m_severity{0.0};
    double   m_confidence{0.0};
    uint32_t m_sinceLastRateChange{0};
    Ptr<Node> m_node;

    // PHASE1 NEW CODE: Feedback-oriented feature buffers
    std::vector<uint8_t> m_rateHistory;           // Recent rates for adaptation dynamics
    std::vector<double> m_throughputHistory;      // Recent throughput (Mbps or rate index as proxy)
    std::vector<uint32_t> m_retryHistory;         // Recent retry counts (placeholder)
    std::vector<bool> m_packetSuccessHistory;     // Recent packet success/failure
    SmartWifiRemoteStationV3Logged () = default;
  };

  SmartWifiRemoteStationV3Logged *Lookup (WifiRemoteStation *st) const
  {
    return static_cast<SmartWifiRemoteStationV3Logged *> (st);
  }

  // History helpers
  void UpdateHistoryOnSuccess (SmartWifiRemoteStationV3Logged *st);
  void UpdateHistoryOnFailure (SmartWifiRemoteStationV3Logged *st);
  void UpdateSnrStats (SmartWifiRemoteStationV3Logged *st, double snr);
  void MaybeAdaptThresholds (SmartWifiRemoteStationV3Logged *st);
  void ComputeSeverityConfidence (SmartWifiRemoteStationV3Logged *st,
                                   double &severity,
                                   double &confidence,
                                   uint8_t &targetTier,
                                   uint8_t maxRateIdx);
  uint8_t ApplyDecision (SmartWifiRemoteStationV3Logged *st,
                       uint8_t targetTier,
                       double severity,
                       double confidence,
                       uint8_t maxRateIdx,
                       int& decisionReason);

  uint8_t TierFromSnr (SmartWifiRemoteStationV3Logged *st, double snr) const;
  double Quantile (std::vector<double> values, double q) const;
  void MaybeRelaxRaiseThreshold (SmartWifiRemoteStationV3Logged *st);

  // PHASE1 NEW CODE: Stratified logging random
  std::mt19937 m_rng;
  std::uniform_real_distribution<double> m_uniformDist;
  double GetStratifiedLogProbability(uint8_t rate, double snr, bool success);
  double GetRandomValue();

  // PHASE1 NEW CODE: Feedback-oriented helpers
  uint32_t CountRecentRateChanges(SmartWifiRemoteStationV3Logged* st, uint32_t window);
  double CalculateRateStability(SmartWifiRemoteStationV3Logged* st);
  double CalculateRecentThroughput(SmartWifiRemoteStationV3Logged* st, uint32_t window);
  double CalculateRecentPacketLoss(SmartWifiRemoteStationV3Logged* st, uint32_t window);
  double CalculateRetrySuccessRatio(SmartWifiRemoteStationV3Logged* st);
  double CalculateOptimalRateDistance(SmartWifiRemoteStationV3Logged* st);
  double CalculateAggressiveFactor(SmartWifiRemoteStationV3Logged* st);
  double CalculateConservativeFactor(SmartWifiRemoteStationV3Logged* st);
  uint8_t GetRecommendedSafeRate(SmartWifiRemoteStationV3Logged* st);
  double CalculateSnrStability(SmartWifiRemoteStationV3Logged* st);
  double CalculateSnrPredictionConfidence(SmartWifiRemoteStationV3Logged* st);

  void LogDecision (SmartWifiRemoteStationV3Logged *st, int decisionReason, bool packetSuccess);

  std::ofstream m_logFile;
  std::string   m_logFilePath;
  bool          m_logHeaderWritten;

  // PHASE1 NEW CODE: Add trace sources and current rate fields for constructor initialization
  uint64_t m_currentRate;
  double m_traceConfidence;
  double m_traceSeverity;
  double m_traceT1;
  double m_traceT2;
  double m_traceT3;

  // Attributes and trace sources (unchanged, omitted for brevity)
  // ...

  enum DecisionReason
  {
    HOLD_STABLE = 0,
    RAISE_CONFIRMED,
    BLOCK_LOW_CONF,
    SOFT_DROP_SEVERITY,
    HARD_DROP_SEVERITY,
    SAFETY_SNR_CLAMP,
    TREND_ACCELERATE_UP,
    TREND_DECELERATE,
    RAISE_DYNAMIC_RELAX
  };
};

} // namespace ns3

#endif // SMART_WIFI_MANAGER_V3_LOGGED_H