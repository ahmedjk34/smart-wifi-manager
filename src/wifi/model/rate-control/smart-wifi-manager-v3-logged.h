/*
 * SmartWifiManagerV3Logged
 *
 * SmartWifiManagerV3 with per-decision logging for ML dataset generation (Phase 4 prep).
 * Logs all relevant features to CSV: time, stationId, rateIdx, phyRate, lastSnr, snrFast, snrSlow,
 * shortSuccRatio, medSuccRatio, consecSuccess, consecFailure, severity, confidence,
 * T1, T2, T3, decisionReason, packetSuccess, offeredLoad, queueLen, retryCount,
 * channelWidth, mobilityMetric, snrVariance.
 *
 * Usage: Replace SmartWifiManagerV3 with SmartWifiManagerV3Logged in your simulation,
 * set LogFilePath attribute to output CSV, run scenarios.
 */

#ifndef SMART_WIFI_MANAGER_V3_LOGGED_H
#define SMART_WIFI_MANAGER_V3_LOGGED_H

#include "ns3/wifi-remote-station-manager.h"
#include "ns3/traced-value.h"
#include "ns3/node.h"
#include <vector>
#include <fstream>
#include <string>

namespace ns3
{

class SmartWifiManagerV3Logged : public WifiRemoteStationManager
{
public:
  static TypeId GetTypeId (void);
  SmartWifiManagerV3Logged ();
  ~SmartWifiManagerV3Logged () override;

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

    // For logging
    Ptr<Node> m_node;
    // Optional: cache last queueLen/retryCount/etc if accessible
    SmartWifiRemoteStationV3Logged () = default;
  };

  SmartWifiRemoteStationV3Logged *Lookup (WifiRemoteStation *st) const
  {
    return static_cast<SmartWifiRemoteStationV3Logged *> (st);
  }

  // Helpers
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

  // Logging related
  void LogDecision (SmartWifiRemoteStationV3Logged *st, int decisionReason, bool packetSuccess);

  std::ofstream m_logFile;
  std::string   m_logFilePath;
  bool          m_logHeaderWritten;

  // Attributes
  uint32_t m_historyShortLen;
  uint32_t m_historyMedLen;
  uint32_t m_thresholdAdaptInterval;
  double   m_raiseConfidenceThreshold;
  uint32_t m_minSuccessForRaise;
  double   m_severityAlpha;
  double   m_severityBeta;
  double   m_severeDropThreshold;
  double   m_softDropThreshold;
  double   m_trendUpDelta;
  double   m_trendDownDelta;
  uint32_t m_maxUpstepPerDecision;
  bool     m_enableRetryWeight;
  double   m_retryWeightLambda;
  bool     m_enableVerboseStats;
  bool     m_enableDynamicConfidence;
  uint32_t m_stuckWindow;
  double   m_relaxDecay;
  double   m_minDynamicRaiseConf;
  uint32_t m_minConsecFailForDown;
  uint32_t m_minConsecFailForHard;

  // Internal smoothing / clamp
  double   m_fastSnrAlpha;
  double   m_slowSnrAlpha;
  double   m_minT1;
  double   m_maxT3;

  double   m_currentRaiseConfidence;

  // Trace sources
  TracedValue<uint64_t> m_currentRate;
  TracedValue<double>   m_traceConfidence;
  TracedValue<double>   m_traceSeverity;
  TracedValue<double>   m_traceT1;
  TracedValue<double>   m_traceT2;
  TracedValue<double>   m_traceT3;

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