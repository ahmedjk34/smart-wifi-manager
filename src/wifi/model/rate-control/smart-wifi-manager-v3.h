/*
 * SmartWifiManagerV3 (Tuned Edition)
 *
 * SAME CONCEPTUAL MODEL AS ORIGINAL V3 (adaptive thresholds, dual history,
 * confidence + severity + trend gating) WITH LIGHT TUNING & SMALL
 * ENHANCEMENTS TO IMPROVE AGGRESSIVENESS (FASTER UPSHIFT) WHILE
 * KEEPING STABILITY (GUARDED DOWNSHIFT) TO EDGE OUT AARF.
 *
 * Added:
 *  - Per-station since-last-rate-change counter
 *  - Adaptive lowering of raise confidence threshold if "stuck"
 *  - Optional dynamic adaptation attribute controls
 *  - Slightly relaxed upward gating / stricter downward requirements
 *  - Hysteresis for downward changes (require minimal consecutive failures)
 *  - Mild acceleration (MaxUpstepPerDecision default now 2)
 */

#ifndef SMART_WIFI_MANAGER_V3_H
#define SMART_WIFI_MANAGER_V3_H

#include "ns3/wifi-remote-station-manager.h"
#include "ns3/traced-value.h"

namespace ns3
{

class SmartWifiManagerV3 : public WifiRemoteStationManager
{
public:
  static TypeId GetTypeId (void);
  SmartWifiManagerV3 ();
  ~SmartWifiManagerV3 () override;

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
  struct SmartWifiRemoteStationV3 : public WifiRemoteStation
  {
    uint32_t m_success{0};
    uint32_t m_failed{0};
    uint8_t  m_rate{0};
    double   m_lastSnr{0.0};

    // Dual history
    std::vector<bool> m_histShort;
    std::vector<bool> m_histMed;
    uint32_t m_histShortIdx{0};
    uint32_t m_histMedIdx{0};
    uint32_t m_histShortFill{0};
    uint32_t m_histMedFill{0};

    // Streaks
    uint32_t m_consecSuccess{0};
    uint32_t m_consecFailure{0};

    // Trend
    double m_snrFast{0.0};
    double m_snrSlow{0.0};
    bool   m_snrInit{false};

    // Adaptive thresholds
    double m_T1{10.0};
    double m_T2{15.0};
    double m_T3{25.0};

    std::vector<double> m_snrSamples;
    uint32_t m_txSinceThresholdUpdate{0};

    // Cached metrics
    double m_severity{0.0};
    double m_confidence{0.0};

    // New: count transmissions since last rate change to trigger adaptive lowering
    uint32_t m_sinceLastRateChange{0};

    SmartWifiRemoteStationV3 () = default;
  };

  SmartWifiRemoteStationV3 *Lookup (WifiRemoteStation *st) const
  {
    return static_cast<SmartWifiRemoteStationV3 *> (st);
  }

  // Helpers
  void UpdateHistoryOnSuccess (SmartWifiRemoteStationV3 *st);
  void UpdateHistoryOnFailure (SmartWifiRemoteStationV3 *st);
  void UpdateSnrStats (SmartWifiRemoteStationV3 *st, double snr);
  void MaybeAdaptThresholds (SmartWifiRemoteStationV3 *st);
  void ComputeSeverityConfidence (SmartWifiRemoteStationV3 *st,
                                   double &severity,
                                   double &confidence,
                                   uint8_t &targetTier,
                                   uint8_t maxRateIdx);
  uint8_t ApplyDecision (SmartWifiRemoteStationV3 *st,
                         uint8_t targetTier,
                         double severity,
                         double confidence,
                         uint8_t maxRateIdx);
  uint8_t TierFromSnr (SmartWifiRemoteStationV3 *st, double snr) const;
  double Quantile (std::vector<double> values, double q) const;

  // Dynamic confidence adaptation
  void MaybeRelaxRaiseThreshold (SmartWifiRemoteStationV3 *st);

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

  // New tuning attributes
  bool     m_enableDynamicConfidence;
  uint32_t m_stuckWindow;            // transmissions without rate change before relaxing threshold
  double   m_relaxDecay;             // decrement applied to raiseConfidence when stuck
  double   m_minDynamicRaiseConf;    // floor for dynamic lowering
  uint32_t m_minConsecFailForDown;   // hysteresis for soft drop
  uint32_t m_minConsecFailForHard;   // hysteresis for hard drop

  // Internal smoothing / clamp
  double   m_fastSnrAlpha;
  double   m_slowSnrAlpha;
  double   m_minT1;
  double   m_maxT3;

  // Mutable dynamic copy of raise confidence (if dynamic enabled)
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

  void VerboseLogDecision (SmartWifiRemoteStationV3 *st,
                           DecisionReason reason,
                           uint8_t oldRate,
                           uint8_t newRate,
                           uint8_t targetTier,
                           double severity,
                           double confidence);
};

} // namespace ns3

#endif // SMART_WIFI_MANAGER_V3_H