#include "smart-wifi-manager-v3.h"

#include "ns3/log.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>

#define Min(a,b) ((a)<(b)?(a):(b))
#define Max(a,b) ((a)>(b)?(a):(b))

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("SmartWifiManagerV3");
NS_OBJECT_ENSURE_REGISTERED (SmartWifiManagerV3);

TypeId
SmartWifiManagerV3::GetTypeId ()
{
  static TypeId tid =
    TypeId ("ns3::SmartWifiManagerV3")
      .SetParent<WifiRemoteStationManager> ()
      .SetGroupName ("Wifi")
      .AddConstructor<SmartWifiManagerV3> ()
      .AddAttribute ("HistoryShortLen",
                     "Length of short success/failure history window.",
                     UintegerValue (5),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_historyShortLen),
                     MakeUintegerChecker<uint32_t> (1))
      .AddAttribute ("HistoryMedLen",
                     "Length of medium success/failure history window.",
                     UintegerValue (20),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_historyMedLen),
                     MakeUintegerChecker<uint32_t> (5))
      .AddAttribute ("ThresholdAdaptInterval",
                     "Number of successful transmissions between threshold re-estimation.",
                     UintegerValue (50),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_thresholdAdaptInterval),
                     MakeUintegerChecker<uint32_t> (10))
      // Tuned default: slightly lower to speed raises
      .AddAttribute ("RaiseConfidenceThreshold",
                     "Base confidence score required to allow upward rate movement.",
                     DoubleValue (0.64),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_raiseConfidenceThreshold),
                     MakeDoubleChecker<double> (0.0, 1.0))
      .AddAttribute ("MinSuccessForRaise",
                     "Minimum consecutive successes required (with confidence) to raise rate.",
                     UintegerValue (3),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_minSuccessForRaise),
                     MakeUintegerChecker<uint32_t> (1))
      // Slightly reduce severityAlpha so medium failures weigh less (more optimistic)
      .AddAttribute ("SeverityAlpha",
                     "Weight for medium-window failure ratio in severity index.",
                     DoubleValue (0.50),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_severityAlpha),
                     MakeDoubleChecker<double> (0.0, 1.0))
      .AddAttribute ("SeverityBeta",
                     "Weight for normalized consecutive failure streak in severity index.",
                     DoubleValue (0.30),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_severityBeta),
                     MakeDoubleChecker<double> (0.0, 1.0))
      // Slightly higher thresholds to avoid premature heavy drops
      .AddAttribute ("SevereDropThreshold",
                     "Severity index threshold triggering hard drop.",
                     DoubleValue (0.85),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_severeDropThreshold),
                     MakeDoubleChecker<double> (0.0, 1.0))
      .AddAttribute ("SoftDropThreshold",
                     "Severity index threshold triggering soft drop consideration.",
                     DoubleValue (0.45),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_softDropThreshold),
                     MakeDoubleChecker<double> (0.0, 1.0))
      // Faster detection of improving channel; allow smaller positive trend
      .AddAttribute ("TrendUpDelta",
                     "Positive SNR trend (fast - slow) enabling accelerated raise.",
                     DoubleValue (0.8),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_trendUpDelta),
                     MakeDoubleChecker<double> ())
      // Slightly more negative required to force preemptive caution
                     .AddAttribute ("TrendDownDelta",
                     "Negative SNR trend triggering preemptive caution.",
                     DoubleValue (-1.2),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_trendDownDelta),
                     MakeDoubleChecker<double> ())
      // Allow two-step acceleration when strongly justified
      .AddAttribute ("MaxUpstepPerDecision",
                     "Maximum upward steps allowed in one decision.",
                     UintegerValue (2),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_maxUpstepPerDecision),
                     MakeUintegerChecker<uint32_t> (1, 3))
      .AddAttribute ("EnableRetryWeight",
                     "Enable retry-based failure weighting (placeholder).",
                     BooleanValue (false),
                     MakeBooleanAccessor (&SmartWifiManagerV3::m_enableRetryWeight),
                     MakeBooleanChecker ())
      .AddAttribute ("RetryWeightLambda",
                     "Lambda factor for retry weighting (placeholder).",
                     DoubleValue (0.5),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_retryWeightLambda),
                     MakeDoubleChecker<double> (0.0))
      .AddAttribute ("EnableVerboseStats",
                     "Enable verbose logging of internal decision variables.",
                     BooleanValue (false),
                     MakeBooleanAccessor (&SmartWifiManagerV3::m_enableVerboseStats),
                     MakeBooleanChecker ())
      // New tuning attributes
      .AddAttribute ("EnableDynamicConfidence",
                     "Dynamically relax raise confidence threshold when stuck.",
                     BooleanValue (true),
                     MakeBooleanAccessor (&SmartWifiManagerV3::m_enableDynamicConfidence),
                     MakeBooleanChecker ())
      .AddAttribute ("StuckWindow",
                     "Transmissions without rate change before relaxing raise confidence.",
                     UintegerValue (70),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_stuckWindow),
                     MakeUintegerChecker<uint32_t> (10))
      .AddAttribute ("RelaxDecay",
                     "Amount to decrement raise threshold when stuck (applied multiplicatively to delta).",
                     DoubleValue (0.04),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_relaxDecay),
                     MakeDoubleChecker<double> (0.0, 0.25))
      .AddAttribute ("MinDynamicRaiseConf",
                     "Floor for dynamically relaxed raise confidence threshold.",
                     DoubleValue (0.50),
                     MakeDoubleAccessor (&SmartWifiManagerV3::m_minDynamicRaiseConf),
                     MakeDoubleChecker<double> (0.0, 0.9))
      .AddAttribute ("MinConsecFailForDown",
                     "Minimum consecutive failures required (with severity/soft threshold) before soft drop.",
                     UintegerValue (2),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_minConsecFailForDown),
                     MakeUintegerChecker<uint32_t> (1))
      .AddAttribute ("MinConsecFailForHard",
                     "Minimum consecutive failures required (with severe threshold) before hard drop.",
                     UintegerValue (3),
                     MakeUintegerAccessor (&SmartWifiManagerV3::m_minConsecFailForHard),
                     MakeUintegerChecker<uint32_t> (1))
      // Trace sources
      .AddTraceSource ("Rate",
                       "Traced value for data rate changes (b/s).",
                       MakeTraceSourceAccessor (&SmartWifiManagerV3::m_currentRate),
                       "ns3::TracedValueCallback::Uint64")
      .AddTraceSource ("Confidence",
                       "Confidence score used in decisions.",
                       MakeTraceSourceAccessor (&SmartWifiManagerV3::m_traceConfidence),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("Severity",
                       "Severity score used in decisions.",
                       MakeTraceSourceAccessor (&SmartWifiManagerV3::m_traceSeverity),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("ThresholdT1",
                       "Adaptive SNR threshold T1.",
                       MakeTraceSourceAccessor (&SmartWifiManagerV3::m_traceT1),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("ThresholdT2",
                       "Adaptive SNR threshold T2.",
                       MakeTraceSourceAccessor (&SmartWifiManagerV3::m_traceT2),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("ThresholdT3",
                       "Adaptive SNR threshold T3.",
                       MakeTraceSourceAccessor (&SmartWifiManagerV3::m_traceT3),
                       "ns3::TracedValueCallback::Double");
  return tid;
}

SmartWifiManagerV3::SmartWifiManagerV3 ()
  : WifiRemoteStationManager (),
    m_currentRate (0),
    m_traceConfidence (0.0),
    m_traceSeverity (0.0),
    m_traceT1 (10.0),
    m_traceT2 (15.0),
    m_traceT3 (25.0)
{
  m_fastSnrAlpha = 0.30;
  m_slowSnrAlpha = 0.05;
  m_minT1 = 4.0;
  m_maxT3 = 35.0;
  m_currentRaiseConfidence = 0.0; // set at Initialize
  NS_LOG_FUNCTION (this);
}

SmartWifiManagerV3::~SmartWifiManagerV3 ()
{
  NS_LOG_FUNCTION (this);
}

void
SmartWifiManagerV3::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  if (GetHtSupported () || GetVhtSupported () || GetHeSupported ())
    {
      NS_FATAL_ERROR ("SmartWifiManagerV3 currently supports only legacy/non-HT modes.");
    }
  // Capture base threshold into dynamic copy
  m_currentRaiseConfidence = m_raiseConfidenceThreshold;
}

/* Create per-station state */
WifiRemoteStation *
SmartWifiManagerV3::DoCreateStation () const
{
  NS_LOG_FUNCTION (this);
  auto st = new SmartWifiRemoteStationV3 ();
  st->m_histShort.resize (m_historyShortLen, true);
  st->m_histMed.resize (m_historyMedLen, true);
  st->m_snrSamples.reserve (200);
  return st;
}

void
SmartWifiManagerV3::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
SmartWifiManagerV3::DoReportRtsOk (WifiRemoteStation *station,
                                   double,
                                   WifiMode,
                                   double)
{
  NS_LOG_FUNCTION (this << station);
}

void
SmartWifiManagerV3::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
SmartWifiManagerV3::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

/* ==================== HISTORY & FEATURE UPDATES ==================== */

void
SmartWifiManagerV3::UpdateHistoryOnSuccess (SmartWifiRemoteStationV3 *st)
{
  st->m_histShort[st->m_histShortIdx] = true;
  st->m_histShortIdx = (st->m_histShortIdx + 1) % m_historyShortLen;
  if (st->m_histShortFill < m_historyShortLen) st->m_histShortFill++;

  st->m_histMed[st->m_histMedIdx] = true;
  st->m_histMedIdx = (st->m_histMedIdx + 1) % m_historyMedLen;
  if (st->m_histMedFill < m_historyMedLen) st->m_histMedFill++;

  st->m_consecSuccess++;
  st->m_consecFailure = 0;
  st->m_success++;
}

void
SmartWifiManagerV3::UpdateHistoryOnFailure (SmartWifiRemoteStationV3 *st)
{
  st->m_histShort[st->m_histShortIdx] = false;
  st->m_histShortIdx = (st->m_histShortIdx + 1) % m_historyShortLen;
  if (st->m_histShortFill < m_historyShortLen) st->m_histShortFill++;

  st->m_histMed[st->m_histMedIdx] = false;
  st->m_histMedIdx = (st->m_histMedIdx + 1) % m_historyMedLen;
  if (st->m_histMedFill < m_historyMedLen) st->m_histMedFill++;

  st->m_consecFailure++;
  st->m_consecSuccess = 0;
  st->m_failed++;
}

void
SmartWifiManagerV3::UpdateSnrStats (SmartWifiRemoteStationV3 *st, double snr)
{
  st->m_lastSnr = snr;
  if (!st->m_snrInit)
    {
      st->m_snrFast = snr;
      st->m_snrSlow = snr;
      st->m_snrInit = true;
    }
  else
    {
      st->m_snrFast = m_fastSnrAlpha * snr + (1.0 - m_fastSnrAlpha) * st->m_snrFast;
      st->m_snrSlow = m_slowSnrAlpha * snr + (1.0 - m_slowSnrAlpha) * st->m_snrSlow;
    }
  st->m_snrSamples.push_back (snr);
  if (st->m_snrSamples.size () > 300)
    {
      st->m_snrSamples.erase (st->m_snrSamples.begin (),
                              st->m_snrSamples.begin () + 100);
    }
}

double
SmartWifiManagerV3::Quantile (std::vector<double> values, double q) const
{
  if (values.empty ()) return 0.0;
  if (q <= 0.0) return *std::min_element (values.begin (), values.end ());
  if (q >= 1.0) return *std::max_element (values.begin (), values.end ());
  size_t idx = static_cast<size_t> (q * (values.size () - 1));
  std::nth_element (values.begin (), values.begin () + idx, values.end ());
  return values[idx];
}

void
SmartWifiManagerV3::MaybeAdaptThresholds (SmartWifiRemoteStationV3 *st)
{
  st->m_txSinceThresholdUpdate++;
  if (st->m_txSinceThresholdUpdate < m_thresholdAdaptInterval) return;
  st->m_txSinceThresholdUpdate = 0;
  if (st->m_snrSamples.size () < 8) return;

  std::vector<double> sampleCopy = st->m_snrSamples;
  double q25 = Quantile (sampleCopy, 0.25);
  double q50 = Quantile (sampleCopy, 0.50);
  double q75 = Quantile (sampleCopy, 0.75);

  double T1 = std::max (m_minT1, std::min (q25, q50 - 0.5));
  double T2 = std::max (T1 + 0.5, std::min (q50, q75 - 0.5));
  double T3 = std::max (T2 + 0.5, std::min (q75, m_maxT3));

  auto limitShift = [] (double oldV, double newV, double maxShift) {
    if (newV > oldV + maxShift) return oldV + maxShift;
    if (newV < oldV - maxShift) return oldV - maxShift;
    return newV;
  };
  const double MAX_STEP = 3.0;
  st->m_T1 = limitShift (st->m_T1, T1, MAX_STEP);
  st->m_T2 = limitShift (st->m_T2, T2, MAX_STEP);
  st->m_T3 = limitShift (st->m_T3, T3, MAX_STEP);

  m_traceT1 = st->m_T1;
  m_traceT2 = st->m_T2;
  m_traceT3 = st->m_T3;

  if (m_enableVerboseStats)
    {
      NS_LOG_DEBUG ("[V3][ThresholdAdapt] St=" << st
                    << " T1=" << st->m_T1
                    << " T2=" << st->m_T2
                    << " T3=" << st->m_T3);
    }
}

uint8_t
SmartWifiManagerV3::TierFromSnr (SmartWifiRemoteStationV3 *st, double snr) const
{
  if (snr < st->m_T1) return 0;
  if (snr < st->m_T2) return 1;
  if (snr < st->m_T3) return 2;
  return 3;
}

void
SmartWifiManagerV3::ComputeSeverityConfidence (SmartWifiRemoteStationV3 *st,
                                               double &severity,
                                               double &confidence,
                                               uint8_t &targetTier,
                                               uint8_t maxRateIdx)
{
  uint32_t succShort = 0;
  for (uint32_t i = 0; i < st->m_histShortFill; ++i)
    if (st->m_histShort[i]) succShort++;

  uint32_t succMed = 0;
  for (uint32_t i = 0; i < st->m_histMedFill; ++i)
    if (st->m_histMed[i]) succMed++;

  double shortDen = std::max<uint32_t> (1, st->m_histShortFill);
  double medDen = std::max<uint32_t> (1, st->m_histMedFill);

  double successRatioShort = double (succShort) / shortDen;
  double successRatioMed   = double (succMed) / medDen;
  double failureRatioMed   = 1.0 - successRatioMed;

  double normFailStreak = double (st->m_consecFailure) / double (m_historyShortLen);
  severity = m_severityAlpha * failureRatioMed +
             m_severityBeta * std::min (1.0, normFailStreak);

  uint8_t tier = TierFromSnr (st, st->m_lastSnr);
  uint8_t maxTier = (maxRateIdx >= 3) ? 3 : maxRateIdx;
  targetTier = std::min (tier, maxTier);

  double lowerBound, nextBound;
  if (targetTier == 0)
    { lowerBound = -5.0; nextBound = st->m_T1; }
  else if (targetTier == 1)
    { lowerBound = st->m_T1; nextBound = st->m_T2; }
  else if (targetTier == 2)
    { lowerBound = st->m_T2; nextBound = st->m_T3; }
  else
    { lowerBound = st->m_T3; nextBound = st->m_T3 + 6.0; }

  double snrMargin = 0.0;
  if (st->m_lastSnr >= lowerBound)
    {
      snrMargin = (st->m_lastSnr - lowerBound) / ( (nextBound - lowerBound) + 1e-6);
      snrMargin = std::max (0.0, std::min (1.0, snrMargin));
    }

  double retryRatio = 0.0; // placeholder

  const double w1 = 0.38; // short reliability slightly more weight (faster upward)
  const double w2 = 0.24;
  const double w3 = 0.18;
  const double w4 = 0.20;

  confidence = w1 * successRatioShort +
               w2 * successRatioMed +
               w3 * (1.0 - retryRatio) +
               w4 * snrMargin;

  st->m_severity = severity;
  st->m_confidence = confidence;
  m_traceSeverity = severity;
  m_traceConfidence = confidence;
}

void
SmartWifiManagerV3::MaybeRelaxRaiseThreshold (SmartWifiRemoteStationV3 *st)
{
  if (!m_enableDynamicConfidence) return;

  // If we've had a long run without a rate change, good SNR relative to top tier, and plenty of successes
  bool stableGoodSNR = (st->m_lastSnr > (st->m_T3 + 1.0));
  if (st->m_sinceLastRateChange >= m_stuckWindow &&
      stableGoodSNR &&
      st->m_consecSuccess >= m_stuckWindow / 2)
    {
      double old = m_currentRaiseConfidence;
      if (m_currentRaiseConfidence > m_minDynamicRaiseConf)
        {
          // Reduce by small delta, not below min floor
          m_currentRaiseConfidence =
            std::max (m_minDynamicRaiseConf,
                      m_currentRaiseConfidence - m_relaxDecay);
          if (m_enableVerboseStats)
            {
              NS_LOG_DEBUG ("[V3][DynamicRelax] St=" << st
                            << " RaiseConf " << old << " -> "
                            << m_currentRaiseConfidence);
            }
        }
    }
}

uint8_t
SmartWifiManagerV3::ApplyDecision (SmartWifiRemoteStationV3 *st,
                                   uint8_t targetTier,
                                   double severity,
                                   double confidence,
                                   uint8_t maxRateIdx)
{
  uint8_t oldRate = st->m_rate;

  // Derive approximate current tier
  uint8_t currentTier = 0;
  if (maxRateIdx >= 3)
    {
      if (st->m_rate <= maxRateIdx / 4) currentTier = 0;
      else if (st->m_rate <= (maxRateIdx / 2)) currentTier = 1;
      else if (st->m_rate <= ( (3 * maxRateIdx) / 4)) currentTier = 2;
      else currentTier = 3;
    }
  else
    {
      if (st->m_rate == 0) currentTier = 0;
      else if (st->m_rate == 1) currentTier = (maxRateIdx >= 1) ? 1 : 0;
      else currentTier = (maxRateIdx >= 2) ? 2 : 1;
    }

  uint8_t newRate = st->m_rate;
  DecisionReason reason = HOLD_STABLE;

  double trend = st->m_snrFast - st->m_snrSlow;
  bool trendPositive = (trend >= m_trendUpDelta);
  bool trendNegative = (trend <= m_trendDownDelta);

  // Use dynamic raise conf if enabled
  double raiseConf = m_enableDynamicConfidence ? m_currentRaiseConfidence : m_raiseConfidenceThreshold;

  // UPWARD
  if (targetTier > currentTier)
    {
      if (confidence >= raiseConf &&
          st->m_consecSuccess >= m_minSuccessForRaise)
        {
          uint32_t step = 1;
          if (trendPositive && m_maxUpstepPerDecision > 1 && confidence >= (raiseConf + 0.10))
            {
              step = std::min<uint32_t> (m_maxUpstepPerDecision, 2u);
              reason = TREND_ACCELERATE_UP;
            }
          else
            {
              reason = (raiseConf < m_raiseConfidenceThreshold) ? RAISE_DYNAMIC_RELAX
                                                                 : RAISE_CONFIRMED;
            }

          uint8_t tentative = std::min<uint8_t> (uint8_t (st->m_rate + step), maxRateIdx);
          newRate = tentative;
        }
      else
        {
          reason = BLOCK_LOW_CONF;
        }
    }
  // DOWNWARD
  else if (targetTier < currentTier)
    {
      if (severity >= m_severeDropThreshold &&
          st->m_consecFailure >= m_minConsecFailForHard)
        {
          newRate = (st->m_rate >= 2) ? uint8_t (st->m_rate - 2) : uint8_t (0);
          reason = HARD_DROP_SEVERITY;
        }
      else if ((severity >= m_softDropThreshold || trendNegative) &&
               st->m_consecFailure >= m_minConsecFailForDown)
        {
          if (st->m_rate > 0)
            {
              newRate = st->m_rate - 1;
              reason = (trendNegative ? TREND_DECELERATE : SOFT_DROP_SEVERITY);
            }
        }
      else
        {
          reason = HOLD_STABLE;
        }
    }
  else
    {
      // Same tier: allow internal micro-raise if strong confidence & trend
      if (confidence >= (raiseConf + 0.18) &&
          trendPositive &&
          st->m_consecSuccess >= (m_minSuccessForRaise + 1) &&
          st->m_rate < maxRateIdx)
        {
          newRate = st->m_rate + 1;
          reason = TREND_ACCELERATE_UP;
        }
    }

  // Safety: approximate required SNR
  double requiredSnr = 2.0 + 2.0 * newRate; // still lightweight
  if (st->m_lastSnr + 0.5 < requiredSnr && newRate > 0)
    {
      newRate -= 1;
      reason = SAFETY_SNR_CLAMP;
    }

  if (newRate != oldRate)
    {
      st->m_rate = newRate;
      st->m_sinceLastRateChange = 0; // reset stagnation counter
      // Reset dynamic raise threshold if we moved up strongly (optional)
      if (m_enableDynamicConfidence && newRate > oldRate)
        {
          m_currentRaiseConfidence =
            std::max (m_currentRaiseConfidence, m_raiseConfidenceThreshold);
        }
    }
  else
    {
      st->m_sinceLastRateChange++;
    }

  if (m_enableVerboseStats)
    {
      VerboseLogDecision (st, reason, oldRate, newRate, targetTier, severity, confidence);
    }

  return st->m_rate;
}

void
SmartWifiManagerV3::VerboseLogDecision (SmartWifiRemoteStationV3 *st,
                                        DecisionReason reason,
                                        uint8_t oldRate,
                                        uint8_t newRate,
                                        uint8_t targetTier,
                                        double severity,
                                        double confidence)
{
  NS_LOG_DEBUG ("[V3][Decision]"
                << " St=" << st
                << " OldRate=" << (uint32_t)oldRate
                << " NewRate=" << (uint32_t)newRate
                << " TargetTier=" << (uint32_t)targetTier
                << " SNR=" << st->m_lastSnr
                << " T1=" << st->m_T1
                << " T2=" << st->m_T2
                << " T3=" << st->m_T3
                << " Conf=" << confidence
                << " DynRaiseConf=" << m_currentRaiseConfidence
                << " Sev=" << severity
                << " ConsecSucc=" << st->m_consecSuccess
                << " ConsecFail=" << st->m_consecFailure
                << " SinceRateChg=" << st->m_sinceLastRateChange
                << " Trend=" << (st->m_snrFast - st->m_snrSlow)
                << " Reason=" << reason);
}

/* ==================== CALLBACKS ==================== */

void
SmartWifiManagerV3::DoReportDataOk (WifiRemoteStation *station,
                                    double,
                                    WifiMode,
                                    double dataSnr,
                                    uint16_t,
                                    uint8_t)
{
  NS_LOG_FUNCTION (this << station << dataSnr);
  auto st = Lookup (station);

  UpdateSnrStats (st, dataSnr);
  UpdateHistoryOnSuccess (st);
  MaybeAdaptThresholds (st);
  MaybeRelaxRaiseThreshold (st);

  uint8_t maxRateIdx = GetNSupported (st) - 1;

  double severity = 0.0;
  double confidence = 0.0;
  uint8_t targetTier = 0;

  ComputeSeverityConfidence (st, severity, confidence, targetTier, maxRateIdx);
  ApplyDecision (st, targetTier, severity, confidence, maxRateIdx);
}

void
SmartWifiManagerV3::DoReportDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  auto st = Lookup (station);

  UpdateHistoryOnFailure (st);

  // Re-evaluate severity / confidence for immediate protective action
  double severity=0.0, confidence=0.0;
  uint8_t targetTier=0;
  uint8_t maxRateIdx = GetNSupported (st) - 1;
  ComputeSeverityConfidence (st, severity, confidence, targetTier, maxRateIdx);

  if (severity >= m_severeDropThreshold &&
      st->m_consecFailure >= m_minConsecFailForHard)
    {
      uint8_t oldRate = st->m_rate;
      st->m_rate = (st->m_rate >= 2) ? st->m_rate - 2 : (uint8_t)0;
      st->m_sinceLastRateChange = 0;
      if (m_enableVerboseStats)
        {
          NS_LOG_DEBUG ("[V3][ImmediateHardDrop] St=" << st
                        << " OldRate=" << (uint32_t)oldRate
                        << " NewRate=" << (uint32_t)st->m_rate
                        << " Sev=" << severity
                        << " ConsecFail=" << st->m_consecFailure);
        }
    }
  else if ((severity >= m_softDropThreshold ||
            (st->m_consecFailure >= m_minConsecFailForDown && (st->m_lastSnr < st->m_T2))) &&
           st->m_rate > 0 &&
           st->m_consecFailure >= m_minConsecFailForDown)
    {
      uint8_t oldRate = st->m_rate;
      st->m_rate = st->m_rate - 1;
      st->m_sinceLastRateChange = 0;
      if (m_enableVerboseStats)
        {
          NS_LOG_DEBUG ("[V3][ImmediateSoftDrop] St=" << st
                        << " OldRate=" << (uint32_t)oldRate
                        << " NewRate=" << (uint32_t)st->m_rate
                        << " Sev=" << severity
                        << " ConsecFail=" << st->m_consecFailure);
        }
    }
}

void
SmartWifiManagerV3::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr);
  auto st = Lookup (station);
  UpdateSnrStats (st, rxSnr);
}

/* ==================== TX VECTOR SELECTION ==================== */

WifiTxVector
SmartWifiManagerV3::DoGetDataTxVector (WifiRemoteStation *station, uint16_t allowedWidth)
{
  NS_LOG_FUNCTION (this << station << allowedWidth);
  auto st = Lookup (station);

  uint8_t rateIndex = std::min<uint8_t> (st->m_rate, GetNSupported (st) - 1);
  WifiMode mode = GetSupported (st, rateIndex);

  uint16_t channelWidth = GetChannelWidth (st);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  uint64_t rate = mode.GetDataRate (channelWidth);
  if (m_currentRate != rate)
    {
      NS_LOG_DEBUG ("[V3] NewDataRate=" << rate
                    << " (idx=" << (uint32_t)rateIndex << ")");
      m_currentRate = rate;
    }

  return WifiTxVector (mode,
                       GetDefaultTxPowerLevel (),
                       GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()),
                       800,
                       1,
                       1,
                       0,
                       channelWidth,
                       GetAggregation (st));
}

WifiTxVector
SmartWifiManagerV3::DoGetRtsTxVector (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  auto st = Lookup (station);
  uint16_t channelWidth = GetChannelWidth (st);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  WifiMode mode;
  if (!GetUseNonErpProtection ())
    {
      mode = GetSupported (st, 0);
    }
  else
    {
      mode = GetNonErpSupported (st, 0);
    }
  return WifiTxVector (mode,
                       GetDefaultTxPowerLevel (),
                       GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()),
                       800,
                       1,
                       1,
                       0,
                       channelWidth,
                       GetAggregation (st));
}

} // namespace ns3