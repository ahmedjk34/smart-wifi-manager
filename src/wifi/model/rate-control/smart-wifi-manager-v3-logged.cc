/*
 * SmartWifiManagerV3Logged
 *
 * Phase 1: Enhanced Data Collection & Feature Engineering
 * - Feedback-oriented features added to logger
 * - Stratified probabilistic logging implemented
 * - All new code marked with PHASE1 NEW CODE comments
 */

#include "smart-wifi-manager-v3-logged.h"

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/node.h"
#include "ns3/wifi-mode.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/boolean.h"
#include "ns3/string.h"

#include <algorithm>
#include <sstream>
#include <iomanip>
#include <random>
#include <cmath>
#include <limits>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("SmartWifiManagerV3Logged");
NS_OBJECT_ENSURE_REGISTERED (SmartWifiManagerV3Logged);

// -----------------------------------------------------------------------------
// TypeId: only advertise things that actually exist in the header.
// -----------------------------------------------------------------------------
TypeId
SmartWifiManagerV3Logged::GetTypeId ()
{
  static TypeId tid =
    TypeId ("ns3::SmartWifiManagerV3Logged")
      .SetParent<WifiRemoteStationManager> ()
      .SetGroupName ("Wifi")
      .AddConstructor<SmartWifiManagerV3Logged> ()
      // Keep attributes minimal and valid. Path is a plain attribute.
      .AddAttribute ("LogFilePath",
                     "CSV log output path for SmartWifiManagerV3Logged.",
                     StringValue (""), // default: disabled
                     MakeStringAccessor (&SmartWifiManagerV3Logged::m_logFilePath),
                     MakeStringChecker ())
      // Only valid trace: the header defines TracedCallback<std::string, bool> m_packetResultTrace.
        .AddTraceSource("PacketResult",
                "Emits (stationId, success) per data attempt.",
                MakeTraceSourceAccessor(&SmartWifiManagerV3Logged::m_packetResultTrace),
                "ns3::TracedCallback::Context")
        .AddTraceSource("RateChange",
                "Emits (newRate, oldRate) when data rate changes",
                MakeTraceSourceAccessor(&SmartWifiManagerV3Logged::m_rateChange),
                "ns3::TracedCallback::Uint64Uint64");
  return tid;
}

// -----------------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------------
SmartWifiManagerV3Logged::SmartWifiManagerV3Logged ()
  : WifiRemoteStationManager (),
    // Keep initialization order matching header declaration order to avoid -Wreorder.
    m_rng (std::random_device{} ()),
    m_uniformDist (0.0, 1.0),
    m_logFile (),
    m_logFilePath (),
    m_logHeaderWritten (false),
    m_currentRate (0),
    m_traceConfidence (0.0),
    m_traceSeverity (0.0),
    m_traceT1 (10.0),
    m_traceT2 (15.0),
    m_traceT3 (25.0)
{
  NS_LOG_FUNCTION (this);
}

SmartWifiManagerV3Logged::~SmartWifiManagerV3Logged ()
{
  NS_LOG_FUNCTION (this);
  if (m_logFile.is_open ())
    {
      m_logFile.close ();
    }
}

// -----------------------------------------------------------------------------
// Initialization: open CSV and write header (once).
// -----------------------------------------------------------------------------
void
SmartWifiManagerV3Logged::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  // No dynamic-confidence state here; we only rely on header-defined fields.

  if (!m_logFilePath.empty () && !m_logHeaderWritten)
    {
      m_logFile.open (m_logFilePath, std::ios::out | std::ios::trunc);
      if (!m_logFile.is_open ())
        {
          NS_LOG_ERROR ("Failed to open log file: " << m_logFilePath);
          // Continue without logging; the algorithm still runs.
        }
      else
        {
          // PHASE1 NEW CODE: enriched CSV header
          m_logFile << "time,stationId,rateIdx,phyRate,"
                       "lastSnr,snrFast,snrSlow,"
                       "snrTrendShort,snrStabilityIndex,snrPredictionConfidence,"
                       "shortSuccRatio,medSuccRatio,consecSuccess,consecFailure,"
                       "recentThroughputTrend,packetLossRate,retrySuccessRatio,"
                       "recentRateChanges,timeSinceLastRateChange,rateStabilityScore,"
                       "optimalRateDistance,aggressiveFactor,conservativeFactor,recommendedSafeRate,"
                       "severity,confidence,T1,T2,T3,decisionReason,packetSuccess,"
                       "offeredLoad,queueLen,retryCount,channelWidth,mobilityMetric,snrVariance\n";
          m_logFile.flush ();
          m_logHeaderWritten = true;
        }
    }

  WifiRemoteStationManager::DoInitialize ();
}

// -----------------------------------------------------------------------------
// Station creation: fill only members that exist in the header.
// -----------------------------------------------------------------------------
WifiRemoteStation *
SmartWifiManagerV3Logged::DoCreateStation () const
{
  NS_LOG_FUNCTION (this);

  auto st = new SmartWifiRemoteStationV3Logged ();
  // Reasonable defaults
  st->m_success = 0;
  st->m_failed = 0;
  st->m_rate = 0;        // start slow/safe
  st->m_lastSnr = 0.0;
  st->m_snrFast = 0.0;
  st->m_snrSlow = 0.0;
  st->m_snrInit = false;
  st->m_T1 = 10.0;
  st->m_T2 = 15.0;
  st->m_T3 = 25.0;
  st->m_txSinceThresholdUpdate = 0;
  st->m_severity = 0.0;
  st->m_confidence = 0.0;
  st->m_sinceLastRateChange = 0;
  st->m_node = nullptr;

  // History sizes: pick fixed lengths, since there are no attributes in header.
  const uint32_t kShort = 10;
  const uint32_t kMed   = 20;
  st->m_histShort.assign (kShort, true);
  st->m_histMed.assign (kMed, true);
  st->m_histShortIdx = 0;
  st->m_histMedIdx = 0;
  st->m_histShortFill = 0;
  st->m_histMedFill = 0;

  // PHASE1 NEW CODE: feedback buffers
  st->m_rateHistory.clear ();
  st->m_throughputHistory.clear ();
  st->m_retryHistory.clear ();
  st->m_packetSuccessHistory.clear ();

  return st;
}

// -----------------------------------------------------------------------------
// RX / RTS reporting: keep light, record SNR samples where sensible.
// -----------------------------------------------------------------------------
void
SmartWifiManagerV3Logged::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr);
  auto st = Lookup (station);
  if (std::isfinite (rxSnr))
    {
      st->m_snrSamples.push_back (rxSnr);
      if (st->m_snrSamples.size () > 200) // cap memory
        {
          st->m_snrSamples.erase (st->m_snrSamples.begin (),
                                  st->m_snrSamples.begin () + (st->m_snrSamples.size () - 200));
        }
    }
}

void
SmartWifiManagerV3Logged::DoReportRtsOk (WifiRemoteStation *, double, WifiMode, double)
{
  NS_LOG_FUNCTION (this);
  // No-op: we don't currently incorporate RTS SNR into adaptation.
}

// -----------------------------------------------------------------------------
// Core reporting: success/failure on DATA
// -----------------------------------------------------------------------------
void
SmartWifiManagerV3Logged::DoReportDataOk (WifiRemoteStation *station,
                                          double /*ackSnr*/,
                                          WifiMode /*ackMode*/,
                                          double dataSnr,
                                          uint16_t /*dataChannelWidth*/,
                                          uint8_t /*dataNss*/)
{
  NS_LOG_FUNCTION (this << station << dataSnr);
  auto st = Lookup (station);

  UpdateSnrStats (st, dataSnr);
  UpdateHistoryOnSuccess (st);

  // PHASE1 NEW CODE: feedback buffers
  st->m_rateHistory.push_back (st->m_rate);
  if (st->m_rateHistory.size () > 20) st->m_rateHistory.erase (st->m_rateHistory.begin ());

  // Throughput proxy: use rate index (plain model); replace with actual Mbps if available.
  st->m_throughputHistory.push_back (static_cast<double> (st->m_rate));
  if (st->m_throughputHistory.size () > 20) st->m_throughputHistory.erase (st->m_throughputHistory.begin ());

  st->m_packetSuccessHistory.push_back (true);
  if (st->m_packetSuccessHistory.size () > 20) st->m_packetSuccessHistory.erase (st->m_packetSuccessHistory.begin ());

  st->m_retryHistory.push_back (0);
  if (st->m_retryHistory.size () > 20) st->m_retryHistory.erase (st->m_retryHistory.begin ());

  // Decision & logging
  uint8_t maxRateIdx = 7; // 8 MCS-equivalent "tiers" (0..7) for a/g-like profiles
  double severity = 0.0, confidence = 0.0;
  uint8_t targetTier = 0;
  ComputeSeverityConfidence (st, severity, confidence, targetTier, maxRateIdx);

  int decisionReason = HOLD_STABLE;
  ApplyDecision (st, targetTier, severity, confidence, maxRateIdx, decisionReason);

  // Emit simple packet result trace (stationId string, success)
  std::ostringstream sid;
  if (st->m_node) sid << st->m_node->GetId (); else sid << reinterpret_cast<uintptr_t> (st);
  m_packetResultTrace (sid.str (), true);

  LogDecision (st, decisionReason, true);
}

void
SmartWifiManagerV3Logged::DoReportDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  auto st = Lookup (station);

  UpdateHistoryOnFailure (st);

  // PHASE1 NEW CODE: feedback buffers
  st->m_rateHistory.push_back (st->m_rate);
  if (st->m_rateHistory.size () > 20) st->m_rateHistory.erase (st->m_rateHistory.begin ());

  st->m_throughputHistory.push_back (0.0);
  if (st->m_throughputHistory.size () > 20) st->m_throughputHistory.erase (st->m_throughputHistory.begin ());

  st->m_packetSuccessHistory.push_back (false);
  if (st->m_packetSuccessHistory.size () > 20) st->m_packetSuccessHistory.erase (st->m_packetSuccessHistory.begin ());

  st->m_retryHistory.push_back (1);
  if (st->m_retryHistory.size () > 20) st->m_retryHistory.erase (st->m_retryHistory.begin ());

  // Compute features for logging (decision itself can be deferred to next TX)
  double severity = 0.0, confidence = 0.0;
  uint8_t targetTier = 0;
  uint8_t maxRateIdx = 7;
  ComputeSeverityConfidence (st, severity, confidence, targetTier, maxRateIdx);

  // Emit simple packet result trace (stationId string, failure)
  std::ostringstream sid;
  if (st->m_node) sid << st->m_node->GetId (); else sid << reinterpret_cast<uintptr_t> (st);
  m_packetResultTrace (sid.str (), false);

  int decisionReason = HOLD_STABLE; // we log the context; next TX vector will react
  LogDecision (st, decisionReason, false);
}

void
SmartWifiManagerV3Logged::DoReportRtsFailed (WifiRemoteStation *)
{
  NS_LOG_FUNCTION (this);
}

void
SmartWifiManagerV3Logged::DoReportFinalRtsFailed (WifiRemoteStation *)
{
  NS_LOG_FUNCTION (this);
}

void
SmartWifiManagerV3Logged::DoReportFinalDataFailed (WifiRemoteStation *)
{
  NS_LOG_FUNCTION (this);
}

// Map our 0..7 "tiers" to 802.11g ERP-OFDM modes (6..54 Mbps).
// PHASE1 NEW CODE: map 0..7 tier -> 802.11g ERP-OFDM WifiMode
static inline ns3::WifiMode
TierToErpOfdmMode (uint8_t tier)
{
  switch (std::min<uint8_t>(tier, 7))
    {
    case 0: return ns3::WifiMode ("ErpOfdmRate6Mbps");
    case 1: return ns3::WifiMode ("ErpOfdmRate9Mbps");
    case 2: return ns3::WifiMode ("ErpOfdmRate12Mbps");
    case 3: return ns3::WifiMode ("ErpOfdmRate18Mbps");
    case 4: return ns3::WifiMode ("ErpOfdmRate24Mbps");
    case 5: return ns3::WifiMode ("ErpOfdmRate36Mbps");
    case 6: return ns3::WifiMode ("ErpOfdmRate48Mbps");
    default: return ns3::WifiMode ("ErpOfdmRate54Mbps");
    }
}

// -----------------------------------------------------------------------------
// TX vectors: return a conservative/default vector. (Fixed implementation.)
// -----------------------------------------------------------------------------
ns3::WifiTxVector
SmartWifiManagerV3Logged::DoGetDataTxVector (ns3::WifiRemoteStation *station, uint16_t allowedWidth)
{
  NS_LOG_FUNCTION (this << station << allowedWidth);

  auto st = Lookup (station);
  uint8_t tier = (st ? st->m_rate : 0);
  ns3::WifiMode mode = TierToErpOfdmMode (tier);

  // Construct a WifiTxVector and set minimum required fields.
  ns3::WifiTxVector v;
  v.SetMode (mode);                       // REQUIRED: set mode
  v.SetChannelWidth (allowedWidth ? allowedWidth : 20); // default 20MHz if 0
  v.SetNss (1);                           // single spatial stream (safe default)
  v.SetPreambleType (ns3::WIFI_PREAMBLE_LONG); // Valid preamble type

  return v;
}

ns3::WifiTxVector
SmartWifiManagerV3Logged::DoGetRtsTxVector (ns3::WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  
  // For RTS, use a conservative mode (lowest rate for reliability)
  ns3::WifiTxVector v;
  v.SetMode (ns3::WifiMode ("ErpOfdmRate6Mbps"));
  v.SetChannelWidth (20);
  v.SetNss (1);
  v.SetPreambleType (ns3::WIFI_PREAMBLE_LONG);
  
  return v;
}

// -----------------------------------------------------------------------------
// Internal helpers (only use members declared in header)
// -----------------------------------------------------------------------------
void
SmartWifiManagerV3Logged::UpdateHistoryOnSuccess (SmartWifiRemoteStationV3Logged *st)
{
  st->m_success++;
  st->m_consecSuccess++;
  st->m_consecFailure = 0;

  if (!st->m_histShort.empty ())
    {
      st->m_histShort[st->m_histShortIdx] = true;
      st->m_histShortIdx = (st->m_histShortIdx + 1) % st->m_histShort.size ();
      st->m_histShortFill = std::min<uint32_t> (st->m_histShortFill + 1, st->m_histShort.size ());
    }
  if (!st->m_histMed.empty ())
    {
      st->m_histMed[st->m_histMedIdx] = true;
      st->m_histMedIdx = (st->m_histMedIdx + 1) % st->m_histMed.size ();
      st->m_histMedFill = std::min<uint32_t> (st->m_histMedFill + 1, st->m_histMed.size ());
    }

  st->m_txSinceThresholdUpdate++;
  MaybeAdaptThresholds (st);
}

void
SmartWifiManagerV3Logged::UpdateHistoryOnFailure (SmartWifiRemoteStationV3Logged *st)
{
  st->m_failed++;
  st->m_consecFailure++;
  st->m_consecSuccess = 0;

  if (!st->m_histShort.empty ())
    {
      st->m_histShort[st->m_histShortIdx] = false;
      st->m_histShortIdx = (st->m_histShortIdx + 1) % st->m_histShort.size ();
      st->m_histShortFill = std::min<uint32_t> (st->m_histShortFill + 1, st->m_histShort.size ());
    }
  if (!st->m_histMed.empty ())
    {
      st->m_histMed[st->m_histMedIdx] = false;
      st->m_histMedIdx = (st->m_histMedIdx + 1) % st->m_histMed.size ();
      st->m_histMedFill = std::min<uint32_t> (st->m_histMedFill + 1, st->m_histMed.size ());
    }

  st->m_txSinceThresholdUpdate++;
  MaybeAdaptThresholds (st);
}

void
SmartWifiManagerV3Logged::UpdateSnrStats (SmartWifiRemoteStationV3Logged *st, double snr)
{
  if (!std::isfinite (snr))
    {
      return;
    }

  // EWMA constants (local; not attributes in header)
  const double kAlphaFast = 0.30;
  const double kAlphaSlow = 0.05;

  if (!st->m_snrInit)
    {
      st->m_snrFast = snr;
      st->m_snrSlow = snr;
      st->m_snrInit = true;
    }
  else
    {
      st->m_snrFast = kAlphaFast * snr + (1.0 - kAlphaFast) * st->m_snrFast;
      st->m_snrSlow = kAlphaSlow * snr + (1.0 - kAlphaSlow) * st->m_snrSlow;
    }

  st->m_lastSnr = snr;
  st->m_snrSamples.push_back (snr);
  if (st->m_snrSamples.size () > 200)
    {
      st->m_snrSamples.erase (st->m_snrSamples.begin (),
                              st->m_snrSamples.begin () + (st->m_snrSamples.size () - 200));
    }
}

double
SmartWifiManagerV3Logged::Quantile (std::vector<double> values, double q) const
{
  if (values.empty ())
    return 0.0;
  std::sort (values.begin (), values.end ());
  if (q <= 0.0) return values.front ();
  if (q >= 1.0) return values.back ();
  const double idx = q * (values.size () - 1);
  const size_t lo = static_cast<size_t> (std::floor (idx));
  const size_t hi = static_cast<size_t> (std::ceil (idx));
  if (lo == hi) return values[lo];
  const double w = idx - lo;
  return values[lo] * (1.0 - w) + values[hi] * w;
}

void
SmartWifiManagerV3Logged::MaybeAdaptThresholds (SmartWifiRemoteStationV3Logged *st)
{
  // Local policy constants (replacing missing attributes)
  const uint32_t kAdaptInterval = 25; // packets
  const double kMinT1 = 4.0;
  const double kMaxT3 = 35.0;

  if (st->m_txSinceThresholdUpdate < kAdaptInterval)
    {
      return;
    }
  st->m_txSinceThresholdUpdate = 0;

  // Use recent SNR samples to re-derive T1/T2/T3
  if (st->m_snrSamples.size () < 8)
    {
      return;
    }
  std::vector<double> window = st->m_snrSamples;
  if (window.size () > 60)
    {
      window.erase (window.begin (), window.end () - 60);
    }

  const double q25 = Quantile (window, 0.25);
  const double q50 = Quantile (window, 0.50);
  const double q75 = Quantile (window, 0.75);

  double T1 = std::max (kMinT1, std::min (q25, q50 - 0.5));
  double T2 = std::max (T1 + 0.5, std::min (q50, q75 - 0.5));
  double T3 = std::max (T2 + 0.5, std::min (q75, kMaxT3));

  st->m_T1 = T1;
  st->m_T2 = T2;
  st->m_T3 = T3;

  // Keep shadow copies to log out easily
  m_traceT1 = T1;
  m_traceT2 = T2;
  m_traceT3 = T3;
}

uint8_t
SmartWifiManagerV3Logged::TierFromSnr (SmartWifiRemoteStationV3Logged *st, double snr) const
{
  (void)st; // currently unused
  if (!std::isfinite (snr)) return 0;
  return (snr > 25) ? 7 :
         (snr > 21) ? 6 :
         (snr > 18) ? 5 :
         (snr > 15) ? 4 :
         (snr > 12) ? 3 :
         (snr > 9)  ? 2 :
         (snr > 6)  ? 1 : 0;
}

void
SmartWifiManagerV3Logged::ComputeSeverityConfidence (SmartWifiRemoteStationV3Logged *st,
                                                     double &severity,
                                                     double &confidence,
                                                     uint8_t &targetTier,
                                                     uint8_t maxRateIdx)
{
  // Compute short/med success ratios
  auto ratioFromHist = [] (const std::vector<bool>& hist, uint32_t fill)
  {
    if (hist.empty () || fill == 0) return 1.0;
    uint32_t succ = 0;
    for (uint32_t i = 0; i < fill && i < hist.size (); ++i)
      {
        if (hist[i]) succ++;
      }
    return static_cast<double> (succ) / static_cast<double> (std::min<uint32_t> (fill, hist.size ()));
  };

  const double shortSucc = ratioFromHist (st->m_histShort, st->m_histShortFill);
  const double medSucc   = ratioFromHist (st->m_histMed,   st->m_histMedFill);

  const double failureRatioMed = 1.0 - medSucc;

  // Local mixing constants for severity (replacing missing attributes)
  const double kSeverityAlpha = 0.6;
  const double kSeverityBeta  = 0.4;

  // Normalize fail streak by short-history length
  const double normFailStreak = (!st->m_histShort.empty ())
                                ? std::min<double> (1.0, static_cast<double> (st->m_consecFailure) /
                                                             static_cast<double> (st->m_histShort.size ()))
                                : 0.0;

  severity = kSeverityAlpha * failureRatioMed + kSeverityBeta * normFailStreak;
  severity = std::clamp (severity, 0.0, 1.0);

  // Confidence: prefer consistency (shortSucc) and stable SNR (small |fast-slow|)
  const double trend = st->m_snrFast - st->m_snrSlow;
  const double trendPenalty = std::min (1.0, std::abs (trend) / 3.0); // |trend|>3 dB => strong penalty
  confidence = std::clamp (shortSucc * (1.0 - 0.5 * trendPenalty), 0.0, 1.0);

  // Target "tier" from SNR
  targetTier = std::min<uint8_t> (TierFromSnr (st, st->m_lastSnr), maxRateIdx);

  // Shadow copies for optional tracing
  m_traceSeverity = severity;
  m_traceConfidence = confidence;
  m_traceT1 = st->m_T1;
  m_traceT2 = st->m_T2;
  m_traceT3 = st->m_T3;
}

uint8_t
SmartWifiManagerV3Logged::ApplyDecision (SmartWifiRemoteStationV3Logged *st,
                                         uint8_t targetTier,
                                         double severity,
                                         double confidence,
                                         uint8_t maxRateIdx,
                                         int &decisionReason)
{
  // Local thresholds (replacing missing attributes)
  const double kSoftDrop = 0.45;
  const double kHardDrop = 0.75;
  const double kRaiseConf = 0.70;
  const int    kMaxUp = 1; // upstep per decision
  const double kTrendUp = +1.0;
  const double kTrendDown = -1.0;

  const double trend = st->m_snrFast - st->m_snrSlow;
  const bool trendPositive = (trend >= kTrendUp);
  const bool trendNegative = (trend <= kTrendDown);

  uint8_t newRate = st->m_rate;

  // Hard drop on severe conditions
  if (severity >= kHardDrop && st->m_consecFailure >= 2)
    {
      if (newRate > 0)
        {
          newRate = std::max<uint8_t> (0, static_cast<int> (newRate) - 2);
          decisionReason = HARD_DROP_SEVERITY;
        }
    }
  // Soft drop when conditions are poor or negative trend
  else if ((severity >= kSoftDrop || trendNegative) && st->m_consecFailure >= 1)
    {
      if (newRate > 0)
        {
          newRate -= 1;
          decisionReason = SOFT_DROP_SEVERITY;
        }
    }
  else
    {
      // Consider raising if we're close to target and confident
      if (confidence >= kRaiseConf && st->m_consecSuccess >= 1)
        {
          // be a little bolder if trend is positive
          int up = trendPositive ? std::min (2, kMaxUp + 1) : kMaxUp;
          newRate = std::min<uint8_t> (static_cast<uint8_t> (newRate + up), std::min<uint8_t> (targetTier, maxRateIdx));
          if (newRate > st->m_rate)
            {
              decisionReason = RAISE_CONFIRMED;
            }
        }
      else
        {
          decisionReason = HOLD_STABLE;
        }
    }

  // Apply and track stability
  if (newRate != st->m_rate)
    {
      uint64_t oldRate = st->m_rate;        // FIX: store old rate
      st->m_rate = newRate;
      st->m_sinceLastRateChange = 0;

      // FIX: Fire RateChange trace
      m_rateChange(static_cast<uint64_t>(newRate),
                   static_cast<uint64_t>(oldRate));
    }
  else
    {
      st->m_sinceLastRateChange++;
    }

  return st->m_rate;
}

// -----------------------------------------------------------------------------
// PHASE1 NEW CODE: Stratified logging + feature helpers
// -----------------------------------------------------------------------------
double
SmartWifiManagerV3Logged::GetStratifiedLogProbability (uint8_t rate, double snr, bool success)
{
  // Favor low-rate, low-SNR, and failure cases; de-emphasize easy/saturated cases.
  const double base[8] = {1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.08};
  const uint8_t idx = std::min<uint8_t> (rate, 7);
  double p = base[idx];
  if (!success) p *= 2.0;
  if (snr < 15.0) p *= 1.5;
  if (idx <= 1) p = 1.0;
  if (idx >= 6 && snr > 25.0 && success) p *= 0.5;
  return std::min (1.0, p);
}

double
SmartWifiManagerV3Logged::GetRandomValue ()
{
  return m_uniformDist (m_rng);
}

uint32_t
SmartWifiManagerV3Logged::CountRecentRateChanges (SmartWifiRemoteStationV3Logged *st, uint32_t window)
{
  uint32_t changes = 0;
  if (st->m_rateHistory.size () > 1)
    {
      const size_t start = (st->m_rateHistory.size () > window)
                             ? st->m_rateHistory.size () - window
                             : 1;
      for (size_t i = start; i < st->m_rateHistory.size (); ++i)
        {
          if (st->m_rateHistory[i] != st->m_rateHistory[i - 1]) changes++;
        }
    }
  return changes;
}

double
SmartWifiManagerV3Logged::CalculateRateStability (SmartWifiRemoteStationV3Logged *st)
{
  const double denom = 20.0;
  const double changes = static_cast<double> (CountRecentRateChanges (st, 20));
  return std::clamp (1.0 - (changes / denom), 0.0, 1.0);
}

double
SmartWifiManagerV3Logged::CalculateRecentThroughput (SmartWifiRemoteStationV3Logged *st, uint32_t window)
{
  if (st->m_throughputHistory.empty ()) return 0.0;
  const size_t start = (st->m_throughputHistory.size () > window)
                         ? st->m_throughputHistory.size () - window
                         : 0;
  double sum = 0.0;
  uint32_t n = 0;
  for (size_t i = start; i < st->m_throughputHistory.size (); ++i)
    {
      sum += st->m_throughputHistory[i];
      n++;
    }
  return (n > 0) ? (sum / n) : 0.0;
}

double
SmartWifiManagerV3Logged::CalculateRecentPacketLoss (SmartWifiRemoteStationV3Logged *st, uint32_t window)
{
  if (st->m_packetSuccessHistory.empty ()) return 0.0;
  const size_t start = (st->m_packetSuccessHistory.size () > window)
                         ? st->m_packetSuccessHistory.size () - window
                         : 0;
  uint32_t total = 0, fails = 0;
  for (size_t i = start; i < st->m_packetSuccessHistory.size (); ++i)
    {
      total++;
      if (!st->m_packetSuccessHistory[i]) fails++;
    }
  return (total > 0) ? static_cast<double> (fails) / static_cast<double> (total) : 0.0;
}

double
SmartWifiManagerV3Logged::CalculateRetrySuccessRatio (SmartWifiRemoteStationV3Logged *st)
{
  uint32_t succ = 0;
  uint32_t totalRetries = 0;
  const size_t n = std::min (st->m_packetSuccessHistory.size (), st->m_retryHistory.size ());
  for (size_t i = 0; i < n; ++i)
    {
      if (st->m_packetSuccessHistory[i]) succ++;
      totalRetries += st->m_retryHistory[i];
    }
  // +1 to avoid div-by-0 and temper the ratio
  return (succ > 0) ? static_cast<double> (succ) / static_cast<double> (totalRetries + 1) : 0.0;
}

double
SmartWifiManagerV3Logged::CalculateOptimalRateDistance (SmartWifiRemoteStationV3Logged *st)
{
  const uint8_t opt = TierFromSnr (st, st->m_lastSnr);
  const int d = static_cast<int> (st->m_rate) - static_cast<int> (opt);
  return std::min (1.0, std::abs (d) / 7.0);
}

double
SmartWifiManagerV3Logged::CalculateAggressiveFactor (SmartWifiRemoteStationV3Logged *st)
{
  if (st->m_rateHistory.empty ()) return 0.0;
  uint32_t cnt = 0;
  for (uint8_t r : st->m_rateHistory) if (r >= 6) cnt++;
  return static_cast<double> (cnt) / static_cast<double> (st->m_rateHistory.size ());
}

double
SmartWifiManagerV3Logged::CalculateConservativeFactor (SmartWifiRemoteStationV3Logged *st)
{
  if (st->m_rateHistory.empty ()) return 0.0;
  uint32_t cnt = 0;
  for (uint8_t r : st->m_rateHistory) if (r <= 2) cnt++;
  return static_cast<double> (cnt) / static_cast<double> (st->m_rateHistory.size ());
}

uint8_t
SmartWifiManagerV3Logged::GetRecommendedSafeRate (SmartWifiRemoteStationV3Logged *st)
{
  return TierFromSnr (st, st->m_lastSnr);
}

double
SmartWifiManagerV3Logged::CalculateSnrStability (SmartWifiRemoteStationV3Logged *st)
{
  const size_t window = std::min<size_t> (10, st->m_snrSamples.size ());
  if (window < 2) return 0.0;
  const size_t start = st->m_snrSamples.size () - window;
  double mean = 0.0;
  for (size_t i = start; i < st->m_snrSamples.size (); ++i) mean += st->m_snrSamples[i];
  mean /= window;
  double var = 0.0;
  for (size_t i = start; i < st->m_snrSamples.size (); ++i)
    {
      const double d = st->m_snrSamples[i] - mean;
      var += d * d;
    }
  return std::sqrt (var / window);
}

double
SmartWifiManagerV3Logged::CalculateSnrPredictionConfidence (SmartWifiRemoteStationV3Logged *st)
{
  const double stability = CalculateSnrStability (st);
  return 1.0 / (1.0 + stability);
}

void
SmartWifiManagerV3Logged::MaybeRelaxRaiseThreshold (SmartWifiRemoteStationV3Logged *st)
{
  // This function is declared in the header but not implemented in the original code
  // Adding a basic implementation to prevent linker errors
  (void)st; // Suppress unused parameter warning
  // Implementation can be added here if needed
}

// -----------------------------------------------------------------------------
// PHASE1 NEW CODE: Decision logging with stratified sampling
// -----------------------------------------------------------------------------
void
SmartWifiManagerV3Logged::LogDecision (SmartWifiRemoteStationV3Logged *st, int decisionReason, bool packetSuccess)
{
  // Stratified sampling guard
  const uint8_t maxRateIdx = 7;
  const uint8_t validatedRateIdx = std::min<uint8_t> (st->m_rate, maxRateIdx);
  const double lastSnr = std::isfinite (st->m_lastSnr) ? st->m_lastSnr : -99.0;

  const double logProb = GetStratifiedLogProbability (validatedRateIdx, lastSnr, packetSuccess);
  if (GetRandomValue () > logProb)
    {
      return; // skip over-sampled cases
    }

  if (!m_logFile.is_open ())
    {
      return;
    }

  // Build station id string
  std::ostringstream stationId;
  if (st->m_node) stationId << st->m_node->GetId ();
  else stationId << reinterpret_cast<uintptr_t> (st);

  // PHY rate placeholder: linear mapping by index (replace with true rate if available)
  const uint64_t phyRate = 1000000ull + static_cast<uint64_t> (validatedRateIdx) * 1000000ull;

  // Success ratios
  auto ratioFromHist = [] (const std::vector<bool>& hist, uint32_t fill)
  {
    if (hist.empty () || fill == 0) return 1.0;
    uint32_t succ = 0;
    for (uint32_t i = 0; i < fill && i < hist.size (); ++i)
      {
        if (hist[i]) succ++;
      }
    return static_cast<double> (succ) / static_cast<double> (std::min<uint32_t> (fill, hist.size ()));
  };
  const double shortSuccRatio = ratioFromHist (st->m_histShort, st->m_histShortFill);
  const double medSuccRatio   = ratioFromHist (st->m_histMed,   st->m_histMedFill);

  // Feedback features
  const double snrFast = std::isfinite (st->m_snrFast) ? st->m_snrFast : lastSnr;
  const double snrSlow = std::isfinite (st->m_snrSlow) ? st->m_snrSlow : lastSnr;
  const double snrTrendShort = snrFast - snrSlow;
  const double snrStabilityIndex = CalculateSnrStability (st);
  const double snrPredictionConfidence = CalculateSnrPredictionConfidence (st);
  const double recentThroughputTrend = CalculateRecentThroughput (st, 10);
  const double packetLossRate = CalculateRecentPacketLoss (st, 20);
  const double retrySuccessRatio = CalculateRetrySuccessRatio (st);
  const uint32_t recentRateChanges = CountRecentRateChanges (st, 20);
  const uint32_t timeSinceLastRateChange = st->m_sinceLastRateChange;
  const double rateStabilityScore = CalculateRateStability (st);
  const double optimalRateDistance = CalculateOptimalRateDistance (st);
  const double aggressiveFactor = CalculateAggressiveFactor (st);
  const double conservativeFactor = CalculateConservativeFactor (st);
  const uint8_t recommendedSafeRate = GetRecommendedSafeRate (st);

  // Variance over recent SNR sample window
  double snrVariance = 0.0;
  if (st->m_snrSamples.size () >= 2)
    {
      const size_t window = std::min<size_t> (st->m_snrSamples.size (), 20);
      const size_t start = st->m_snrSamples.size () - window;
      double mean = 0.0;
      for (size_t i = start; i < st->m_snrSamples.size (); ++i) mean += st->m_snrSamples[i];
      mean /= window;
      double var = 0.0;
      for (size_t i = start; i < st->m_snrSamples.size (); ++i)
        {
          const double d = st->m_snrSamples[i] - mean;
          var += d * d;
        }
      snrVariance = var / window;
      if (!std::isfinite (snrVariance)) snrVariance = 0.0;
    }

  // Defaults for fields we don't currently observe in this minimal integration
  const double offeredLoad = 0.0;
  const int queueLen = 0;
  const int retryCount = 0;
  const double mobilityMetric = 0.0;
  const uint16_t channelWidth = 20; // MHz (placeholder)

  const double simTime = Simulator::Now ().GetSeconds ();
  const uint32_t consecSuccess = st->m_consecSuccess;
  const uint32_t consecFailure = st->m_consecFailure;
  const double severity = std::isfinite (st->m_severity) ? st->m_severity : 0.0;
  const double confidence = std::isfinite (st->m_confidence) ? st->m_confidence : 0.0;
  const double T1 = std::isfinite (st->m_T1) ? st->m_T1 : 10.0;
  const double T2 = std::isfinite (st->m_T2) ? st->m_T2 : 15.0;
  const double T3 = std::isfinite (st->m_T3) ? st->m_T3 : 25.0;

  m_logFile << std::fixed << std::setprecision (6)
            << simTime << "," << stationId.str () << ","
            << static_cast<int> (validatedRateIdx) << "," << phyRate << ","
            << lastSnr << "," << snrFast << "," << snrSlow << ","
            << snrTrendShort << "," << snrStabilityIndex << "," << snrPredictionConfidence << ","
            << shortSuccRatio << "," << medSuccRatio << ","
            << consecSuccess << "," << consecFailure << ","
            << recentThroughputTrend << "," << packetLossRate << "," << retrySuccessRatio << ","
            << recentRateChanges << "," << timeSinceLastRateChange << "," << rateStabilityScore << ","
            << optimalRateDistance << "," << aggressiveFactor << "," << conservativeFactor << "," << static_cast<int> (recommendedSafeRate) << ","
            << severity << "," << confidence << ","
            << T1 << "," << T2 << "," << T3 << ","
            << decisionReason << "," << (packetSuccess ? 1 : 0) << ","
            << offeredLoad << "," << queueLen << "," << retryCount << ","
            << channelWidth << "," << mobilityMetric << "," << snrVariance << "\n";
  m_logFile.flush ();
}

} // namespace ns3