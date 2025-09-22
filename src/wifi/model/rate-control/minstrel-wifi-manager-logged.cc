/* minstrel-wifi-manager-logged.cc
 *
 * Logged Minstrel implementation (Phase 1)
 * - Based on original MinstrelWifiManager logic (preserve algorithm)
 * - Adds stratified probabilistic logging and feedback-oriented features
 * - Follows the style of SmartWifiManagerV3Logged for helpers and CSV logging
 */

#include "minstrel-wifi-manager-logged.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-phy.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("MinstrelWifiManagerLogged");
NS_OBJECT_ENSURE_REGISTERED(MinstrelWifiManagerLogged);

TypeId
MinstrelWifiManagerLogged::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::MinstrelWifiManagerLogged")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<MinstrelWifiManagerLogged>()
            .AddAttribute("UpdateStatistics",
                          "The interval between updating statistics table",
                          TimeValue(Seconds(0.1)),
                          MakeTimeAccessor(&MinstrelWifiManagerLogged::m_updateStats),
                          MakeTimeChecker())
            .AddAttribute("LookAroundRate",
                          "The percentage to try other rates",
                          UintegerValue(10),
                          MakeUintegerAccessor(&MinstrelWifiManagerLogged::m_lookAroundRate),
                          MakeUintegerChecker<uint8_t>())
            .AddAttribute("EwmaLevel",
                          "EWMA level",
                          UintegerValue(75),
                          MakeUintegerAccessor(&MinstrelWifiManagerLogged::m_ewmaLevel),
                          MakeUintegerChecker<uint8_t>())
            .AddAttribute("SampleColumn",
                          "The number of columns used for sampling",
                          UintegerValue(10),
                          MakeUintegerAccessor(&MinstrelWifiManagerLogged::m_sampleCol),
                          MakeUintegerChecker<uint8_t>())
            .AddAttribute("PacketLength",
                          "The packet length used for calculating mode TxTime",
                          UintegerValue(1200),
                          MakeUintegerAccessor(&MinstrelWifiManagerLogged::m_pktLen),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("PrintStats",
                          "Print statistics table",
                          BooleanValue(false),
                          MakeBooleanAccessor(&MinstrelWifiManagerLogged::m_printStats),
                          MakeBooleanChecker())
            .AddAttribute("PrintSamples",
                          "Print samples table",
                          BooleanValue(false),
                          MakeBooleanAccessor(&MinstrelWifiManagerLogged::m_printSamples),
                          MakeBooleanChecker())
            .AddAttribute("LogFilePath",
                          "CSV log output path for MinstrelWifiManagerLogged.",
                          StringValue(""),
                          MakeStringAccessor(&MinstrelWifiManagerLogged::m_logFilePath),
                          MakeStringChecker())
            // Trace sources: map to the traced members in the header
            .AddTraceSource(
                "PacketResult",
                "Emits (stationId, success) per data attempt.",
                MakeTraceSourceAccessor(&MinstrelWifiManagerLogged::m_packetResultTrace),
                "ns3::TracedCallback::Context")
            .AddTraceSource("RateChange",
                            "Emits (newRate, oldRate) when data rate changes",
                            MakeTraceSourceAccessor(&MinstrelWifiManagerLogged::m_rateChange),
                            "ns3::TracedCallback::Uint64Uint64");
    return tid;
}

MinstrelWifiManagerLogged::MinstrelWifiManagerLogged()
    : WifiRemoteStationManager(),
      m_rng(std::random_device{}()),
      m_uniformDist(0.0, 1.0),
      m_updateStats(Seconds(0.1)),
      m_lookAroundRate(10),
      m_ewmaLevel(75),
      m_sampleCol(10),
      m_pktLen(1200),
      m_printStats(false),
      m_printSamples(false),
      m_uniformRandomVariable(CreateObject<UniformRandomVariable>()),
      m_logFile(),
      m_logFilePath(),
      m_logHeaderWritten(false),
      m_currentRate(0),
      m_traceConfidence(0.0),
      m_traceSeverity(0.0),
      m_traceT1(0.0),
      m_traceT2(0.0),
      m_traceT3(0.0)
{
    NS_LOG_FUNCTION(this);
}

MinstrelWifiManagerLogged::~MinstrelWifiManagerLogged()
{
    NS_LOG_FUNCTION(this);
    if (m_logFile.is_open())
    {
        m_logFile.close();
    }
}

void
MinstrelWifiManagerLogged::SetupPhy(const Ptr<WifiPhy> phy)
{
    NS_LOG_FUNCTION(this << phy);
    for (const auto& mode : phy->GetModeList())
    {
        WifiTxVector txVector;
        txVector.SetMode(mode);
        txVector.SetPreambleType(WIFI_PREAMBLE_LONG);
        AddCalcTxTime(mode, phy->CalculateTxDuration(m_pktLen, txVector, phy->GetPhyBand()));
    }
    WifiRemoteStationManager::SetupPhy(phy);
}

void
MinstrelWifiManagerLogged::SetupMac(const Ptr<WifiMac> mac)
{
    NS_LOG_FUNCTION(this << mac);
    WifiRemoteStationManager::SetupMac(mac);
}

void
MinstrelWifiManagerLogged::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    if (GetHtSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support HT rates");
    }
    if (GetVhtSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support VHT rates");
    }
    if (GetHeSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support HE rates");
    }

    // open log file if requested and write CSV header
    if (!m_logFilePath.empty() && !m_logHeaderWritten)
    {
        m_logFile.open(m_logFilePath, std::ios::out | std::ios::trunc);
        if (!m_logFile.is_open())
        {
            NS_LOG_ERROR("Failed to open log file: " << m_logFilePath);
        }
        else
        {
            m_logFile
                << "time,stationId,txRateIdx,phyRate,lastSnr,"
                   "consecSuccess,consecFailure,recentThroughput,packetLossRate,retrySuccessRatio,"
                   "recentRateChanges,timeSinceLastRateChange,rateStabilityScore,"
                   "optimalRateDistance,aggressiveFactor,conservativeFactor,recommendedSafeRate,"
                   "decisionReason,packetSuccess,sampleFlag,offeredLoad,queueLen,retryCount,"
                   "channelWidth,snrVariance\n";
            m_logFile.flush();
            m_logHeaderWritten = true;
        }
    }
}

int64_t
MinstrelWifiManagerLogged::AssignStreams(int64_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_uniformRandomVariable->SetStream(stream);
    return 1;
}

Time
MinstrelWifiManagerLogged::GetCalcTxTime(WifiMode mode) const
{
    NS_LOG_FUNCTION(this << mode);
    auto it = m_calcTxTime.find(mode);
    NS_ASSERT(it != m_calcTxTime.end());
    return it->second;
}

void
MinstrelWifiManagerLogged::AddCalcTxTime(WifiMode mode, Time t)
{
    NS_LOG_FUNCTION(this << mode << t);
    m_calcTxTime.insert(std::make_pair(mode, t));
}

WifiRemoteStation*
MinstrelWifiManagerLogged::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    auto station = new MinstrelWifiRemoteStationLogged();

    station->m_nextStatsUpdate = Simulator::Now() + m_updateStats;
    station->m_col = 0;
    station->m_index = 0;
    station->m_maxTpRate = 0;
    station->m_maxTpRate2 = 0;
    station->m_maxProbRate = 0;
    station->m_nModes = 0;
    station->m_totalPacketsCount = 0;
    station->m_samplePacketsCount = 0;
    station->m_numSamplesDeferred = 0;
    station->m_isSampling = false;
    station->m_sampleRate = 0;
    station->m_sampleDeferred = false;
    station->m_shortRetry = 0;
    station->m_longRetry = 0;
    station->m_retry = 0;
    station->m_txrate = 0;
    station->m_initialized = false;

    // feedback buffers
    station->m_rateHistory.clear();
    station->m_throughputHistory.clear();
    station->m_retryHistory.clear();
    station->m_packetSuccessHistory.clear();

    station->m_lastSnr = 0.0;
    station->m_sinceLastRateChange = 0;
    station->m_consecSuccess = 0;
    station->m_consecFailure = 0;
    station->m_node = nullptr;

    return station;
}

void
MinstrelWifiManagerLogged::CheckInit(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    if (!station->m_initialized && GetNSupported(station) > 1)
    {
        station->m_nModes = GetNSupported(station);
        station->m_minstrelTable = MinstrelRateLogged(station->m_nModes);
        station->m_sampleTable =
            SampleRateLogged(station->m_nModes, std::vector<uint8_t>(m_sampleCol));
        InitSampleTable(station);
        RateInit(station);
        station->m_initialized = true;
    }
}

/* The original Minstrel UpdateRate is preserved */
void
MinstrelWifiManagerLogged::UpdateRate(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    station->m_longRetry++;
    station->m_minstrelTable[station->m_txrate].numRateAttempt++;

    // original logic unchanged (copied verbatim for fidelity)
    if (!station->m_isSampling)
    {
        if (station->m_longRetry <
            station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount)
        {
            station->m_txrate = station->m_maxTpRate;
        }
        else if (station->m_longRetry <=
                 (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount))
        {
            station->m_txrate = station->m_maxTpRate2;
        }
        else if (station->m_longRetry <=
                 (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
        {
            station->m_txrate = station->m_maxProbRate;
        }
        else if (station->m_longRetry >
                 (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
        {
            station->m_txrate = 0;
        }
    }
    else
    {
        if (station->m_sampleDeferred)
        {
            if (station->m_longRetry <
                station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount)
            {
                station->m_txrate = station->m_maxTpRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount))
            {
                station->m_txrate = station->m_sampleRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                station->m_txrate = station->m_maxProbRate;
            }
            else if (station->m_longRetry >
                     (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                station->m_txrate = 0;
            }
        }
        else
        {
            if (station->m_longRetry <
                station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount)
            {
                station->m_txrate = station->m_sampleRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount))
            {
                station->m_txrate = station->m_maxTpRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                station->m_txrate = station->m_maxProbRate;
            }
            else if (station->m_longRetry >
                     (station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                station->m_txrate = 0;
            }
        }
    }
}

/* GetDataTxVector: keep original behavior, but update m_currentRate trace and emit rate-change
 * trace */
WifiTxVector
MinstrelWifiManagerLogged::GetDataTxVector(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    uint16_t channelWidth = GetChannelWidth(station);
    if (channelWidth > 20 && channelWidth != 22)
    {
        channelWidth = 20;
    }
    if (!station->m_initialized)
    {
        CheckInit(station);
    }
    WifiMode mode = GetSupported(station, station->m_txrate);
    uint64_t rate = mode.GetDataRate(channelWidth);
    if (m_currentRate != rate && !station->m_isSampling)
    {
        uint64_t oldRate = m_currentRate;
        m_currentRate = rate;
        // trace as (newRate, oldRate)
        m_rateChange(static_cast<uint64_t>(m_currentRate), static_cast<uint64_t>(oldRate));
    }
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        800,
        1,
        1,
        0,
        channelWidth,
        GetAggregation(station));
}

WifiTxVector
MinstrelWifiManagerLogged::GetRtsTxVector(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    uint16_t channelWidth = GetChannelWidth(station);
    if (channelWidth > 20 && channelWidth != 22)
    {
        channelWidth = 20;
    }
    WifiMode mode;
    if (!GetUseNonErpProtection())
    {
        mode = GetSupported(station, 0);
    }
    else
    {
        mode = GetNonErpSupported(station, 0);
    }
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        800,
        1,
        1,
        0,
        channelWidth,
        GetAggregation(station));
}

uint32_t
MinstrelWifiManagerLogged::CountRetries(MinstrelWifiRemoteStationLogged* station)
{
    if (!station->m_isSampling)
    {
        return station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
               station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount +
               station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount +
               station->m_minstrelTable[0].adjustedRetryCount;
    }
    else
    {
        return station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
               station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
               station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount +
               station->m_minstrelTable[0].adjustedRetryCount;
    }
}

uint16_t
MinstrelWifiManagerLogged::FindRate(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);

    if (station->m_totalPacketsCount == 0)
    {
        return 0;
    }

    uint16_t idx = 0;
    int delta = (station->m_totalPacketsCount * m_lookAroundRate / 100) -
                (station->m_samplePacketsCount + station->m_numSamplesDeferred / 2);

    if (delta >= 0)
    {
        idx = GetNextSample(station);

        if (idx >= station->m_nModes)
        {
            NS_LOG_DEBUG("ALERT!!! ERROR - sample index out of range");
        }

        station->m_sampleRate = idx;

        if ((station->m_minstrelTable[idx].perfectTxTime >
             station->m_minstrelTable[station->m_maxTpRate].perfectTxTime) &&
            (station->m_minstrelTable[idx].numSamplesSkipped < 20))
        {
            station->m_sampleDeferred = true;
            station->m_numSamplesDeferred++;
            station->m_isSampling = true;
        }
        else
        {
            if (!station->m_minstrelTable[idx].sampleLimit)
            {
                idx = station->m_maxTpRate;
                station->m_isSampling = false;
            }
            else
            {
                station->m_isSampling = true;
                if (station->m_minstrelTable[idx].sampleLimit > 0)
                {
                    station->m_minstrelTable[idx].sampleLimit--;
                }
            }
        }

        if (station->m_sampleDeferred)
        {
            idx = station->m_maxTpRate;
        }
    }
    else
    {
        idx = station->m_maxTpRate;
    }

    return idx;
}

void
MinstrelWifiManagerLogged::UpdateStats(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    if (Simulator::Now() < station->m_nextStatsUpdate)
    {
        return;
    }

    if (!station->m_initialized)
    {
        return;
    }

    station->m_nextStatsUpdate = Simulator::Now() + m_updateStats;

    Time txTime;
    uint32_t tempProb;

    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        txTime = station->m_minstrelTable[i].perfectTxTime;
        if (txTime.GetMicroSeconds() == 0)
        {
            txTime = Seconds(1);
        }

        if (station->m_minstrelTable[i].numRateAttempt)
        {
            station->m_minstrelTable[i].numSamplesSkipped = 0;
            tempProb = (station->m_minstrelTable[i].numRateSuccess * 18000) /
                       station->m_minstrelTable[i].numRateAttempt;
            station->m_minstrelTable[i].prob = tempProb;

            if (station->m_minstrelTable[i].successHist == 0)
            {
                station->m_minstrelTable[i].ewmaProb = tempProb;
            }
            else
            {
                tempProb = ((tempProb * (100 - m_ewmaLevel)) +
                            (station->m_minstrelTable[i].ewmaProb * m_ewmaLevel)) /
                           100;
                station->m_minstrelTable[i].ewmaProb = tempProb;
            }

            station->m_minstrelTable[i].throughput =
                tempProb * static_cast<uint32_t>(1000000 / txTime.GetMicroSeconds());
        }
        else
        {
            station->m_minstrelTable[i].numSamplesSkipped++;
        }

        station->m_minstrelTable[i].successHist += station->m_minstrelTable[i].numRateSuccess;
        station->m_minstrelTable[i].attemptHist += station->m_minstrelTable[i].numRateAttempt;
        station->m_minstrelTable[i].prevNumRateSuccess = station->m_minstrelTable[i].numRateSuccess;
        station->m_minstrelTable[i].prevNumRateAttempt = station->m_minstrelTable[i].numRateAttempt;
        station->m_minstrelTable[i].numRateSuccess = 0;
        station->m_minstrelTable[i].numRateAttempt = 0;

        if ((station->m_minstrelTable[i].ewmaProb > 17100) ||
            (station->m_minstrelTable[i].ewmaProb < 1800))
        {
            if (station->m_minstrelTable[i].retryCount > 2)
            {
                station->m_minstrelTable[i].adjustedRetryCount = 2;
            }
            station->m_minstrelTable[i].sampleLimit = 4;
        }
        else
        {
            station->m_minstrelTable[i].sampleLimit = -1;
            station->m_minstrelTable[i].adjustedRetryCount = station->m_minstrelTable[i].retryCount;
        }

        if (station->m_minstrelTable[i].adjustedRetryCount == 0)
        {
            station->m_minstrelTable[i].adjustedRetryCount = 2;
        }
    }

    uint32_t max_tp = 0;
    uint8_t index_max_tp = 0;
    uint8_t index_max_tp2 = 0;

    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        if (max_tp < station->m_minstrelTable[i].throughput)
        {
            index_max_tp = i;
            max_tp = station->m_minstrelTable[i].throughput;
        }
    }

    max_tp = 0;
    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        if ((i != index_max_tp) && (max_tp < station->m_minstrelTable[i].throughput))
        {
            index_max_tp2 = i;
            max_tp = station->m_minstrelTable[i].throughput;
        }
    }

    uint32_t max_prob = 0;
    uint8_t index_max_prob = 0;
    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        if ((station->m_minstrelTable[i].ewmaProb >= 95 * 180) &&
            (station->m_minstrelTable[i].throughput >=
             station->m_minstrelTable[index_max_prob].throughput))
        {
            index_max_prob = i;
            max_prob = station->m_minstrelTable[i].ewmaProb;
        }
        else if (station->m_minstrelTable[i].ewmaProb >= max_prob)
        {
            index_max_prob = i;
            max_prob = station->m_minstrelTable[i].ewmaProb;
        }
    }

    station->m_maxTpRate = index_max_tp;
    station->m_maxTpRate2 = index_max_tp2;
    station->m_maxProbRate = index_max_prob;

    if (index_max_tp > station->m_txrate)
    {
        station->m_txrate = index_max_tp;
    }

    if (m_printStats)
    {
        PrintTable(station);
    }
    if (m_printSamples)
    {
        PrintSampleTable(station);
    }
}

void
MinstrelWifiManagerLogged::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
    auto station = Lookup(st);
    if (std::isfinite(rxSnr))
    {
        station->m_lastSnr = rxSnr;
    }
}

void
MinstrelWifiManagerLogged::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = Lookup(st);
    station->m_shortRetry++;
}

void
MinstrelWifiManagerLogged::DoReportRtsOk(WifiRemoteStation* st,
                                         double ctsSnr,
                                         WifiMode ctsMode,
                                         double rtsSnr)
{
    NS_LOG_FUNCTION(this << st << ctsSnr << ctsMode << rtsSnr);
}

void
MinstrelWifiManagerLogged::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = Lookup(st);
    UpdateRetry(station);
}

void
MinstrelWifiManagerLogged::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = Lookup(st);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return;
    }

    // update feedback history
    station->m_rateHistory.push_back(station->m_txrate);
    if (station->m_rateHistory.size() > 20)
        station->m_rateHistory.erase(station->m_rateHistory.begin());

    station->m_throughputHistory.push_back(0.0);
    if (station->m_throughputHistory.size() > 20)
        station->m_throughputHistory.erase(station->m_throughputHistory.begin());

    station->m_packetSuccessHistory.push_back(false);
    if (station->m_packetSuccessHistory.size() > 20)
        station->m_packetSuccessHistory.erase(station->m_packetSuccessHistory.begin());

    station->m_retryHistory.push_back(1);
    if (station->m_retryHistory.size() > 20)
        station->m_retryHistory.erase(station->m_retryHistory.begin());

    // increment failure streak
    station->m_consecFailure++;
    station->m_consecSuccess = 0;

    // trace and log
    std::ostringstream sid;
    if (station->m_node)
        sid << station->m_node->GetId();
    else
        sid << reinterpret_cast<uintptr_t>(station);
    m_packetResultTrace(sid.str(), false);

    int decisionReason =
        HOLD_STABLE; // we log context; algorithm will react via UpdateRate/FindRate
    LogDecision(station, decisionReason, false);

    UpdateRate(station);
}

void
MinstrelWifiManagerLogged::DoReportDataOk(WifiRemoteStation* st,
                                          double ackSnr,
                                          WifiMode ackMode,
                                          double dataSnr,
                                          uint16_t dataChannelWidth,
                                          uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
    auto station = Lookup(st);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return;
    }

    station->m_minstrelTable[station->m_txrate].numRateSuccess++;
    station->m_minstrelTable[station->m_txrate].numRateAttempt++;

    UpdatePacketCounters(station);

    // feedback buffers update
    station->m_rateHistory.push_back(station->m_txrate);
    if (station->m_rateHistory.size() > 20)
        station->m_rateHistory.erase(station->m_rateHistory.begin());

    station->m_throughputHistory.push_back(static_cast<double>(station->m_txrate));
    if (station->m_throughputHistory.size() > 20)
        station->m_throughputHistory.erase(station->m_throughputHistory.begin());

    station->m_packetSuccessHistory.push_back(true);
    if (station->m_packetSuccessHistory.size() > 20)
        station->m_packetSuccessHistory.erase(station->m_packetSuccessHistory.begin());

    station->m_retryHistory.push_back(0);
    if (station->m_retryHistory.size() > 20)
        station->m_retryHistory.erase(station->m_retryHistory.begin());

    station->m_consecSuccess++;
    station->m_consecFailure = 0;

    UpdateRetry(station);
    UpdateStats(station);

    if (station->m_nModes >= 1)
    {
        station->m_txrate = FindRate(station);
    }

    // emit packet result trace
    std::ostringstream sid;
    if (station->m_node)
        sid << station->m_node->GetId();
    else
        sid << reinterpret_cast<uintptr_t>(station);
    m_packetResultTrace(sid.str(), true);

    int decisionReason = HOLD_STABLE;
    LogDecision(station, decisionReason, true);
}

void
MinstrelWifiManagerLogged::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = Lookup(st);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return;
    }

    UpdatePacketCounters(station);

    UpdateRetry(station);
    UpdateStats(station);

    if (station->m_nModes >= 1)
    {
        station->m_txrate = FindRate(station);
    }
}

void
MinstrelWifiManagerLogged::UpdatePacketCounters(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);

    station->m_totalPacketsCount++;
    if (station->m_isSampling &&
        (!station->m_sampleDeferred ||
         station->m_longRetry >= station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount))
    {
        station->m_samplePacketsCount++;
    }

    if (station->m_numSamplesDeferred > 0)
    {
        station->m_numSamplesDeferred--;
    }

    if (station->m_totalPacketsCount == ~0)
    {
        station->m_numSamplesDeferred = 0;
        station->m_samplePacketsCount = 0;
        station->m_totalPacketsCount = 0;
    }
    station->m_isSampling = false;
    station->m_sampleDeferred = false;
}

void
MinstrelWifiManagerLogged::UpdateRetry(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    station->m_retry = station->m_shortRetry + station->m_longRetry;
    station->m_shortRetry = 0;
    station->m_longRetry = 0;
}

WifiTxVector
MinstrelWifiManagerLogged::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    auto station = Lookup(st);
    return GetDataTxVector(station);
}

WifiTxVector
MinstrelWifiManagerLogged::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = Lookup(st);
    return GetRtsTxVector(station);
}

bool
MinstrelWifiManagerLogged::DoNeedRetransmission(WifiRemoteStation* st,
                                                Ptr<const Packet> packet,
                                                bool normally)
{
    NS_LOG_FUNCTION(this << st << packet << normally);
    auto station = Lookup(st);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return normally;
    }
    if (station->m_longRetry >= CountRetries(station))
    {
        NS_LOG_DEBUG("No re-transmission allowed. Retries: "
                     << station->m_longRetry << " Max retries: " << CountRetries(station));
        return false;
    }
    else
    {
        NS_LOG_DEBUG("Re-transmit. Retries: " << station->m_longRetry
                                              << " Max retries: " << CountRetries(station));
        return true;
    }
}

uint16_t
MinstrelWifiManagerLogged::GetNextSample(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    uint16_t bitrate;
    bitrate = station->m_sampleTable[station->m_index][station->m_col];
    station->m_index++;

    NS_ABORT_MSG_IF(station->m_nModes < 2, "Integer overflow detected");
    if (station->m_index > station->m_nModes - 2)
    {
        station->m_index = 0;
        station->m_col++;
        if (station->m_col >= m_sampleCol)
        {
            station->m_col = 0;
        }
    }
    return bitrate;
}

void
MinstrelWifiManagerLogged::RateInit(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        station->m_minstrelTable[i].numRateAttempt = 0;
        station->m_minstrelTable[i].numRateSuccess = 0;
        station->m_minstrelTable[i].prevNumRateSuccess = 0;
        station->m_minstrelTable[i].prevNumRateAttempt = 0;
        station->m_minstrelTable[i].successHist = 0;
        station->m_minstrelTable[i].attemptHist = 0;
        station->m_minstrelTable[i].numSamplesSkipped = 0;
        station->m_minstrelTable[i].prob = 0;
        station->m_minstrelTable[i].ewmaProb = 0;
        station->m_minstrelTable[i].throughput = 0;
        station->m_minstrelTable[i].perfectTxTime = GetCalcTxTime(GetSupported(station, i));
        station->m_minstrelTable[i].retryCount = 1;
        station->m_minstrelTable[i].adjustedRetryCount = 1;

        Time totalTxTimeWithGivenRetries = Seconds(0.0);
        for (uint32_t retries = 2; retries < 11; retries++)
        {
            totalTxTimeWithGivenRetries =
                CalculateTimeUnicastPacket(station->m_minstrelTable[i].perfectTxTime, 0, retries);
            if (totalTxTimeWithGivenRetries > MilliSeconds(6))
            {
                break;
            }
            station->m_minstrelTable[i].sampleLimit = -1;
            station->m_minstrelTable[i].retryCount = retries;
            station->m_minstrelTable[i].adjustedRetryCount = retries;
        }
    }
    UpdateStats(station);
}

Time
MinstrelWifiManagerLogged::CalculateTimeUnicastPacket(Time dataTransmissionTime,
                                                      uint32_t shortRetries,
                                                      uint32_t longRetries)
{
    NS_LOG_FUNCTION(this << dataTransmissionTime << shortRetries << longRetries);

    Time tt = dataTransmissionTime + GetPhy()->GetSifs() + GetPhy()->GetAckTxTime();

    uint32_t cwMax = 1023;
    uint32_t cw = 31;
    for (uint32_t retry = 0; retry < longRetries; retry++)
    {
        tt += dataTransmissionTime + GetPhy()->GetSifs() + GetPhy()->GetAckTxTime();
        tt += (cw / 2.0) * GetPhy()->GetSlot();
        cw = std::min(cwMax, (cw + 1) * 2);
    }

    return tt;
}

void
MinstrelWifiManagerLogged::InitSampleTable(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    station->m_col = station->m_index = 0;
    uint8_t numSampleRates = station->m_nModes;

    for (uint8_t col = 0; col < m_sampleCol; col++)
    {
        for (uint8_t i = 0; i < numSampleRates; i++)
        {
            int uv = m_uniformRandomVariable->GetInteger(0, numSampleRates);
            uint16_t newIndex = (i + uv) % numSampleRates;
            while (station->m_sampleTable[newIndex][col] != 0)
            {
                newIndex = (newIndex + 1) % station->m_nModes;
            }
            station->m_sampleTable[newIndex][col] = i;
        }
    }
}

void
MinstrelWifiManagerLogged::PrintSampleTable(MinstrelWifiRemoteStationLogged* station) const
{
    uint8_t numSampleRates = station->m_nModes;
    std::stringstream table;
    for (uint8_t i = 0; i < numSampleRates; i++)
    {
        for (uint8_t j = 0; j < m_sampleCol; j++)
        {
            table << station->m_sampleTable[i][j] << "\t";
        }
        table << std::endl;
    }
    NS_LOG_DEBUG(table.str());
}

void
MinstrelWifiManagerLogged::PrintTable(MinstrelWifiRemoteStationLogged* station)
{
    if (!station->m_statsFile.is_open())
    {
        std::ostringstream tmp;
        tmp << "minstrel-stats-" << station->m_state->m_address << ".txt";
        station->m_statsFile.open(tmp.str(), std::ios::out);
    }

    station->m_statsFile
        << "best   _______________rate________________    ________statistics________    "
           "________last_______    ______sum-of________\n"
        << "rate  [      name       idx airtime max_tp]  [avg(tp) avg(prob) sd(prob)]  "
           "[prob.|retry|suc|att]  [#success | #attempts]\n";

    uint16_t maxTpRate = station->m_maxTpRate;
    uint16_t maxTpRate2 = station->m_maxTpRate2;
    uint16_t maxProbRate = station->m_maxProbRate;

    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        RateInfoLogged rate = station->m_minstrelTable[i];

        if (i == maxTpRate)
            station->m_statsFile << 'A';
        else
            station->m_statsFile << ' ';
        if (i == maxTpRate2)
            station->m_statsFile << 'B';
        else
            station->m_statsFile << ' ';
        if (i == maxProbRate)
            station->m_statsFile << 'P';
        else
            station->m_statsFile << ' ';

        float tmpTh = rate.throughput / 100000.0F;
        station->m_statsFile << "   " << std::setw(17) << GetSupported(station, i) << "  "
                             << std::setw(2) << i << "  " << std::setw(4)
                             << rate.perfectTxTime.GetMicroSeconds() << std::setw(8)
                             << "    -----    " << std::setw(8) << tmpTh << "    " << std::setw(3)
                             << rate.ewmaProb / 180 << std::setw(3) << "       ---      "
                             << std::setw(3) << rate.prob / 180 << "     " << std::setw(1)
                             << rate.adjustedRetryCount << "   " << std::setw(3)
                             << rate.prevNumRateSuccess << " " << std::setw(3)
                             << rate.prevNumRateAttempt << "   " << std::setw(9) << rate.successHist
                             << "   " << std::setw(9) << rate.attemptHist << "\n";
    }
    station->m_statsFile << "\nTotal packet count:    ideal "
                         << station->m_totalPacketsCount - station->m_samplePacketsCount
                         << "      lookaround " << station->m_samplePacketsCount << "\n\n";

    station->m_statsFile.flush();
}

/* ----------------------
   PHASE1: Logging helpers
   ---------------------- */

double
MinstrelWifiManagerLogged::GetStratifiedLogProbability(uint8_t rate, double snr, bool success)
{
    const double base[8] = {1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.08};
    const uint8_t idx = std::min<uint8_t>(rate, 7);
    double p = base[idx];
    if (!success)
        p *= 2.0;
    if (snr < 15.0)
        p *= 1.5;
    if (idx <= 1)
        p = 1.0;
    if (idx >= 6 && snr > 25.0 && success)
        p *= 0.5;
    return std::min(1.0, p);
}

double
MinstrelWifiManagerLogged::GetRandomValue()
{
    return m_uniformDist(m_rng);
}

uint32_t
MinstrelWifiManagerLogged::CountRecentRateChanges(MinstrelWifiRemoteStationLogged* st,
                                                  uint32_t window)
{
    uint32_t changes = 0;
    if (st->m_rateHistory.size() > 1)
    {
        const size_t start =
            (st->m_rateHistory.size() > window) ? st->m_rateHistory.size() - window : 1;
        for (size_t i = start; i < st->m_rateHistory.size(); ++i)
        {
            if (st->m_rateHistory[i] != st->m_rateHistory[i - 1])
                changes++;
        }
    }
    return changes;
}

double
MinstrelWifiManagerLogged::CalculateRateStability(MinstrelWifiRemoteStationLogged* st)
{
    const double denom = 20.0;
    const double changes = static_cast<double>(CountRecentRateChanges(st, 20));
    return std::clamp(1.0 - (changes / denom), 0.0, 1.0);
}

double
MinstrelWifiManagerLogged::CalculateRecentThroughput(MinstrelWifiRemoteStationLogged* st,
                                                     uint32_t window)
{
    if (st->m_throughputHistory.empty())
        return 0.0;
    const size_t start =
        (st->m_throughputHistory.size() > window) ? st->m_throughputHistory.size() - window : 0;
    double sum = 0.0;
    uint32_t n = 0;
    for (size_t i = start; i < st->m_throughputHistory.size(); ++i)
    {
        sum += st->m_throughputHistory[i];
        n++;
    }
    return (n > 0) ? (sum / n) : 0.0;
}

double
MinstrelWifiManagerLogged::CalculateRecentPacketLoss(MinstrelWifiRemoteStationLogged* st,
                                                     uint32_t window)
{
    if (st->m_packetSuccessHistory.empty())
        return 0.0;
    const size_t start = (st->m_packetSuccessHistory.size() > window)
                             ? st->m_packetSuccessHistory.size() - window
                             : 0;
    uint32_t total = 0, fails = 0;
    for (size_t i = start; i < st->m_packetSuccessHistory.size(); ++i)
    {
        total++;
        if (!st->m_packetSuccessHistory[i])
            fails++;
    }
    return (total > 0) ? static_cast<double>(fails) / static_cast<double>(total) : 0.0;
}

double
MinstrelWifiManagerLogged::CalculateRetrySuccessRatio(MinstrelWifiRemoteStationLogged* st)
{
    uint32_t succ = 0;
    uint32_t totalRetries = 0;
    const size_t n = std::min(st->m_packetSuccessHistory.size(), st->m_retryHistory.size());
    for (size_t i = 0; i < n; ++i)
    {
        if (st->m_packetSuccessHistory[i])
            succ++;
        totalRetries += st->m_retryHistory[i];
    }
    return (succ > 0) ? static_cast<double>(succ) / static_cast<double>(totalRetries + 1) : 0.0;
}

double
MinstrelWifiManagerLogged::CalculateOptimalRateDistance(MinstrelWifiRemoteStationLogged* st)
{
    const uint8_t opt = TierFromSnr(st, st->m_lastSnr);
    const int d = static_cast<int>(st->m_txrate) - static_cast<int>(opt);
    return std::min(1.0, std::abs(d) / 7.0);
}

double
MinstrelWifiManagerLogged::CalculateAggressiveFactor(MinstrelWifiRemoteStationLogged* st)
{
    if (st->m_rateHistory.empty())
        return 0.0;
    uint32_t cnt = 0;
    for (uint8_t r : st->m_rateHistory)
        if (r >= 6)
            cnt++;
    return static_cast<double>(cnt) / static_cast<double>(st->m_rateHistory.size());
}

double
MinstrelWifiManagerLogged::CalculateConservativeFactor(MinstrelWifiRemoteStationLogged* st)
{
    if (st->m_rateHistory.empty())
        return 0.0;
    uint32_t cnt = 0;
    for (uint8_t r : st->m_rateHistory)
        if (r <= 2)
            cnt++;
    return static_cast<double>(cnt) / static_cast<double>(st->m_rateHistory.size());
}

uint8_t
MinstrelWifiManagerLogged::GetRecommendedSafeRate(MinstrelWifiRemoteStationLogged* st)
{
    return TierFromSnr(st, st->m_lastSnr);
}

double
MinstrelWifiManagerLogged::CalculateSnrStability(MinstrelWifiRemoteStationLogged* st)
{
    const size_t window = std::min<size_t>(10, st->m_snrSamples.size());
    if (window < 2)
        return 0.0;
    const size_t start = st->m_snrSamples.size() - window;
    double mean = 0.0;
    for (size_t i = start; i < st->m_snrSamples.size(); ++i)
        mean += st->m_snrSamples[i];
    mean /= window;
    double var = 0.0;
    for (size_t i = start; i < st->m_snrSamples.size(); ++i)
    {
        const double d = st->m_snrSamples[i] - mean;
        var += d * d;
    }
    return std::sqrt(var / window);
}

double
MinstrelWifiManagerLogged::CalculateSnrPredictionConfidence(MinstrelWifiRemoteStationLogged* st)
{
    const double stability = CalculateSnrStability(st);
    return 1.0 / (1.0 + stability);
}

uint8_t
MinstrelWifiManagerLogged::TierFromSnr(MinstrelWifiRemoteStationLogged* st, double snr) const
{
    (void)st;
    if (!std::isfinite(snr))
        return 0;
    return (snr > 25)   ? 7
           : (snr > 21) ? 6
           : (snr > 18) ? 5
           : (snr > 15) ? 4
           : (snr > 12) ? 3
           : (snr > 9)  ? 2
           : (snr > 6)  ? 1
                        : 0;
}

void
MinstrelWifiManagerLogged::LogDecision(MinstrelWifiRemoteStationLogged* st,
                                       int decisionReason,
                                       bool packetSuccess)
{
    if (!m_logFile.is_open())
        return;

    const uint8_t validatedRateIdx = static_cast<uint8_t>(std::min<int>(st->m_txrate, 255));
    const double lastSnr = std::isfinite(st->m_lastSnr) ? st->m_lastSnr : -99.0;

    const double logProb = GetStratifiedLogProbability(validatedRateIdx, lastSnr, packetSuccess);
    if (GetRandomValue() > logProb)
        return;

    // station id
    std::ostringstream stationId;
    if (st->m_node)
        stationId << st->m_node->GetId();
    else
        stationId << reinterpret_cast<uintptr_t>(st);

    // map rate idx to approximate phy rate (placeholder)
    const uint64_t phyRate = 1000000ull + static_cast<uint64_t>(validatedRateIdx) * 1000000ull;

    // compute features
    const double recentThroughputTrend = CalculateRecentThroughput(st, 10);
    const double packetLossRate = CalculateRecentPacketLoss(st, 20);
    const double retrySuccessRatio = CalculateRetrySuccessRatio(st);
    const uint32_t recentRateChanges = CountRecentRateChanges(st, 20);
    const uint32_t timeSinceLastRateChange = st->m_sinceLastRateChange;
    const double rateStabilityScore = CalculateRateStability(st);
    const double optimalRateDistance = CalculateOptimalRateDistance(st);
    const double aggressiveFactor = CalculateAggressiveFactor(st);
    const double conservativeFactor = CalculateConservativeFactor(st);
    const uint8_t recommendedSafeRate = GetRecommendedSafeRate(st);

    // SNR variance
    double snrVariance = 0.0;
    if (st->m_snrSamples.size() >= 2)
    {
        const size_t window = std::min<size_t>(st->m_snrSamples.size(), 20);
        const size_t start = st->m_snrSamples.size() - window;
        double mean = 0.0;
        for (size_t i = start; i < st->m_snrSamples.size(); ++i)
            mean += st->m_snrSamples[i];
        mean /= window;
        double var = 0.0;
        for (size_t i = start; i < st->m_snrSamples.size(); ++i)
        {
            const double d = st->m_snrSamples[i] - mean;
            var += d * d;
        }
        snrVariance = var / window;
        if (!std::isfinite(snrVariance))
            snrVariance = 0.0;
    }

    const double offeredLoad = 0.0;
    const int queueLen = 0;
    const int retryCount = st->m_retry;
    const uint16_t channelWidth = 20;

    const double simTime = Simulator::Now().GetSeconds();
    const uint32_t consecSuccess = st->m_consecSuccess;
    const uint32_t consecFailure = st->m_consecFailure;

    m_logFile << std::fixed << std::setprecision(6) << simTime << "," << stationId.str() << ","
              << static_cast<int>(validatedRateIdx) << "," << phyRate << "," << lastSnr << ","
              << consecSuccess << "," << consecFailure << "," << recentThroughputTrend << ","
              << packetLossRate << "," << retrySuccessRatio << "," << recentRateChanges << ","
              << timeSinceLastRateChange << "," << rateStabilityScore << "," << optimalRateDistance
              << "," << aggressiveFactor << "," << conservativeFactor << ","
              << static_cast<int>(recommendedSafeRate) << "," << decisionReason << ","
              << (packetSuccess ? 1 : 0) << "," << (st->m_isSampling ? 1 : 0) << "," << offeredLoad
              << "," << queueLen << "," << retryCount << "," << channelWidth << "," << snrVariance
              << "\n";
    m_logFile.flush();
}

} // namespace ns3