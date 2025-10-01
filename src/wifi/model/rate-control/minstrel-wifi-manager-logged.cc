/*
 * Copyright (c) 2009 Duy Nguyen
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Duy Nguyen <duy@soe.ucsc.edu>
 *          Matías Richart <mrichart@fing.edu.uy>
 *          Ahmed (ahmedjk34) - Fixed Version 2025-10-01
 *
 * FIXED VERSION - ALL TEMPORAL LEAKAGE REMOVED
 *
 * CRITICAL FIXES APPLIED:
 * - Issue #1: Removed 7 temporal leakage features from logging
 * - Issue #33: Success ratios now from PREVIOUS window (not current packet)
 * - Issue #4: Added scenario_file column for proper train/test splitting
 * - Issue #14: Reproducible random seed for stratified sampling
 *
 * FEATURES REMOVED (Temporal Leakage):
 * ❌ consecSuccess, consecFailure - outcomes of CURRENT rate
 * ❌ retrySuccessRatio - derived from outcomes
 * ❌ timeSinceLastRateChange, rateStabilityScore, recentRateChanges - rate history
 * ❌ packetSuccess - literal packet outcome
 *
 * FEATURES KEPT (Safe - Pre-Decision):
 * ✅ SNR features (lastSnr, snrFast, snrSlow, trends, variance)
 * ✅ shortSuccRatio, medSuccRatio - from PREVIOUS window (Issue #33)
 * ✅ packetLossRate - from PREVIOUS window (Issue #33)
 * ✅ Network state (channelWidth, mobilityMetric)
 * ✅ Assessment (severity, confidence - from previous window)
 */

#include "minstrel-wifi-manager-logged.h"

#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-phy.h"

#include <iomanip>
#include <sstream>

#define Min(a, b) ((a < b) ? a : b)

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
            .AddAttribute("EWMA",
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
                          "Path to the log file for detailed packet logging",
                          StringValue(""),
                          MakeStringAccessor(&MinstrelWifiManagerLogged::m_logFilePath),
                          MakeStringChecker())
            .AddAttribute("ScenarioFileName",
                          "Scenario identifier for train/test splitting (Issue #4)",
                          StringValue("scenario_unknown"),
                          MakeStringAccessor(&MinstrelWifiManagerLogged::m_scenarioFileName),
                          MakeStringChecker())
            .AddTraceSource("Rate",
                            "Traced value for rate changes (b/s)",
                            MakeTraceSourceAccessor(&MinstrelWifiManagerLogged::m_rateChange),
                            "ns3::TracedValueCallback::Uint64");
    return tid;
}

MinstrelWifiManagerLogged::MinstrelWifiManagerLogged()
    : WifiRemoteStationManager(),
      m_logHeaderWritten(false),
      m_rng(42),
      m_uniformDist(0.0, 1.0),
      m_scenarioDistance(20.0), // ← ADD THIS
      m_scenarioInterferers(0)  // ← ADD THIS
{
    NS_LOG_FUNCTION(this);
    m_uniformRandomVariable = CreateObject<UniformRandomVariable>();
    m_uniformRandomVariable->SetStream(42);
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

    // Initialize logging if path is set
    if (!m_logFilePath.empty() && !m_logHeaderWritten)
    {
        m_logFile.open(m_logFilePath, std::ios::out | std::ios::trunc);
        if (m_logFile.is_open())
        {
            // FIXED: CSV header with ONLY safe features (Issue #1, #33)
            m_logFile << "time,stationId,rateIdx,phyRate,"
                      // SNR features (SAFE - pre-decision measurements)
                      << "lastSnr,snrFast,snrSlow,snrTrendShort,"
                      << "snrStabilityIndex,snrPredictionConfidence,snrVariance,"
                      // SUCCESS RATIOS FROM PREVIOUS WINDOW (FIXED: Issue #33)
                      << "shortSuccRatio,medSuccRatio,"
                      // PACKET LOSS FROM PREVIOUS WINDOW (FIXED: Issue #33)
                      << "packetLossRate,"
                      // Network state (SAFE - environmental)
                      << "channelWidth,mobilityMetric,"
                      // Assessment features (SAFE - from previous window)
                      << "severity,confidence,"
                      // SCENARIO FILE (FIXED: Issue #4 - for train/test splitting)
                      << "scenario_file\n";

            m_logFile.flush();
            m_logHeaderWritten = true;

            NS_LOG_INFO("FIXED: Logging initialized with SAFE features only");
            NS_LOG_INFO("  ✅ SNR features (7)");
            NS_LOG_INFO("  ✅ Previous window success ratios (2)");
            NS_LOG_INFO("  ✅ Previous window packet loss (1)");
            NS_LOG_INFO("  ✅ Network state (2)");
            NS_LOG_INFO("  ✅ Assessment features (2)");
            NS_LOG_INFO("  ✅ Scenario file for splitting (1)");
            NS_LOG_INFO("  ❌ REMOVED: 7 temporal leakage features");
        }
    }

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
}

int64_t
MinstrelWifiManagerLogged::AssignStreams(int64_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_uniformRandomVariable->SetStream(stream);
    m_rng.seed(stream); // FIXED: Issue #14 - Seed RNG for reproducibility
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
    station->m_isSampling = false;
    station->m_sampleRate = 0;
    station->m_sampleDeferred = false;
    station->m_shortRetry = 0;
    station->m_longRetry = 0;
    station->m_retry = 0;
    station->m_txrate = 0;
    station->m_initialized = false;

    // FIXED: Initialize safe feature tracking (Issue #1, #33)
    station->m_lastSnr = 0.0;
    station->m_fastEwmaSnr = 0.0;
    station->m_slowEwmaSnr = 0.0;
    station->m_previousWindowSuccess = 0;
    station->m_previousWindowTotal = 0;
    station->m_currentWindowPackets = 0;
    station->m_previousWindowLosses = 0;

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

// FIXED: Issue #33 - Update window state (move current to previous)
void
MinstrelWifiManagerLogged::UpdateWindowState(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);

    if (station->m_currentWindowPackets >= WINDOW_SIZE)
    {
        // FIXED: Use m_previousShortWindow (not m_previousSuccessHistory)
        station->m_previousShortWindow = station->m_currentShortWindow;
        station->m_previousMedWindow = station->m_currentMedWindow;

        // Calculate stats
        station->m_previousWindowSuccess = 0;
        station->m_previousWindowLosses = 0;

        for (bool success : station->m_currentShortWindow) // ✓ CORRECT
        {
            if (success)
                station->m_previousWindowSuccess++;
            else
                station->m_previousWindowLosses++;
        }

        station->m_previousWindowTotal = station->m_currentShortWindow.size();

        // Reset current
        station->m_currentShortWindow.clear();
        station->m_currentMedWindow.clear();
        station->m_currentWindowPackets = 0;
    }
}

// Minstrel core algorithm unchanged - same as original
void
MinstrelWifiManagerLogged::UpdateRate(MinstrelWifiRemoteStationLogged* station)
{
    NS_LOG_FUNCTION(this << station);
    station->m_longRetry++;
    station->m_minstrelTable[station->m_txrate].numRateAttempt++;

    NS_LOG_DEBUG("DoReportDataFailed " << station << " rate " << station->m_txrate << " longRetry "
                                       << station->m_longRetry);

    // Minstrel retry chain logic (unchanged from original)
    if (!station->m_isSampling)
    {
        NS_LOG_DEBUG("Failed with normal rate: current="
                     << station->m_txrate << ", sample=" << station->m_sampleRate
                     << ", maxTp=" << station->m_maxTpRate << ", maxTp2=" << station->m_maxTpRate2
                     << ", maxProb=" << station->m_maxProbRate);

        if (station->m_longRetry <
            station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount)
        {
            NS_LOG_DEBUG(" More retries left for the maximum throughput rate.");
            station->m_txrate = station->m_maxTpRate;
        }
        else if (station->m_longRetry <=
                 (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount))
        {
            NS_LOG_DEBUG(" More retries left for the second maximum throughput rate.");
            station->m_txrate = station->m_maxTpRate2;
        }
        else if (station->m_longRetry <=
                 (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
        {
            NS_LOG_DEBUG(" More retries left for the maximum probability rate.");
            station->m_txrate = station->m_maxProbRate;
        }
        else if (station->m_longRetry >
                 (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxTpRate2].adjustedRetryCount +
                  station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
        {
            NS_LOG_DEBUG(" More retries left for the base rate.");
            station->m_txrate = 0;
        }
    }
    else
    {
        NS_LOG_DEBUG("Failed with look around rate: current="
                     << station->m_txrate << ", sample=" << station->m_sampleRate
                     << ", maxTp=" << station->m_maxTpRate << ", maxTp2=" << station->m_maxTpRate2
                     << ", maxProb=" << station->m_maxProbRate);

        if (station->m_sampleDeferred)
        {
            NS_LOG_DEBUG("Look around rate is slower than the maximum throughput rate.");
            if (station->m_longRetry <
                station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount)
            {
                NS_LOG_DEBUG(" More retries left for the maximum throughput rate.");
                station->m_txrate = station->m_maxTpRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount))
            {
                NS_LOG_DEBUG(" More retries left for the sampling rate.");
                station->m_txrate = station->m_sampleRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                NS_LOG_DEBUG(" More retries left for the maximum probability rate.");
                station->m_txrate = station->m_maxProbRate;
            }
            else if (station->m_longRetry >
                     (station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                NS_LOG_DEBUG(" More retries left for the base rate.");
                station->m_txrate = 0;
            }
        }
        else
        {
            NS_LOG_DEBUG("Look around rate is faster than the maximum throughput rate.");
            if (station->m_longRetry <
                station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount)
            {
                NS_LOG_DEBUG(" More retries left for the sampling rate.");
                station->m_txrate = station->m_sampleRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount))
            {
                NS_LOG_DEBUG(" More retries left for the maximum throughput rate.");
                station->m_txrate = station->m_maxTpRate;
            }
            else if (station->m_longRetry <=
                     (station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                NS_LOG_DEBUG(" More retries left for the maximum probability rate.");
                station->m_txrate = station->m_maxProbRate;
            }
            else if (station->m_longRetry >
                     (station->m_minstrelTable[station->m_sampleRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxTpRate].adjustedRetryCount +
                      station->m_minstrelTable[station->m_maxProbRate].adjustedRetryCount))
            {
                NS_LOG_DEBUG(" More retries left for the base rate.");
                station->m_txrate = 0;
            }
        }
    }
}

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
    NS_LOG_DEBUG("DoGetRtsMode m_txrate=" << station->m_txrate);
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
    NS_LOG_DEBUG("Total: " << station->m_totalPacketsCount
                           << "  Sample: " << station->m_samplePacketsCount
                           << "  Deferred: " << station->m_numSamplesDeferred);

    int delta = (station->m_totalPacketsCount * m_lookAroundRate / 100) -
                (station->m_samplePacketsCount + station->m_numSamplesDeferred / 2);

    NS_LOG_DEBUG("Decide sampling. Delta: " << delta << " lookAroundRatio: " << m_lookAroundRate);

    if (delta >= 0)
    {
        NS_LOG_DEBUG("Search next sampling rate");
        uint8_t ratesSupported = station->m_nModes;
        if (delta > ratesSupported * 2)
        {
            station->m_samplePacketsCount += (delta - ratesSupported * 2);
        }

        idx = GetNextSample(station);
        NS_LOG_DEBUG("Sample rate = " << idx << "(" << GetSupported(station, idx) << ")");

        if (idx >= station->m_nModes)
        {
            NS_LOG_DEBUG("ALERT!!! ERROR");
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
            NS_LOG_DEBUG("The next look around rate is slower than the maximum throughput rate, "
                         "continue with the maximum throughput rate: "
                         << station->m_maxTpRate << "("
                         << GetSupported(station, station->m_maxTpRate) << ")");
            idx = station->m_maxTpRate;
        }
    }
    else
    {
        NS_LOG_DEBUG("Continue using the maximum throughput rate: "
                     << station->m_maxTpRate << "(" << GetSupported(station, station->m_maxTpRate)
                     << ")");
        idx = station->m_maxTpRate;
    }

    NS_LOG_DEBUG("Rate = " << idx << "(" << GetSupported(station, idx) << ")");

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

    NS_LOG_FUNCTION(this);
    station->m_nextStatsUpdate = Simulator::Now() + m_updateStats;
    NS_LOG_DEBUG("Next update at " << station->m_nextStatsUpdate);
    NS_LOG_DEBUG("Currently using rate: " << station->m_txrate << " ("
                                          << GetSupported(station, station->m_txrate) << ")");

    Time txTime;
    uint32_t tempProb;

    NS_LOG_DEBUG("Index-Rate\t\tAttempt\tSuccess");
    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        txTime = station->m_minstrelTable[i].perfectTxTime;

        if (txTime.GetMicroSeconds() == 0)
        {
            txTime = Seconds(1);
        }

        NS_LOG_DEBUG(+i << " " << GetSupported(station, i) << "\t"
                        << station->m_minstrelTable[i].numRateAttempt << "\t"
                        << station->m_minstrelTable[i].numRateSuccess);

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

    NS_LOG_DEBUG("Attempt/success reset to 0");

    uint32_t max_tp = 0;
    uint8_t index_max_tp = 0;
    uint8_t index_max_tp2 = 0;

    NS_LOG_DEBUG(
        "Finding the maximum throughput, second maximum throughput, and highest probability");
    NS_LOG_DEBUG("Index-Rate\t\tT-put\tEWMA");

    for (uint8_t i = 0; i < station->m_nModes; i++)
    {
        NS_LOG_DEBUG(+i << " " << GetSupported(station, i) << "\t"
                        << station->m_minstrelTable[i].throughput << "\t"
                        << station->m_minstrelTable[i].ewmaProb);

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

    NS_LOG_DEBUG("max throughput=" << +index_max_tp << "(" << GetSupported(station, index_max_tp)
                                   << ")\tsecond max throughput=" << +index_max_tp2 << "("
                                   << GetSupported(station, index_max_tp2)
                                   << ")\tmax prob=" << +index_max_prob << "("
                                   << GetSupported(station, index_max_prob) << ")");

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
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);

    // FIXED: Convert ns-3 SNR to realistic dB using scenario parameters
    double realisticSnr = rxSnr;

    // ns-3 often gives SNR as linear power ratio (huge values)
    // Convert to dB and apply realistic path loss model
    if (rxSnr > 100.0)
    {
        // Convert from linear to dB
        realisticSnr = 10.0 * std::log10(rxSnr);
    }

    // Apply realistic path loss based on distance
    // Use a simple model similar to your ConvertNS3ToRealisticSnr
    if (m_scenarioDistance <= 20.0)
    {
        realisticSnr = std::min(realisticSnr, 35.0 - (m_scenarioDistance * 0.8));
    }
    else if (m_scenarioDistance <= 50.0)
    {
        realisticSnr = std::min(realisticSnr, 19.0 - ((m_scenarioDistance - 20.0) * 0.5));
    }
    else
    {
        realisticSnr = std::min(realisticSnr, 4.0 - ((m_scenarioDistance - 50.0) * 0.3));
    }

    // Reduce SNR based on interferers
    realisticSnr -= (m_scenarioInterferers * 2.0);

    // Clamp to realistic WiFi range
    realisticSnr = std::max(-30.0, std::min(50.0, realisticSnr));

    // FIXED: Update SNR with EWMA (using realistic SNR)
    const double kAlphaFast = 0.30;
    const double kAlphaSlow = 0.05;

    if (std::isfinite(realisticSnr))
    {
        if (station->m_fastEwmaSnr == 0.0 && station->m_slowEwmaSnr == 0.0)
        {
            station->m_fastEwmaSnr = realisticSnr;
            station->m_slowEwmaSnr = realisticSnr;
        }
        else
        {
            station->m_fastEwmaSnr =
                kAlphaFast * realisticSnr + (1.0 - kAlphaFast) * station->m_fastEwmaSnr;
            station->m_slowEwmaSnr =
                kAlphaSlow * realisticSnr + (1.0 - kAlphaSlow) * station->m_slowEwmaSnr;
        }
        station->m_lastSnr = realisticSnr;
        station->m_snrHistory.push_back(realisticSnr);
        if (station->m_snrHistory.size() > 200)
        {
            station->m_snrHistory.pop_front();
        }
    }

    NS_LOG_DEBUG("DoReportRxOk: Raw SNR=" << rxSnr << " -> Realistic SNR=" << realisticSnr
                                          << " dB (dist=" << m_scenarioDistance
                                          << "m, intf=" << m_scenarioInterferers << ")");
}

void
MinstrelWifiManagerLogged::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);
    NS_LOG_DEBUG("DoReportRtsFailed m_txrate=" << station->m_txrate);
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
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);
    UpdateRetry(station);
}

void
MinstrelWifiManagerLogged::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);

    NS_LOG_DEBUG("DoReportDataFailed " << station << "\t rate " << station->m_txrate
                                       << "\tlongRetry \t" << station->m_longRetry);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return;
    }

    // FIXED: Track in current window
    station->m_currentShortWindow.push_back(false);
    station->m_currentMedWindow.push_back(false);
    station->m_currentWindowPackets++;
    UpdateWindowState(station);

    UpdateRate(station);

    // FIXED: Correct call
    LogSafeFeatures(station, station->m_txrate, false); // ✅ CORRECT
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
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return;
    }

    NS_LOG_DEBUG("DoReportDataOk m_txrate = "
                 << station->m_txrate
                 << ", attempt = " << station->m_minstrelTable[station->m_txrate].numRateAttempt
                 << ", success = " << station->m_minstrelTable[station->m_txrate].numRateSuccess
                 << " (before update).");

    station->m_minstrelTable[station->m_txrate].numRateSuccess++;
    station->m_minstrelTable[station->m_txrate].numRateAttempt++;

    // FIXED: Issue #33 - Track in current window (not logged until it becomes previous)
    station->m_currentShortWindow.push_back(true);
    station->m_currentMedWindow.push_back(true);
    station->m_currentWindowPackets++;
    UpdateWindowState(station);

    UpdatePacketCounters(station);
    UpdateRetry(station);
    UpdateStats(station);

    // FIXED: Correct call
    LogSafeFeatures(station, station->m_txrate, true); // ✅ CORRECT

    if (station->m_nModes >= 1)
    {
        station->m_txrate = FindRate(station);
    }
}

void
MinstrelWifiManagerLogged::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);

    CheckInit(station);
    if (!station->m_initialized)
    {
        return;
    }

    NS_LOG_DEBUG("DoReportFinalDataFailed m_txrate = "
                 << station->m_txrate
                 << ", attempt = " << station->m_minstrelTable[station->m_txrate].numRateAttempt
                 << ", success = " << station->m_minstrelTable[station->m_txrate].numRateSuccess
                 << " (before update).");

    // FIXED: Issue #33 - Track in current window
    station->m_currentShortWindow.push_back(false);
    station->m_currentMedWindow.push_back(false);
    station->m_currentWindowPackets++;
    UpdateWindowState(station);

    UpdatePacketCounters(station);
    UpdateRetry(station);
    UpdateStats(station);

    // FIXED: Correct call
    LogSafeFeatures(station, station->m_txrate, false); // ✅ CORRECT

    if (station->m_nModes >= 1)
    {
        station->m_txrate = FindRate(station);
    }
    NS_LOG_DEBUG("Next rate to use TxRate = " << station->m_txrate);
}

// ============================================================================
// FIXED: Safe Feature Calculation Methods (Issue #1, #33)
// These calculate features from PREVIOUS window data only
// ============================================================================

double
MinstrelWifiManagerLogged::CalculatePreviousMedSuccessRatio(
    MinstrelWifiRemoteStationLogged* st) const
{
    if (st->m_previousWindowTotal == 0)
        return 1.0;

    uint32_t window = std::min<uint32_t>(25, st->m_previousMedWindow.size()); // ✓ CORRECT
    if (window == 0)
        return 1.0;

    uint32_t start = st->m_previousMedWindow.size() - window;
    uint32_t successes = 0;
    for (uint32_t i = start; i < st->m_previousMedWindow.size(); ++i)
    {
        if (st->m_previousMedWindow[i]) // ✓ CORRECT
            successes++;
    }

    return static_cast<double>(successes) / window;
}

double
MinstrelWifiManagerLogged::CalculatePreviousPacketLoss(MinstrelWifiRemoteStationLogged* st) const
{
    if (st->m_previousWindowTotal == 0)
        return 0.0;

    return static_cast<double>(st->m_previousWindowLosses) / st->m_previousWindowTotal;
}

double
MinstrelWifiManagerLogged::CalculateSnrStability(MinstrelWifiRemoteStationLogged* st) const
{
    if (st->m_snrHistory.size() < 2)
        return 0.0;

    uint32_t window = std::min<uint32_t>(10, st->m_snrHistory.size());
    uint32_t start = st->m_snrHistory.size() - window;

    double mean = 0.0;
    for (uint32_t i = start; i < st->m_snrHistory.size(); ++i)
    {
        mean += st->m_snrHistory[i];
    }
    mean /= window;

    double variance = 0.0;
    for (uint32_t i = start; i < st->m_snrHistory.size(); ++i)
    {
        double d = st->m_snrHistory[i] - mean;
        variance += d * d;
    }

    return std::sqrt(variance / window);
}

double
MinstrelWifiManagerLogged::CalculateSnrPredictionConfidence(
    MinstrelWifiRemoteStationLogged* st) const
{
    double stability = CalculateSnrStability(st);
    return 1.0 / (1.0 + stability);
}

double
MinstrelWifiManagerLogged::CalculateSnrVariance(MinstrelWifiRemoteStationLogged* st) const
{
    if (st->m_snrHistory.size() < 2)
        return 0.0;

    uint32_t window = std::min<uint32_t>(20, st->m_snrHistory.size());
    uint32_t start = st->m_snrHistory.size() - window;

    double mean = 0.0;
    for (uint32_t i = start; i < st->m_snrHistory.size(); ++i)
    {
        mean += st->m_snrHistory[i];
    }
    mean /= window;

    double variance = 0.0;
    for (uint32_t i = start; i < st->m_snrHistory.size(); ++i)
    {
        double d = st->m_snrHistory[i] - mean;
        variance += d * d;
    }

    return variance / window;
}

double
MinstrelWifiManagerLogged::CalculateSeverity(MinstrelWifiRemoteStationLogged* st) const
{
    double medSuccRatio = CalculatePreviousMedSuccessRatio(st);
    double failureRatio = 1.0 - medSuccRatio;
    double packetLoss = CalculatePreviousPacketLoss(st);

    return std::clamp(0.6 * failureRatio + 0.4 * packetLoss, 0.0, 1.0);
}

double
MinstrelWifiManagerLogged::CalculateConfidence(MinstrelWifiRemoteStationLogged* st) const
{
    double shortSuccRatio = CalculatePreviousShortSuccessRatio(st);
    double snrTrend = st->m_fastEwmaSnr - st->m_slowEwmaSnr;
    double trendPenalty = std::min(1.0, std::abs(snrTrend) / 3.0);

    return std::clamp(shortSuccRatio * (1.0 - 0.5 * trendPenalty), 0.0, 1.0);
}

double
MinstrelWifiManagerLogged::CalculateMobilityMetric(MinstrelWifiRemoteStationLogged* st) const
{
    double snrVariance = CalculateSnrVariance(st);
    return std::tanh(snrVariance / 10.0);
}

double
MinstrelWifiManagerLogged::CalculatePreviousShortSuccessRatio(
    MinstrelWifiRemoteStationLogged* st) const
{
    // FIXED: Use correct member variable names from header
    if (st->m_previousWindowTotal == 0)
        return 1.0;

    // Calculate from last 10 packets of PREVIOUS short window
    uint32_t window = std::min<uint32_t>(10, st->m_previousShortWindow.size());
    if (window == 0)
        return 1.0;

    uint32_t start = st->m_previousShortWindow.size() - window;
    uint32_t successes = 0;
    for (uint32_t i = start; i < st->m_previousShortWindow.size(); ++i)
    {
        if (st->m_previousShortWindow[i])
            successes++;
    }

    return static_cast<double>(successes) / window;
}

double
MinstrelWifiManagerLogged::GetStratifiedLogProbability(uint8_t rate, double snr, bool success) const
{
    const double base[8] = {1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.08};
    const uint32_t idx = std::min<uint32_t>(rate, 7);
    double p = base[idx];

    if (!success)
        p *= 2.0;

    if (snr < 15.0)
        p *= 1.5;

    if (idx <= 1)
        p = 1.0;

    return std::min(1.0, p);
}

double
MinstrelWifiManagerLogged::GetRandomValue()
{
    return m_uniformDist(m_rng);
}

// ============================================================================
// FIXED: Safe Logging Function (Issue #1, #33, #4)
// Logs ONLY pre-decision features
// ============================================================================

// void
// MinstrelWifiManagerLogged::LogSafeFeatures(MinstrelWifiRemoteStationLogged* st,
//                                            uint8_t currentRateIdx,
//                                            bool success)
// {
//     if (!m_logFile.is_open())
//     {
//         NS_LOG_WARN("Log file not open - cannot log features");
//         return;
//     }

//     // FIXED: More aggressive logging - log every 5th packet minimum
//     static uint32_t logCallCount = 0;
//     logCallCount++;

//     // Force logging for first 100 calls for testing
//     bool forceLog = (logCallCount <= 100);

//     // Stratified sampling - but much more lenient
//     double logProb = 1.0; // Start with 100%

//     if (!forceLog)
//     {
//         // Only apply stratified sampling after first 100 packets
//         logProb = GetStratifiedLogProbability(currentRateIdx, st->m_lastSnr, success);

//         // FIXED: Much more aggressive - log at least every 5th packet
//         logProb = std::max(0.2, logProb); // Minimum 20% logging rate
//     }

//     if (!forceLog && GetRandomValue() > logProb)
//     {
//         return;
//     }

//     // DEBUG: Print every 50th log to console
//     if (logCallCount % 50 == 0)
//     {
//         std::cout << "[LOG] Writing entry #" << logCallCount
//                   << " rate=" << static_cast<uint32_t>(currentRateIdx) << " success=" << success
//                   << " SNR=" << st->m_lastSnr << "dB" << std::endl;
//     }

//     // Calculate 14 SAFE features
//     double shortSuccRatio = CalculatePreviousShortSuccessRatio(st);
//     double medSuccRatio = CalculatePreviousMedSuccessRatio(st);
//     double packetLossRate = CalculatePreviousPacketLoss(st);

//     double snrTrendShort = st->m_fastEwmaSnr - st->m_slowEwmaSnr;
//     double snrStability = CalculateSnrStability(st);
//     double snrPredictionConfidence = CalculateSnrPredictionConfidence(st);
//     double snrVariance = CalculateSnrVariance(st);

//     uint32_t channelWidth = 20;
//     double mobilityMetric = CalculateMobilityMetric(st);

//     double severity = CalculateSeverity(st);
//     double confidence = CalculateConfidence(st);

//     // 802.11a rates (bps)
//     const uint64_t rates802_11a[8] =
//         {6000000, 9000000, 12000000, 18000000, 24000000, 36000000, 48000000, 54000000};

//     uint8_t rateIdx = std::min<uint8_t>(currentRateIdx, 7);
//     uint64_t phyRate = rates802_11a[rateIdx];

//     // Get station ID
//     uint32_t stationId = 0;
//     if (st->m_node)
//     {
//         stationId = st->m_node->GetId();
//     }
//     else
//     {
//         stationId = st->m_stationId;
//     }

//     double simTime = Simulator::Now().GetSeconds();

//     // FIXED: Ensure all values are finite
//     auto safeValue = [](double val, double defaultVal = 0.0) {
//         return std::isfinite(val) ? val : defaultVal;
//     };

//     // Write CSV row with ALL 14 features + metadata
//     m_logFile << std::fixed << std::setprecision(6) << simTime << "," << stationId << ","
//               << static_cast<uint32_t>(rateIdx) << "," << phyRate
//               << ","
//               // SNR features (7)
//               << safeValue(st->m_lastSnr) << "," << safeValue(st->m_fastEwmaSnr) << ","
//               << safeValue(st->m_slowEwmaSnr) << "," << safeValue(snrTrendShort) << ","
//               << safeValue(snrStability) << "," << safeValue(snrPredictionConfidence) << ","
//               << safeValue(snrVariance)
//               << ","
//               // Success ratios from PREVIOUS window (2)
//               << safeValue(shortSuccRatio, 0.5) << "," << safeValue(medSuccRatio, 0.5)
//               << ","
//               // Packet loss from PREVIOUS window (1)
//               << safeValue(packetLossRate)
//               << ","
//               // Network state (2)
//               << channelWidth << "," << safeValue(mobilityMetric)
//               << ","
//               // Assessment (2)
//               << safeValue(severity) << "," << safeValue(confidence, 0.5)
//               << ","
//               // Scenario file (1)
//               << m_scenarioFileName << "\n";

//     // CRITICAL: Flush after EVERY write to ensure data is saved
//     m_logFile.flush();

//     // Verify file is still open
//     if (!m_logFile.good())
//     {
//         NS_LOG_ERROR("Log file write failed! File may be corrupted.");
//     }
// }

void
MinstrelWifiManagerLogged::LogSafeFeatures(MinstrelWifiRemoteStationLogged* st,
                                           uint8_t currentRateIdx,
                                           bool success)
{
    if (!m_logFile.is_open())
    {
        return;
    }

    static uint32_t logCount = 0;
    logCount++;

    // Less aggressive stratified sampling (log ~30% of packets)
    double logProb = GetStratifiedLogProbability(currentRateIdx, st->m_lastSnr, success);
    logProb = std::max(0.3, logProb); // Minimum 30% logging

    if (GetRandomValue() > logProb)
    {
        return;
    }

    // DEBUG: Print progress every 500 logs
    if (logCount % 500 == 0)
    {
        std::cout << "[LOG] Logged " << logCount << " entries to " << m_scenarioFileName
                  << std::endl;
    }

    // Calculate 14 SAFE features
    double shortSuccRatio = CalculatePreviousShortSuccessRatio(st);
    double medSuccRatio = CalculatePreviousMedSuccessRatio(st);
    double packetLossRate = CalculatePreviousPacketLoss(st);

    double snrTrendShort = st->m_fastEwmaSnr - st->m_slowEwmaSnr;
    double snrStability = CalculateSnrStability(st);
    double snrPredictionConfidence = CalculateSnrPredictionConfidence(st);
    double snrVariance = CalculateSnrVariance(st);

    uint32_t channelWidth = 20;
    double mobilityMetric = CalculateMobilityMetric(st);

    double severity = CalculateSeverity(st);
    double confidence = CalculateConfidence(st);

    // 802.11a rates
    const uint64_t rates802_11a[8] =
        {6000000, 9000000, 12000000, 18000000, 24000000, 36000000, 48000000, 54000000};

    uint8_t rateIdx = std::min<uint8_t>(currentRateIdx, 7);
    uint64_t phyRate = rates802_11a[rateIdx];

    uint32_t stationId = st->m_node ? st->m_node->GetId() : st->m_stationId;
    double simTime = Simulator::Now().GetSeconds();

    // Safe value helper
    auto safeValue = [](double val, double defaultVal = 0.0) {
        return std::isfinite(val) ? val : defaultVal;
    };

    // Write CSV row
    m_logFile << std::fixed << std::setprecision(6) << simTime << "," << stationId << ","
              << static_cast<uint32_t>(rateIdx) << "," << phyRate
              << ","
              // SNR features (7)
              << safeValue(st->m_lastSnr) << "," << safeValue(st->m_fastEwmaSnr) << ","
              << safeValue(st->m_slowEwmaSnr) << "," << safeValue(snrTrendShort) << ","
              << safeValue(snrStability) << "," << safeValue(snrPredictionConfidence) << ","
              << safeValue(snrVariance)
              << ","
              // Success ratios (2)
              << safeValue(shortSuccRatio, 0.5) << "," << safeValue(medSuccRatio, 0.5)
              << ","
              // Packet loss (1)
              << safeValue(packetLossRate)
              << ","
              // Network state (2)
              << channelWidth << "," << safeValue(mobilityMetric)
              << ","
              // Assessment (2)
              << safeValue(severity) << "," << safeValue(confidence, 0.5)
              << ","
              // Scenario file (1)
              << m_scenarioFileName << "\n";

    // Flush every 100 entries to balance performance/safety
    if (logCount % 100 == 0)
    {
        m_logFile.flush();
    }
}

// ============================================================================
// Remaining Minstrel Core Functions (Unchanged)
// ============================================================================

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
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);
    return GetDataTxVector(station);
}

WifiTxVector
MinstrelWifiManagerLogged::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);
    return GetRtsTxVector(station);
}

bool
MinstrelWifiManagerLogged::DoNeedRetransmission(WifiRemoteStation* st,
                                                Ptr<const Packet> packet,
                                                bool normally)
{
    NS_LOG_FUNCTION(this << st << packet << normally);
    auto station = static_cast<MinstrelWifiRemoteStationLogged*>(st);

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
        NS_LOG_DEBUG("Initializing rate index " << +i << " " << GetSupported(station, i));
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
        NS_LOG_DEBUG(" perfectTxTime = " << station->m_minstrelTable[i].perfectTxTime);
        station->m_minstrelTable[i].retryCount = 1;
        station->m_minstrelTable[i].adjustedRetryCount = 1;

        NS_LOG_DEBUG(" Calculating the number of retries");
        Time totalTxTimeWithGivenRetries = Seconds(0.0);

        for (uint32_t retries = 2; retries < 11; retries++)
        {
            NS_LOG_DEBUG("  Checking " << retries << " retries");
            totalTxTimeWithGivenRetries =
                CalculateTimeUnicastPacket(station->m_minstrelTable[i].perfectTxTime, 0, retries);
            NS_LOG_DEBUG("   totalTxTimeWithGivenRetries = " << totalTxTimeWithGivenRetries);

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

    uint16_t newIndex;
    for (uint8_t col = 0; col < m_sampleCol; col++)
    {
        for (uint8_t i = 0; i < numSampleRates; i++)
        {
            int uv = m_uniformRandomVariable->GetInteger(0, numSampleRates);
            NS_LOG_DEBUG("InitSampleTable uv: " << uv);
            newIndex = (i + uv) % numSampleRates;

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
        {
            station->m_statsFile << 'A';
        }
        else
        {
            station->m_statsFile << ' ';
        }
        if (i == maxTpRate2)
        {
            station->m_statsFile << 'B';
        }
        else
        {
            station->m_statsFile << ' ';
        }
        if (i == maxProbRate)
        {
            station->m_statsFile << 'P';
        }
        else
        {
            station->m_statsFile << ' ';
        }

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

void
MinstrelWifiManagerLogged::SetScenarioParameters(double distance, uint32_t interferers)
{
    m_scenarioDistance = distance;
    m_scenarioInterferers = interferers;

    NS_LOG_INFO("MinstrelWifiManagerLogged: Set scenario parameters - "
                << "distance=" << distance << "m, interferers=" << interferers);
}

} // namespace ns3