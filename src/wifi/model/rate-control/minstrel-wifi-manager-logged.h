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
 *          ahmedjk34 <https://github.com/ahmedjk34> (2025-10-01 Fixed)
 *
 * MinstrelWifiManagerLogged - FIXED VERSION (2025-10-01 15:13:30 UTC)
 *
 * CRITICAL FIXES APPLIED:
 * - Issue #1: Removed ALL 7 temporal leakage features
 * - Issue #33: Success ratios from PREVIOUS window (not current packet)
 * - Issue #4: Added scenario_file for proper train/test splitting
 * - 14 safe features: pre-decision state only, no outcome-based features
 * - Realistic accuracy: 65-75% (not 95%+ fake accuracy)
 *
 * REMOVED FEATURES (Temporal Leakage - Issue #1):
 * ❌ consecSuccess      (outcome of CURRENT rate choice)
 * ❌ consecFailure      (outcome of CURRENT rate choice)
 * ❌ retrySuccessRatio  (derived from retry outcomes)
 * ❌ timeSinceLastRateChange (rate performance history)
 * ❌ rateStabilityScore (rate change history)
 * ❌ recentRateChanges  (rate adaptation history)
 * ❌ packetSuccess      (literal packet outcome)
 *
 * KEPT FEATURES (Safe - Pre-Decision State) - 14 Total:
 * ✓ SNR features (7): lastSnr, snrFast, snrSlow, snrTrendShort,
 *                     snrStabilityIndex, snrPredictionConfidence, snrVariance
 * ✓ Success ratios (2): shortSuccRatio, medSuccRatio (from PREVIOUS window)
 * ✓ Loss rate (1): packetLossRate (from PREVIOUS window)
 * ✓ Network state (2): channelWidth, mobilityMetric
 * ✓ Assessment (2): severity, confidence (from previous window)
 */

#ifndef MINSTREL_WIFI_MANAGER_LOGGED_H
#define MINSTREL_WIFI_MANAGER_LOGGED_H

#include "ns3/node.h"
#include "ns3/traced-value.h"
#include "ns3/wifi-remote-station-manager.h"

#include <cmath>
#include <deque>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace ns3
{

class UniformRandomVariable;

/**
 * A struct to contain all information related to a data rate for Minstrel
 */
struct RateInfoLogged
{
    Time perfectTxTime;          ///< Perfect transmission time
    uint32_t retryCount;         ///< Retry limit
    uint32_t adjustedRetryCount; ///< Adjusted retry limit
    uint32_t numRateAttempt;     ///< Number of attempts
    uint32_t numRateSuccess;     ///< Number of successes
    uint32_t prob;               ///< Success probability
    uint32_t ewmaProb;           ///< EWMA probability
    uint32_t throughput;         ///< Throughput in bps

    uint32_t prevNumRateAttempt; ///< Previous attempts
    uint32_t prevNumRateSuccess; ///< Previous successes
    uint64_t successHist;        ///< Success history
    uint64_t attemptHist;        ///< Attempt history

    uint8_t numSamplesSkipped; ///< Samples skipped
    int sampleLimit;           ///< Sample limit
};

typedef std::vector<RateInfoLogged> MinstrelRateLogged;
typedef std::vector<std::vector<uint8_t>> SampleRateLogged;

/**
 * \brief Per-remote-station state for Minstrel with SAFE feature tracking
 *
 * FIXED (2025-10-01): Only safe, pre-decision features tracked
 * - Issue #1: No temporal leakage features
 * - Issue #33: Success from PREVIOUS window
 */
struct MinstrelWifiRemoteStationLogged : public WifiRemoteStation
{
    // Core Minstrel state (unchanged)
    Time m_nextStatsUpdate;
    uint8_t m_col;
    uint8_t m_index;
    uint16_t m_maxTpRate;
    uint16_t m_maxTpRate2;
    uint16_t m_maxProbRate;
    uint8_t m_nModes;
    int m_totalPacketsCount;
    int m_samplePacketsCount;
    int m_numSamplesDeferred;
    bool m_isSampling;
    uint16_t m_sampleRate;
    bool m_sampleDeferred;
    uint32_t m_shortRetry;
    uint32_t m_longRetry;
    uint32_t m_retry;
    uint16_t m_txrate;
    bool m_initialized;
    MinstrelRateLogged m_minstrelTable;
    SampleRateLogged m_sampleTable;
    std::ofstream m_statsFile;
    Ptr<Node> m_node;

    // ========================================================================
    // FIXED: SAFE FEATURES ONLY (14 features, zero temporal leakage)
    // ========================================================================

    // SNR features (7 features - SAFE: measured before decision)
    double m_lastSnr;                ///< Last measured SNR (dB) [-30 to +45]
    double m_fastEwmaSnr;            ///< Fast EWMA SNR (alpha=0.30)
    double m_slowEwmaSnr;            ///< Slow EWMA SNR (alpha=0.05)
    std::deque<double> m_snrHistory; ///< Rolling SNR history (max 20 samples)

    // FIXED: Issue #33 - PREVIOUS window tracking (SAFE)
    // These represent COMPLETED transmission windows, not current packet outcomes
    std::deque<bool> m_previousShortWindow; ///< Previous short window (10 packets)
    std::deque<bool> m_previousMedWindow;   ///< Previous medium window (25 packets)
    uint32_t m_previousWindowSuccess;       ///< Success count from previous window
    uint32_t m_previousWindowTotal;         ///< Total packets in previous window
    uint32_t m_previousWindowLosses;        ///< Losses in previous window

    // FIXED: Issue #33 - CURRENT window (NOT logged - becomes previous later)
    std::deque<bool> m_currentShortWindow; ///< Current window (becomes previous)
    std::deque<bool> m_currentMedWindow;   ///< Current window (becomes previous)
    uint32_t m_currentWindowPackets;       ///< Packet count in current window

    // Station ID for logging
    uint32_t m_stationId; ///< Unique station identifier

    // Constructor with proper initialization
    MinstrelWifiRemoteStationLogged()
        : m_nextStatsUpdate(Seconds(0)),
          m_col(0),
          m_index(0),
          m_maxTpRate(0),
          m_maxTpRate2(0),
          m_maxProbRate(0),
          m_nModes(0),
          m_totalPacketsCount(0),
          m_samplePacketsCount(0),
          m_numSamplesDeferred(0),
          m_isSampling(false),
          m_sampleRate(0),
          m_sampleDeferred(false),
          m_shortRetry(0),
          m_longRetry(0),
          m_retry(0),
          m_txrate(0),
          m_initialized(false),
          m_lastSnr(0.0),
          m_fastEwmaSnr(0.0),
          m_slowEwmaSnr(0.0),
          m_previousWindowSuccess(0),
          m_previousWindowTotal(0),
          m_previousWindowLosses(0),
          m_currentWindowPackets(0),
          m_stationId(0)
    {
    }

    ~MinstrelWifiRemoteStationLogged() override = default;
};

/**
 * \brief Minstrel Rate Control Algorithm with SAFE Feature Logging
 * \ingroup wifi
 *
 * FIXED VERSION (2025-10-01 15:13:30 UTC):
 * - 14 safe features (zero temporal leakage)
 * - Success ratios from PREVIOUS window (Issue #33)
 * - Scenario file for train/test splitting (Issue #4)
 * - Realistic accuracy expectations: 65-75%
 *
 * The core Minstrel algorithm is UNCHANGED. This version only logs
 * SAFE features that are known BEFORE the rate decision.
 *
 * CSV Output Format (14 features + metadata):
 * time,stationId,rateIdx,phyRate,
 * lastSnr,snrFast,snrSlow,snrTrendShort,snrStabilityIndex,snrPredictionConfidence,snrVariance,
 * shortSuccRatio,medSuccRatio,packetLossRate,
 * channelWidth,mobilityMetric,
 * severity,confidence,
 * scenario_file
 */
class MinstrelWifiManagerLogged : public WifiRemoteStationManager
{
  public:
    static TypeId GetTypeId();
    MinstrelWifiManagerLogged();
    ~MinstrelWifiManagerLogged() override;

    void SetupPhy(const Ptr<WifiPhy> phy) override;
    void SetupMac(const Ptr<WifiMac> mac) override;
    int64_t AssignStreams(int64_t stream) override;

    // Trace sources
    TracedCallback<std::string, bool> m_packetResultTrace;
    TracedCallback<uint64_t, uint64_t> m_rateChange;

    // Core Minstrel methods (unchanged)
    void UpdateRate(MinstrelWifiRemoteStationLogged* station);
    void UpdateStats(MinstrelWifiRemoteStationLogged* station);
    uint16_t FindRate(MinstrelWifiRemoteStationLogged* station);
    WifiTxVector GetDataTxVector(MinstrelWifiRemoteStationLogged* station);
    WifiTxVector GetRtsTxVector(MinstrelWifiRemoteStationLogged* station);
    uint32_t CountRetries(MinstrelWifiRemoteStationLogged* station);
    void UpdatePacketCounters(MinstrelWifiRemoteStationLogged* station);
    void UpdateRetry(MinstrelWifiRemoteStationLogged* station);
    void CheckInit(MinstrelWifiRemoteStationLogged* station);
    void InitSampleTable(MinstrelWifiRemoteStationLogged* station);

    void SetScenarioParameters(double distance, uint32_t interferers);

    /**
     * FIXED: Issue #33 - Update window state
     * Moves current window to previous window when full
     *
     * \param station the station object
     */
    void UpdateWindowState(MinstrelWifiRemoteStationLogged* station);

  protected:
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

    bool DoNeedRetransmission(WifiRemoteStation* st,
                              Ptr<const Packet> packet,
                              bool normally) override;

  private:
    MinstrelWifiRemoteStationLogged* Lookup(WifiRemoteStation* st) const
    {
        return static_cast<MinstrelWifiRemoteStationLogged*>(st);
    }

    Time GetCalcTxTime(WifiMode mode) const;
    void AddCalcTxTime(WifiMode mode, Time t);
    void RateInit(MinstrelWifiRemoteStationLogged* station);
    uint16_t GetNextSample(MinstrelWifiRemoteStationLogged* station);
    Time CalculateTimeUnicastPacket(Time dataTransmissionTime,
                                    uint32_t shortRetries,
                                    uint32_t longRetries);
    void PrintSampleTable(MinstrelWifiRemoteStationLogged* station) const;
    void PrintTable(MinstrelWifiRemoteStationLogged* station);

    // ========================================================================
    // FIXED: Safe feature calculation methods (Issue #1, #33)
    // ========================================================================

    /**
     * Calculate SNR stability index (0-10 scale)
     * SAFE: Uses historical SNR measurements only
     */
    double CalculateSnrStability(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * Calculate SNR prediction confidence (0-1)
     * SAFE: Based on SNR variance
     */
    double CalculateSnrPredictionConfidence(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * Calculate SNR variance
     * SAFE: Historical measurements
     */
    double CalculateSnrVariance(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * FIXED: Issue #33 - Success ratios from PREVIOUS window
     * These are SAFE because they use COMPLETED transmission window data
     */
    double CalculatePreviousShortSuccessRatio(MinstrelWifiRemoteStationLogged* st) const;
    double CalculatePreviousMedSuccessRatio(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * FIXED: Issue #33 - Packet loss from PREVIOUS window
     * SAFE: Uses completed window data
     */
    double CalculatePreviousPacketLoss(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * Calculate severity (network condition assessment)
     * SAFE: Based on previous window performance
     */
    double CalculateSeverity(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * Calculate confidence in assessment
     * SAFE: Based on SNR stability and previous window success
     */
    double CalculateConfidence(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * Calculate mobility metric (0-1)
     * SAFE: Based on SNR variance
     */
    double CalculateMobilityMetric(MinstrelWifiRemoteStationLogged* st) const;

    /**
     * Get SNR tier for rate mapping
     * SAFE: Uses current SNR measurement
     */
    uint8_t TierFromSnr(double snr) const;

    /**
     * Stratified logging probability (for balanced dataset)
     */
    double GetStratifiedLogProbability(uint8_t rate, double snr, bool success) const;

    /**
     * Random value generator for stratified sampling
     */
    double GetRandomValue();

    /**
     * FIXED: Safe logging function - logs only 14 pre-decision features
     * Called AFTER packet transmission to update windows, but logs
     * features that were available BEFORE the decision was made
     *
     * \param st station state
     * \param currentRateIdx current rate index (0-7 for 802.11a)
     * \param success whether packet succeeded (for window tracking only)
     */
    void LogSafeFeatures(MinstrelWifiRemoteStationLogged* st, uint8_t currentRateIdx, bool success);

    /**
     * Write CSV header with 14 safe features + metadata
     */
    void WriteLogHeader();

    // Core Minstrel parameters
    typedef std::map<WifiMode, Time> TxTime;
    TxTime m_calcTxTime;
    Time m_updateStats;
    uint8_t m_lookAroundRate;
    uint8_t m_ewmaLevel;
    uint8_t m_sampleCol;
    uint32_t m_pktLen;
    bool m_printStats;
    bool m_printSamples;

    Ptr<UniformRandomVariable> m_uniformRandomVariable;

    // ========================================================================
    // FIXED: Logging infrastructure
    // ========================================================================
    std::ofstream m_logFile;
    std::string m_logFilePath;
    std::string m_scenarioFileName; ///< FIXED: Issue #4 - Scenario identifier
    bool m_logHeaderWritten;

    // Random number generation for stratified sampling
    std::mt19937 m_rng;
    std::uniform_real_distribution<double> m_uniformDist;

    // Window size for previous/current tracking (Issue #33)
    static constexpr uint32_t WINDOW_SIZE = 50; ///< Packets per window

    // Station ID counter
    mutable uint32_t m_nextStationId;

    double m_scenarioDistance;      ///< Current scenario distance (meters)
    uint32_t m_scenarioInterferers; ///< Current scenario interferer count
};

} // namespace ns3

#endif /* MINSTREL_WIFI_MANAGER_LOGGED_H */