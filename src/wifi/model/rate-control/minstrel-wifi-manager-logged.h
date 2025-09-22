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
 *
 * MinstrelWifiManagerLogged - Phase 1: Enhanced Data Collection & Feature Engineering
 * - Feedback-oriented features added to logger
 * - Stratified probabilistic logging implemented
 * - All new code marked with PHASE1 NEW CODE comments
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
 * A struct to contain all information related to a data rate for logged Minstrel
 */
struct RateInfoLogged
{
    /**
     * Perfect transmission time calculation, or frame calculation
     * Given a bit rate and a packet length n bytes
     */
    Time perfectTxTime;

    uint32_t retryCount;         ///< retry limit
    uint32_t adjustedRetryCount; ///< adjust the retry limit for this rate
    uint32_t numRateAttempt;     ///< how many number of attempts so far
    uint32_t numRateSuccess;     ///< number of successful packets
    uint32_t prob;               ///< (# packets success)/(# total packets)
    /**
     * EWMA calculation
     * ewma_prob =[prob *(100 - ewma_level) + (ewma_prob_old * ewma_level)]/100
     */
    uint32_t ewmaProb;
    uint32_t throughput; ///< throughput of a rate in bps

    uint32_t prevNumRateAttempt; //!< Number of transmission attempts with previous rate.
    uint32_t prevNumRateSuccess; //!< Number of successful frames transmitted with previous rate.
    uint64_t successHist;        //!< Aggregate of all transmission successes.
    uint64_t attemptHist;        //!< Aggregate of all transmission attempts.

    uint8_t numSamplesSkipped; //!< number of samples skipped
    int sampleLimit;           //!< sample limit
};

/**
 * Data structure for a Minstrel Rate table
 * A vector of a struct RateInfoLogged
 */
typedef std::vector<RateInfoLogged> MinstrelRateLogged;
/**
 * Data structure for a Sample Rate table
 * A vector of a vector uint8_t
 */
typedef std::vector<std::vector<uint8_t>> SampleRateLogged;

/**
 * \brief hold per-remote-station state for Minstrel Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the Minstrel Wifi manager
 */
struct MinstrelWifiRemoteStationLogged : public WifiRemoteStation
{
    Time m_nextStatsUpdate; ///< 10 times every second

    /**
     * To keep track of the current position in the our random sample table
     * going row by row from 1st column until the 10th column(Minstrel defines 10)
     * then we wrap back to the row 1 column 1.
     * note: there are many other ways to do this.
     */
    uint8_t m_col;                      ///< column index
    uint8_t m_index;                    ///< vector index
    uint16_t m_maxTpRate;               ///< the current throughput rate in bps
    uint16_t m_maxTpRate2;              ///< second highest throughput rate in bps
    uint16_t m_maxProbRate;             ///< rate with highest probability of success in bps
    uint8_t m_nModes;                   ///< number of modes supported
    int m_totalPacketsCount;            ///< total number of packets as of now
    int m_samplePacketsCount;           ///< how many packets we have sample so far
    int m_numSamplesDeferred;           ///< number samples deferred
    bool m_isSampling;                  ///< a flag to indicate we are currently sampling
    uint16_t m_sampleRate;              ///< current sample rate in bps
    bool m_sampleDeferred;              ///< a flag to indicate sample rate is on the second stage
    uint32_t m_shortRetry;              ///< short retries such as control packets
    uint32_t m_longRetry;               ///< long retries such as data packets
    uint32_t m_retry;                   ///< total retries short + long
    uint16_t m_txrate;                  ///< current transmit rate in bps
    bool m_initialized;                 ///< for initializing tables
    MinstrelRateLogged m_minstrelTable; ///< minstrel table (using logged version)
    SampleRateLogged m_sampleTable;     ///< sample table (using logged version)
    std::ofstream m_statsFile;          ///< stats file
    Ptr<Node> m_node;                   ///< node reference for station ID

    // PHASE1 NEW CODE: Feedback-oriented feature buffers
    std::vector<uint8_t> m_rateHistory;       // Recent rates for adaptation dynamics
    std::vector<double> m_throughputHistory;  // Recent throughput (Mbps or rate index as proxy)
    std::vector<uint32_t> m_retryHistory;     // Recent retry counts
    std::vector<bool> m_packetSuccessHistory; // Recent packet success/failure

    // PHASE1 NEW CODE: SNR-related state
    double m_lastSnr{0.0};
    std::deque<double> m_snrHistory;  // rolling SNR values
    std::vector<double> m_snrSamples; // raw SNR samples for quantile stats
    double m_fastEwmaSnr{0.0};        // fast EWMA of SNR
    double m_slowEwmaSnr{0.0};        // slow EWMA of SNR

    // PHASE1 NEW CODE: Additional logging state
    uint32_t m_sinceLastRateChange{0};
    uint32_t m_consecSuccess{0};
    uint32_t m_consecFailure{0};

    // Default constructor
    MinstrelWifiRemoteStationLogged() = default;

    // Virtual destructor to match base class
    ~MinstrelWifiRemoteStationLogged() override = default;
};

/**
 * \brief Implementation of Minstrel Rate Control Algorithm with Logging
 * \ingroup wifi
 *
 * This is a logged version of MinstrelWifiManager that preserves the exact
 * algorithm logic while adding comprehensive data collection and feature
 * engineering capabilities for analysis and machine learning.
 *
 * The core Minstrel algorithm remains unchanged - this version only adds
 * stratified logging, feedback-oriented features, and trace sources.
 */
class MinstrelWifiManagerLogged : public WifiRemoteStationManager
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    MinstrelWifiManagerLogged();
    ~MinstrelWifiManagerLogged() override;

    void SetupPhy(const Ptr<WifiPhy> phy) override;
    void SetupMac(const Ptr<WifiMac> mac) override;
    int64_t AssignStreams(int64_t stream) override;

    // Trace sources
    TracedCallback<std::string, bool> m_packetResultTrace;
    TracedCallback<uint64_t, uint64_t> m_rateChange;

    /**
     * Update the rate.
     *
     * \param station the station object
     */
    void UpdateRate(MinstrelWifiRemoteStationLogged* station);

    /**
     * Update the Minstrel Table.
     *
     * \param station the station object
     */
    void UpdateStats(MinstrelWifiRemoteStationLogged* station);

    /**
     * Find a rate to use from Minstrel Table.
     *
     * \param station the station object
     * \returns the rate in bps
     */
    uint16_t FindRate(MinstrelWifiRemoteStationLogged* station);

    /**
     * Get data transmit vector.
     *
     * \param station the station object
     * \returns WifiTxVector
     */
    WifiTxVector GetDataTxVector(MinstrelWifiRemoteStationLogged* station);

    /**
     * Get RTS transmit vector.
     *
     * \param station the station object
     * \returns WifiTxVector
     */
    WifiTxVector GetRtsTxVector(MinstrelWifiRemoteStationLogged* station);

    /**
     * Get the number of retries.
     *
     * \param station the station object
     * \returns the number of retries
     */
    uint32_t CountRetries(MinstrelWifiRemoteStationLogged* station);

    /**
     * Update packet counters.
     *
     * \param station the station object
     */
    void UpdatePacketCounters(MinstrelWifiRemoteStationLogged* station);

    /**
     * Update the number of retries and reset accordingly.
     *
     * \param station the station object
     */
    void UpdateRetry(MinstrelWifiRemoteStationLogged* station);

    /**
     * Check for initializations.
     *
     * \param station the station object
     */
    void CheckInit(MinstrelWifiRemoteStationLogged* station);

    /**
     * Initialize Sample Table.
     *
     * \param station the station object
     */
    void InitSampleTable(MinstrelWifiRemoteStationLogged* station);

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

    /**
     * Estimate the TxTime of a packet with a given mode.
     *
     * \param mode Wi-Fi mode
     * \returns the transmission time
     */
    Time GetCalcTxTime(WifiMode mode) const;
    /**
     * Add transmission time for the given mode to an internal list.
     *
     * \param mode Wi-Fi mode
     * \param t transmission time
     */
    void AddCalcTxTime(WifiMode mode, Time t);

    /**
     * Initialize Minstrel Table.
     *
     * \param station the station object
     */
    void RateInit(MinstrelWifiRemoteStationLogged* station);

    /**
     * Get the next sample from Sample Table.
     *
     * \param station the station object
     * \returns the next sample
     */
    uint16_t GetNextSample(MinstrelWifiRemoteStationLogged* station);

    /**
     * Estimate the time to transmit the given packet with the given number of retries.
     * This function is "roughly" the function "calc_usecs_unicast_packet" in minstrel.c
     * in the madwifi implementation.
     *
     * The basic idea is that, we try to estimate the "average" time used to transmit the
     * packet for the given number of retries while also accounting for the 802.11 congestion
     * window change. The original code in the madwifi seems to estimate the number of backoff
     * slots as the half of the current CW size.
     *
     * There are four main parts:
     *  - wait for DIFS (sense idle channel)
     *  - Ack timeouts
     *  - Data transmission
     *  - backoffs according to CW
     *
     * \param dataTransmissionTime the data transmission time
     * \param shortRetries short retries
     * \param longRetries long retries
     * \returns the unicast packet time
     */
    Time CalculateTimeUnicastPacket(Time dataTransmissionTime,
                                    uint32_t shortRetries,
                                    uint32_t longRetries);

    /**
     * Print Sample Table.
     *
     * \param station the station object
     */
    void PrintSampleTable(MinstrelWifiRemoteStationLogged* station) const;

    /**
     * Print Minstrel Table.
     *
     * \param station the station object
     */
    void PrintTable(MinstrelWifiRemoteStationLogged* station);

    // PHASE1 NEW CODE: Stratified logging and feature engineering
    std::mt19937 m_rng;
    std::uniform_real_distribution<double> m_uniformDist;
    double GetStratifiedLogProbability(uint8_t rate, double snr, bool success);
    double GetRandomValue();

    // PHASE1 NEW CODE: Feedback-oriented helpers
    uint32_t CountRecentRateChanges(MinstrelWifiRemoteStationLogged* st, uint32_t window);
    double CalculateRateStability(MinstrelWifiRemoteStationLogged* st);
    double CalculateRecentThroughput(MinstrelWifiRemoteStationLogged* st, uint32_t window);
    double CalculateRecentPacketLoss(MinstrelWifiRemoteStationLogged* st, uint32_t window);
    double CalculateRetrySuccessRatio(MinstrelWifiRemoteStationLogged* st);
    double CalculateOptimalRateDistance(MinstrelWifiRemoteStationLogged* st);
    double CalculateAggressiveFactor(MinstrelWifiRemoteStationLogged* st);
    double CalculateConservativeFactor(MinstrelWifiRemoteStationLogged* st);
    uint8_t GetRecommendedSafeRate(MinstrelWifiRemoteStationLogged* st);
    double CalculateSnrStability(MinstrelWifiRemoteStationLogged* st);
    double CalculateSnrPredictionConfidence(MinstrelWifiRemoteStationLogged* st);
    uint8_t TierFromSnr(MinstrelWifiRemoteStationLogged* st, double snr) const;

    void LogDecision(MinstrelWifiRemoteStationLogged* st, int decisionReason, bool packetSuccess);

    /**
     * typedef for a vector of a pair of Time, WifiMode.
     * Essentially a map from WifiMode to its corresponding transmission time
     * to transmit a reference packet.
     */
    typedef std::map<WifiMode, Time> TxTime;

    TxTime m_calcTxTime;      ///< to hold all the calculated TxTime for all modes
    Time m_updateStats;       ///< how frequent do we calculate the stats
    uint8_t m_lookAroundRate; ///< the % to try other rates than our current rate
    uint8_t m_ewmaLevel;      ///< exponential weighted moving average
    uint8_t m_sampleCol;      ///< number of sample columns
    uint32_t m_pktLen;        ///< packet length used to calculate mode TxTime
    bool m_printStats;        ///< whether statistics table should be printed.
    bool m_printSamples;      ///< whether samples table should be printed.

    /// Provides uniform random variables.
    Ptr<UniformRandomVariable> m_uniformRandomVariable;

    // Logging infrastructure
    std::ofstream m_logFile;
    std::string m_logFilePath;
    bool m_logHeaderWritten;

    // Trace fields for constructor initialization
    uint64_t m_currentRate;
    double m_traceConfidence;
    double m_traceSeverity;
    double m_traceT1;
    double m_traceT2;
    double m_traceT3;

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
        RAISE_DYNAMIC_RELAX,
        MINSTREL_SAMPLE,
        MINSTREL_MAX_TP,
        MINSTREL_MAX_PROB
    };
};

} // namespace ns3

#endif /* MINSTREL_WIFI_MANAGER_LOGGED_H */