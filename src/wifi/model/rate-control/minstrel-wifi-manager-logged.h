/*
 * Minstrel WiFi Manager Logged - PHASE 1B COMBINED (Most comprehensive)
 *
 * Merges PHASE 1A (20 SAFE features) and PHASE 1B (4 new features)
 * into a single, consolidated header that contains all structs, members,
 * method declarations, and logging/scenario infrastructure from both
 * inputs. Preserves original authorship headers and adds PHASE 1B note.
 *
 * Authors: Duy Nguyen <duy@soe.ucsc.edu>
 *          Matías Richart <mrichart@fing.edu.uy>
 *          ahmedjk34 <https://github.com/ahmedjk34> (2025-10-03 PHASE 1A)
 *          Combined by ChatGPT (2025-10-03 PHASE 1B merge)
 *
 * LICENSE: GNU General Public License version 2
 */

#ifndef MINSTREL_WIFI_MANAGER_LOGGED_H
#define MINSTREL_WIFI_MANAGER_LOGGED_H

#include "ns3/node.h"
#include "ns3/traced-value.h"
#include "ns3/wifi-remote-station-manager.h"

#include <cmath>
#include <cstdint>
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
 * (merged and normalized types)
 */
struct RateInfoLogged
{
    Time perfectTxTime;          ///< Perfect transmission time (ns-3 Time)
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
 * (PHASE 1A + 1B merged)
 *
 * - SAFE: no temporal leakage features present
 * - PHASE 1A: 20 safe, pre-decision features tracked
 * - PHASE 1B: 4 additional telemetry features added (rssiVariance,
 *             interferenceLevel, distanceMetric, avgPacketSize)
 */
struct MinstrelWifiRemoteStationLogged : public WifiRemoteStation
{
    // Core Minstrel state (unchanged)
    Time m_nextStatsUpdate;
    uint8_t m_col;
    uint8_t m_index;
    uint16_t m_maxTpRate; // kept as 16-bit to be compatible
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

    // Station ID for logging
    uint32_t m_stationId;

    // ========================================================================
    // FIXED: SAFE FEATURES ONLY (PHASE 1A) + PHASE 1B additions
    // ========================================================================

    // SNR features (7 features - SAFE: measured before decision)
    double m_lastSnr;                ///< Last measured SNR (dB) [-30 to +45]
    double m_fastEwmaSnr;            ///< Fast EWMA SNR (alpha=0.30)
    double m_slowEwmaSnr;            ///< Slow EWMA SNR (alpha=0.05)
    std::deque<double> m_snrHistory; ///< Rolling SNR history (max 200 samples)

    // FIXED: Issue #33 - PREVIOUS window tracking (SAFE)
    std::deque<bool> m_previousShortWindow; ///< Previous short window (10 packets)
    std::deque<bool> m_previousMedWindow;   ///< Previous medium window (25 packets)
    uint32_t m_previousWindowSuccess;       ///< Success count from previous window
    uint32_t m_previousWindowTotal;         ///< Total packets in previous window
    uint32_t m_previousWindowLosses;        ///< Losses in previous window

    // FIXED: Issue #33 - CURRENT window (NOT logged - becomes previous later)
    std::deque<bool> m_currentShortWindow; ///< Current window (becomes previous)
    std::deque<bool> m_currentMedWindow;   ///< Current window (becomes previous)
    uint32_t m_currentWindowPackets;       ///< Packet count in current window

    // PHASE 1A: ns-3 telemetry for 6 new features
    uint32_t m_framesRetried;          ///< Frames retried in current window
    uint32_t m_framesSent;             ///< Total frames sent in current window
    uint32_t m_framesFailed;           ///< Frames that failed in current window
    double m_channelBusyTime;          ///< Accumulated channel busy time (seconds)
    double m_observationTime;          ///< Total observation time (seconds)
    std::deque<uint8_t> m_rateHistory; ///< Last 10 rate decisions (for rateStability)
    uint32_t m_packetsSinceRateChange; ///< Packets since last rate change

    // PHASE 1B: NEW FEATURE TRACKING (4 features)
    std::deque<double> m_rssiHistory;         ///< For rssiVariance (last N samples)
    uint32_t m_recentCollisions;              ///< Collisions in observation window
    uint32_t m_recentTransmissions;           ///< Transmissions in observation window
    std::deque<uint32_t> m_packetSizeHistory; ///< For avgPacketSize (last N packets)

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
          m_currentWindowPackets(0),
          m_previousWindowLosses(0),
          m_framesRetried(0),
          m_framesSent(0),
          m_framesFailed(0),
          m_channelBusyTime(0.0),
          m_observationTime(0.1),
          m_packetsSinceRateChange(0),
          m_stationId(0),
          m_recentCollisions(0),
          m_recentTransmissions(0)
    {
    }

    ~MinstrelWifiRemoteStationLogged() override = default;
};

/**
 * \brief Minstrel Rate Control Algorithm with SAFE Feature Logging (PHASE 1B)
 * \ingroup wifi
 *
 * PHASE 1B VERSION (merged):
 * - 24 safe features (20 Phase 1A + 4 Phase 1B)
 * - Success ratios from PREVIOUS window (Issue #33)
 * - Scenario file for train/test splitting (Issue #4)
 * - Realistic accuracy expectations: 65-75%
 *
 * CSV Output Format (merged):
 * time,stationId,rateIdx,phyRate,
 * lastSnr,snrFast,snrSlow,snrTrendShort,snrStabilityIndex,snrPredictionConfidence,snrVariance,
 * shortSuccRatio,medSuccRatio,packetLossRate,
 * channelWidth,mobilityMetric,
 * severity,confidence,
 * retryRate,frameErrorRate,channelBusyRatio,recentRateAvg,rateStability,sinceLastChange,
 * rssiVariance,interferenceLevel,distanceMetric,avgPacketSize,
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

    /**
     * PHASE 1B: Set scenario parameters (distance, interferers)
     */
    void SetScenarioParameters(double distance, uint32_t interferers);

    /**
     * FIXED: Issue #33 - Update window state
     * Moves current window to previous window when full
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
    // SAFE: Feature calculation methods (Issue #1, #33, PHASE 1A + 1B)
    // ========================================================================

    double CalculateSnrStability(MinstrelWifiRemoteStationLogged* st) const;
    double CalculateSnrPredictionConfidence(MinstrelWifiRemoteStationLogged* st) const;
    double CalculateSnrVariance(MinstrelWifiRemoteStationLogged* st) const;

    // Issue #33 - previous window metrics
    double CalculatePreviousShortSuccessRatio(MinstrelWifiRemoteStationLogged* st) const;
    double CalculatePreviousMedSuccessRatio(MinstrelWifiRemoteStationLogged* st) const;
    double CalculatePreviousPacketLoss(MinstrelWifiRemoteStationLogged* st) const;

    double CalculateSeverity(MinstrelWifiRemoteStationLogged* st) const;
    double CalculateConfidence(MinstrelWifiRemoteStationLogged* st) const;
    double CalculateMobilityMetric(MinstrelWifiRemoteStationLogged* st) const;

    // PHASE 1B: NEW FEATURE CALCULATIONS
    double CalculateRssiVariance(MinstrelWifiRemoteStationLogged* st) const;
    double CalculateInterferenceLevel(MinstrelWifiRemoteStationLogged* st) const;
    double GetDistanceMetric() const; // returns scenario distance metric
    double CalculateAvgPacketSize(MinstrelWifiRemoteStationLogged* st) const;

    // Stratified logging probability (for balanced dataset)
    double GetStratifiedLogProbability(uint8_t rate, double snr, bool success) const;

    // Random value generator for stratified sampling
    double GetRandomValue();

    /**
     * PHASE 1A/1B: Safe logging function - logs pre-decision features
     * Called AFTER packet transmission to update windows, but logs
     * features that were available BEFORE the decision was made
     */
    void LogSafeFeatures(MinstrelWifiRemoteStationLogged* st, uint8_t currentRateIdx, bool success);

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
    // FIXED: Logging & Sampling infrastructure
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
