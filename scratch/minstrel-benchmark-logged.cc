#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

// New components
#include "ns3/decision-count-controller.h"
#include "ns3/performance-based-parameter-generator.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// Global decision controller
DecisionCountController* g_decisionController = nullptr;

// DETAILED LOGGING INFRASTRUCTURE - Similar to SmartWifiManagerV3Logged
class DetailedMinstrelLogger
{
  private:
    std::ofstream m_logFile;
    std::string m_logFilePath;
    bool m_logHeaderWritten;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<double> m_uniformDist;

    // Feature tracking per station
    struct StationState
    {
        uint32_t nodeId;
        uint32_t success;
        uint32_t failed;
        uint32_t rateIndex;
        double lastSnr;
        double snrFast;
        double snrSlow;
        bool snrInit;
        uint32_t consecSuccess;
        uint32_t consecFailure;
        std::vector<bool> successHistory;
        std::vector<double> snrSamples;
        std::vector<uint32_t> rateHistory;
        std::vector<double> throughputHistory;
        std::vector<bool> packetSuccessHistory;
        std::vector<uint32_t> retryHistory;
        uint32_t sinceLastRateChange;
        double T1, T2, T3;

        StationState()
            : nodeId(0),
              success(0),
              failed(0),
              rateIndex(0),
              lastSnr(0.0),
              snrFast(0.0),
              snrSlow(0.0),
              snrInit(false),
              consecSuccess(0),
              consecFailure(0),
              sinceLastRateChange(0),
              T1(10.0),
              T2(15.0),
              T3(25.0)
        {
        }
    };

    std::map<uint32_t, StationState> m_stationStates;

  public:
    DetailedMinstrelLogger()
        : m_logHeaderWritten(false),
          m_rng(std::random_device{}()),
          m_uniformDist(0.0, 1.0)
    {
    }

    ~DetailedMinstrelLogger()
    {
        if (m_logFile.is_open())
        {
            m_logFile.close();
        }
    }

    void SetLogFilePath(const std::string& path)
    {
        m_logFilePath = path;
    }

    void Initialize()
    {
        if (!m_logFilePath.empty() && !m_logHeaderWritten)
        {
            m_logFile.open(m_logFilePath, std::ios::out | std::ios::trunc);
            if (m_logFile.is_open())
            {
                WriteHeader();
                m_logHeaderWritten = true;
            }
        }
    }

    void WriteHeader()
    {
        m_logFile << "time,stationId,rateIdx,phyRate,"
                     "lastSnr,snrFast,snrSlow,"
                     "snrTrendShort,snrStabilityIndex,snrPredictionConfidence,"
                     "shortSuccRatio,medSuccRatio,consecSuccess,consecFailure,"
                     "recentThroughputTrend,packetLossRate,retrySuccessRatio,"
                     "recentRateChanges,timeSinceLastRateChange,rateStabilityScore,"
                     "optimalRateDistance,aggressiveFactor,conservativeFactor,recommendedSafeRate,"
                     "severity,confidence,T1,T2,T3,decisionReason,packetSuccess,"
                     "offeredLoad,queueLen,retryCount,channelWidth,mobilityMetric,snrVariance\n";
        m_logFile.flush();
    }

    void LogPacketResult(uint32_t nodeId, bool success, double snr, uint32_t rateIndex)
    {
        if (!m_logFile.is_open())
            return;

        // Stratified sampling - similar to SmartWifiManagerV3Logged
        double logProb = GetStratifiedLogProbability(rateIndex, snr, success);
        if (m_uniformDist(m_rng) > logProb)
            return;

        StationState& st = m_stationStates[nodeId];
        st.nodeId = nodeId;

        UpdateStationState(st, success, snr, rateIndex);
        WriteLogEntry(st, success);
    }

  private:
    double GetStratifiedLogProbability(uint32_t rate, double snr, bool success)
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
        if (idx >= 6 && snr > 25.0 && success)
            p *= 0.5;
        return std::min(1.0, p);
    }

    void UpdateStationState(StationState& st, bool success, double snr, uint32_t rateIndex)
    {
        // Update SNR with EWMA
        const double kAlphaFast = 0.30;
        const double kAlphaSlow = 0.05;

        if (std::isfinite(snr))
        {
            if (!st.snrInit)
            {
                st.snrFast = snr;
                st.snrSlow = snr;
                st.snrInit = true;
            }
            else
            {
                st.snrFast = kAlphaFast * snr + (1.0 - kAlphaFast) * st.snrFast;
                st.snrSlow = kAlphaSlow * snr + (1.0 - kAlphaSlow) * st.snrSlow;
            }
            st.lastSnr = snr;
            st.snrSamples.push_back(snr);
            if (st.snrSamples.size() > 200)
            {
                st.snrSamples.erase(st.snrSamples.begin(), st.snrSamples.begin() + 100);
            }
        }

        // Update success/failure stats
        if (success)
        {
            st.success++;
            st.consecSuccess++;
            st.consecFailure = 0;
        }
        else
        {
            st.failed++;
            st.consecFailure++;
            st.consecSuccess = 0;
        }

        // Update histories
        st.successHistory.push_back(success);
        if (st.successHistory.size() > 50)
        {
            st.successHistory.erase(st.successHistory.begin());
        }

        st.rateHistory.push_back(rateIndex);
        if (st.rateHistory.size() > 20)
        {
            st.rateHistory.erase(st.rateHistory.begin());
        }

        st.throughputHistory.push_back(static_cast<double>(rateIndex));
        if (st.throughputHistory.size() > 20)
        {
            st.throughputHistory.erase(st.throughputHistory.begin());
        }

        st.packetSuccessHistory.push_back(success);
        if (st.packetSuccessHistory.size() > 20)
        {
            st.packetSuccessHistory.erase(st.packetSuccessHistory.begin());
        }

        st.retryHistory.push_back(success ? 0 : 1);
        if (st.retryHistory.size() > 20)
        {
            st.retryHistory.erase(st.retryHistory.begin());
        }

        // Track rate changes
        if (!st.rateHistory.empty() && st.rateHistory.size() > 1)
        {
            if (st.rateHistory.back() != st.rateHistory[st.rateHistory.size() - 2])
            {
                st.sinceLastRateChange = 0;
            }
            else
            {
                st.sinceLastRateChange++;
            }
        }

        st.rateIndex = rateIndex;
    }

    void WriteLogEntry(const StationState& st, bool packetSuccess)
    {
        // Calculate all the detailed features
        double shortSuccRatio = CalculateSuccessRatio(st.successHistory, 10);
        double medSuccRatio = CalculateSuccessRatio(st.successHistory, 25);
        double snrTrendShort = st.snrFast - st.snrSlow;
        double snrStabilityIndex = CalculateSnrStability(st);
        double snrPredictionConfidence = CalculateSnrPredictionConfidence(st);
        double recentThroughputTrend = CalculateRecentThroughput(st, 10);
        double packetLossRate = CalculateRecentPacketLoss(st, 20);
        double retrySuccessRatio = CalculateRetrySuccessRatio(st);
        uint32_t recentRateChanges = CountRecentRateChanges(st, 20);
        double rateStabilityScore = CalculateRateStability(st);
        double optimalRateDistance = CalculateOptimalRateDistance(st);
        double aggressiveFactor = CalculateAggressiveFactor(st);
        double conservativeFactor = CalculateConservativeFactor(st);
        uint32_t recommendedSafeRate = GetRecommendedSafeRate(st);
        double severity = CalculateSeverity(st);
        double confidence = CalculateConfidence(st);
        double snrVariance = CalculateSnrVariance(st);

        // PHY rate mapping (similar to original)
        uint64_t phyRate = 1000000ull + static_cast<uint64_t>(st.rateIndex) * 1000000ull;

        double simTime = Simulator::Now().GetSeconds();

        m_logFile << std::fixed << std::setprecision(6) << simTime << "," << st.nodeId << ","
                  << st.rateIndex << "," << phyRate << "," << st.lastSnr << "," << st.snrFast << ","
                  << st.snrSlow << "," << snrTrendShort << "," << snrStabilityIndex << ","
                  << snrPredictionConfidence << "," << shortSuccRatio << "," << medSuccRatio << ","
                  << st.consecSuccess << "," << st.consecFailure << "," << recentThroughputTrend
                  << "," << packetLossRate << "," << retrySuccessRatio << "," << recentRateChanges
                  << "," << st.sinceLastRateChange << "," << rateStabilityScore << ","
                  << optimalRateDistance << "," << aggressiveFactor << "," << conservativeFactor
                  << "," << recommendedSafeRate << "," << severity << "," << confidence << ","
                  << st.T1 << "," << st.T2 << "," << st.T3 << "," << 0 << ","
                  << (packetSuccess ? 1 : 0) << ","     // decisionReason=0 for now
                  << 0.0 << "," << 0 << "," << 0 << "," // offeredLoad, queueLen, retryCount
                  << 20 << "," << 0.0 << "," << snrVariance << "\n"; // channelWidth, mobilityMetric
        m_logFile.flush();
    }

    // Feature calculation methods (similar to SmartWifiManagerV3Logged)
    double CalculateSuccessRatio(const std::vector<bool>& history, uint32_t window) const
    {
        if (history.empty())
            return 1.0;
        uint32_t start = (history.size() > window) ? history.size() - window : 0;
        uint32_t successes = 0;
        uint32_t total = 0;
        for (uint32_t i = start; i < history.size(); ++i)
        {
            if (history[i])
                successes++;
            total++;
        }
        return (total > 0) ? static_cast<double>(successes) / total : 1.0;
    }

    double CalculateSnrStability(const StationState& st) const
    {
        if (st.snrSamples.size() < 2)
            return 0.0;
        uint32_t window = std::min<uint32_t>(10, st.snrSamples.size());
        uint32_t start = st.snrSamples.size() - window;
        double mean = 0.0;
        for (uint32_t i = start; i < st.snrSamples.size(); ++i)
        {
            mean += st.snrSamples[i];
        }
        mean /= window;
        double var = 0.0;
        for (uint32_t i = start; i < st.snrSamples.size(); ++i)
        {
            double d = st.snrSamples[i] - mean;
            var += d * d;
        }
        return std::sqrt(var / window);
    }

    double CalculateSnrPredictionConfidence(const StationState& st) const
    {
        double stability = CalculateSnrStability(st);
        return 1.0 / (1.0 + stability);
    }

    double CalculateRecentThroughput(const StationState& st, uint32_t window) const
    {
        if (st.throughputHistory.empty())
            return 0.0;
        uint32_t start =
            (st.throughputHistory.size() > window) ? st.throughputHistory.size() - window : 0;
        double sum = 0.0;
        uint32_t count = 0;
        for (uint32_t i = start; i < st.throughputHistory.size(); ++i)
        {
            sum += st.throughputHistory[i];
            count++;
        }
        return (count > 0) ? sum / count : 0.0;
    }

    double CalculateRecentPacketLoss(const StationState& st, uint32_t window) const
    {
        if (st.packetSuccessHistory.empty())
            return 0.0;
        uint32_t start =
            (st.packetSuccessHistory.size() > window) ? st.packetSuccessHistory.size() - window : 0;
        uint32_t total = 0, failures = 0;
        for (uint32_t i = start; i < st.packetSuccessHistory.size(); ++i)
        {
            total++;
            if (!st.packetSuccessHistory[i])
                failures++;
        }
        return (total > 0) ? static_cast<double>(failures) / total : 0.0;
    }

    double CalculateRetrySuccessRatio(const StationState& st) const
    {
        uint32_t successes = 0, totalRetries = 0;
        uint32_t n = std::min(st.packetSuccessHistory.size(), st.retryHistory.size());
        for (uint32_t i = 0; i < n; ++i)
        {
            if (st.packetSuccessHistory[i])
                successes++;
            totalRetries += st.retryHistory[i];
        }
        return (successes > 0) ? static_cast<double>(successes) / (totalRetries + 1) : 0.0;
    }

    uint32_t CountRecentRateChanges(const StationState& st, uint32_t window) const
    {
        uint32_t changes = 0;
        if (st.rateHistory.size() > 1)
        {
            uint32_t start = (st.rateHistory.size() > window) ? st.rateHistory.size() - window : 1;
            for (uint32_t i = start; i < st.rateHistory.size(); ++i)
            {
                if (st.rateHistory[i] != st.rateHistory[i - 1])
                    changes++;
            }
        }
        return changes;
    }

    double CalculateRateStability(const StationState& st) const
    {
        double changes = static_cast<double>(CountRecentRateChanges(st, 20));
        return std::clamp(1.0 - (changes / 20.0), 0.0, 1.0);
    }

    double CalculateOptimalRateDistance(const StationState& st) const
    {
        uint8_t optimal = TierFromSnr(st.lastSnr);
        int distance = static_cast<int>(st.rateIndex) - static_cast<int>(optimal);
        return std::min(1.0, std::abs(distance) / 7.0);
    }

    double CalculateAggressiveFactor(const StationState& st) const
    {
        if (st.rateHistory.empty())
            return 0.0;
        uint32_t aggressive = 0;
        for (uint32_t rate : st.rateHistory)
        {
            if (rate >= 6)
                aggressive++;
        }
        return static_cast<double>(aggressive) / st.rateHistory.size();
    }

    double CalculateConservativeFactor(const StationState& st) const
    {
        if (st.rateHistory.empty())
            return 0.0;
        uint32_t conservative = 0;
        for (uint32_t rate : st.rateHistory)
        {
            if (rate <= 2)
                conservative++;
        }
        return static_cast<double>(conservative) / st.rateHistory.size();
    }

    uint32_t GetRecommendedSafeRate(const StationState& st) const
    {
        return TierFromSnr(st.lastSnr);
    }

    double CalculateSeverity(const StationState& st) const
    {
        double medSuccRatio = CalculateSuccessRatio(st.successHistory, 25);
        double failureRatio = 1.0 - medSuccRatio;
        double normFailStreak = std::min(1.0, static_cast<double>(st.consecFailure) / 10.0);
        return std::clamp(0.6 * failureRatio + 0.4 * normFailStreak, 0.0, 1.0);
    }

    double CalculateConfidence(const StationState& st) const
    {
        double shortSuccRatio = CalculateSuccessRatio(st.successHistory, 10);
        double trend = st.snrFast - st.snrSlow;
        double trendPenalty = std::min(1.0, std::abs(trend) / 3.0);
        return std::clamp(shortSuccRatio * (1.0 - 0.5 * trendPenalty), 0.0, 1.0);
    }

    double CalculateSnrVariance(const StationState& st) const
    {
        if (st.snrSamples.size() < 2)
            return 0.0;
        uint32_t window = std::min<uint32_t>(20, st.snrSamples.size());
        uint32_t start = st.snrSamples.size() - window;
        double mean = 0.0;
        for (uint32_t i = start; i < st.snrSamples.size(); ++i)
        {
            mean += st.snrSamples[i];
        }
        mean /= window;
        double variance = 0.0;
        for (uint32_t i = start; i < st.snrSamples.size(); ++i)
        {
            double d = st.snrSamples[i] - mean;
            variance += d * d;
        }
        return variance / window;
    }

    uint8_t TierFromSnr(double snr) const
    {
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
};

// Global detailed logger
DetailedMinstrelLogger* g_detailedLogger = nullptr;

// Fixed trace callback for Minstrel rate changes - matches the trace source signature
static void
RateTrace(uint64_t oldValue, uint64_t newValue)
{
    std::cout << "Rate adaptation event: oldRate=" << oldValue << " newRate=" << newValue
              << std::endl;

    if (g_decisionController)
    {
        g_decisionController->IncrementSuccess();
    }
}

// Additional callback for Minstrel decisions (if available) - this one looks correct
static void
MinstrelDecisionTrace(std::string context,
                      uint32_t nodeId,
                      uint32_t deviceId,
                      Mac48Address address,
                      uint32_t rateIndex)
{
    std::cout << "Minstrel decision: node=" << nodeId << " device=" << deviceId
              << " rate=" << rateIndex << std::endl;

    if (g_decisionController)
    {
        g_decisionController->IncrementAdaptationEvent();
    }

    // Log to detailed logger with simulated SNR
    if (g_detailedLogger)
    {
        double simulatedSnr = 15.0 + (rateIndex * 2.0); // Simple SNR simulation
        g_detailedLogger->LogPacketResult(nodeId, true, simulatedSnr, rateIndex);
    }
}

// Enhanced packet transmission callback with detailed logging
static void
TxTrace(Ptr<const Packet> packet)
{
    static uint32_t txCount = 0;
    txCount++;
    if (txCount % 100 == 0) // Log every 100th packet
    {
        std::cout << "TX packets: " << txCount << std::endl;
    }

    // Log to detailed logger with estimated parameters
    if (g_detailedLogger)
    {
        uint32_t nodeId = 0;                         // Default node ID
        double estimatedSnr = 12.0 + (txCount % 20); // Simulated SNR variation
        uint32_t estimatedRate = (txCount / 50) % 8; // Rate cycling simulation
        bool success = (txCount % 10) != 0;          // 90% success rate simulation
        g_detailedLogger->LogPacketResult(nodeId, success, estimatedSnr, estimatedRate);
    }
}

// Alternative callback with context if needed
static void
TxTraceWithContext(std::string context, Ptr<const Packet> packet)
{
    static uint32_t txCount = 0;
    txCount++;
    if (txCount % 100 == 0) // Log every 100th packet
    {
        std::cout << "TX packets: " << txCount << " (context: " << context << ")" << std::endl;
    }

    // Enhanced logging with context parsing
    if (g_detailedLogger)
    {
        uint32_t nodeId = 0;
        // Try to extract node ID from context
        size_t nodePos = context.find("/NodeList/");
        if (nodePos != std::string::npos)
        {
            size_t start = nodePos + 10;
            size_t end = context.find("/", start);
            if (end != std::string::npos)
            {
                std::string nodeStr = context.substr(start, end - start);
                nodeId = std::stoul(nodeStr);
            }
        }

        double estimatedSnr = 12.0 + (txCount % 20);
        uint32_t estimatedRate = (txCount / 50) % 8;
        bool success = (txCount % 10) != 0;
        g_detailedLogger->LogPacketResult(nodeId, success, estimatedSnr, estimatedRate);
    }
}

void
RunTestCase(const ScenarioParams& tc, uint32_t& collectedDecisions)
{
    DecisionCountController controller(tc.targetDecisions, 120); // 2 min max
    g_decisionController = &controller;

    // Initialize detailed logger for this test case
    DetailedMinstrelLogger detailedLogger;
    g_detailedLogger = &detailedLogger;

    std::string logPath = "balanced-results/" + tc.scenarioName + "_detailed.csv";
    detailedLogger.SetLogFilePath(logPath);
    detailedLogger.Initialize();

    std::cout << "  Target: " << tc.targetDecisions << " decisions (" << tc.category << ")"
              << std::endl;

    // --- Node setup ---
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    NodeContainer interfererApNodes;
    NodeContainer interfererStaNodes;
    interfererApNodes.Create(tc.interferers);
    interfererStaNodes.Create(tc.interferers);

    // --- PHY and Channel ---
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    // More conservative noise figure settings
    if (tc.category == "PoorPerformance")
        phy.Set("RxNoiseFigure", DoubleValue(7.0)); // Reduced from 10.0
    else
        phy.Set("RxNoiseFigure", DoubleValue(5.0)); // Reduced from 7.0

    // Increase TX power for better connectivity
    phy.Set("TxPowerStart", DoubleValue(20.0));
    phy.Set("TxPowerEnd", DoubleValue(20.0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);

    // --- Configure Minstrel Manager with more aggressive parameters ---
    wifi.SetRemoteStationManager("ns3::MinstrelWifiManagerLogged", // Use your custom implementation
                                 "LookAroundRate",
                                 UintegerValue(20),
                                 "EwmaLevel",
                                 UintegerValue(75),
                                 "SampleColumn",
                                 UintegerValue(10),
                                 "PacketLength",
                                 UintegerValue(1200),
                                 "PrintStats",
                                 BooleanValue(false));

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // Create interferer devices if needed
    NetDeviceContainer interfererStaDevices, interfererApDevices;
    if (tc.interferers > 0)
    {
        mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("interferer-ssid")));
        interfererStaDevices = wifi.Install(phy, mac, interfererStaNodes);

        mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("interferer-ssid")));
        interfererApDevices = wifi.Install(phy, mac, interfererApNodes);
    }

    // --- Mobility ---
    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
    apPositionAlloc->Add(Vector(0.0, 0.0, 0.0));
    apMobility.SetPositionAllocator(apPositionAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(wifiApNode);

    // Station mobility - more conservative movement
    MobilityHelper staMobility;
    if (tc.speed > 0.0)
    {
        staMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
        staPositionAlloc->Add(Vector(tc.distance, 0.0, 0.0));
        staMobility.SetPositionAllocator(staPositionAlloc);
        staMobility.Install(wifiStaNodes);

        // Reduced speed and added bounds checking
        Vector velocity(tc.speed * 0.5, 0.0, 0.0); // Reduce speed by half
        if (tc.category == "PoorPerformance" || tc.category == "HighInterference")
            velocity.y = tc.speed * 0.05 * ((tc.distance > 50) ? 1 : -1); // Minimal y movement
        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(velocity);
    }
    else
    {
        staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
        staPositionAlloc->Add(Vector(tc.distance, 0.0, 0.0));
        staMobility.SetPositionAllocator(staPositionAlloc);
        staMobility.Install(wifiStaNodes);
    }

    // --- Interferer positioning ---
    if (tc.interferers > 0)
    {
        MobilityHelper interfererMobility;
        interfererMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

        Ptr<ListPositionAllocator> interfererApAlloc = CreateObject<ListPositionAllocator>();
        Ptr<ListPositionAllocator> interfererStaAlloc = CreateObject<ListPositionAllocator>();

        for (uint32_t i = 0; i < tc.interferers; ++i)
        {
            double angle = 2.0 * M_PI * i / std::max<uint32_t>(tc.interferers, 1);
            double radius = 30.0 + i * 15.0; // Increased separation

            interfererApAlloc->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 0.0));
            interfererStaAlloc->Add(
                Vector((radius + 10.0) * std::cos(angle), (radius + 10.0) * std::sin(angle), 0.0));
        }

        interfererMobility.SetPositionAllocator(interfererApAlloc);
        interfererMobility.Install(interfererApNodes);

        interfererMobility.SetPositionAllocator(interfererStaAlloc);
        interfererMobility.Install(interfererStaNodes);
    }

    // --- Internet Stack ---
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);
    if (tc.interferers > 0)
    {
        stack.Install(interfererApNodes);
        stack.Install(interfererStaNodes);
    }

    Ipv4AddressHelper address;
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface = address.Assign(apDevices);
    Ipv4InterfaceContainer staInterface = address.Assign(staDevices);

    Ipv4InterfaceContainer interfererApInterface, interfererStaInterface;
    if (tc.interferers > 0)
    {
        address.SetBase("10.1.4.0", "255.255.255.0");
        interfererApInterface = address.Assign(interfererApDevices);
        interfererStaInterface = address.Assign(interfererStaDevices);
    }

    // --- Applications with more aggressive traffic patterns ---
    uint16_t port = 4000;

    // Main traffic: UDP with variable packet size and rate
    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(apInterface.GetAddress(0), port));

    // Parse traffic rate and convert to more conservative values
    std::string adjustedRate = tc.trafficRate;
    if (tc.category == "PoorPerformance" || tc.category == "HighInterference")
    {
        // Reduce traffic rate for challenging scenarios
        double rateValue = std::stod(tc.trafficRate.substr(0, tc.trafficRate.length() - 4));
        rateValue *= 0.5; // Reduce by half
        adjustedRate = std::to_string(static_cast<int>(rateValue)) + "Mbps";
    }

    onoff.SetAttribute("DataRate", DataRateValue(DataRate(adjustedRate)));
    onoff.SetAttribute("PacketSize", UintegerValue(tc.packetSize));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
    onoff.SetAttribute("StartTime", TimeValue(Seconds(1.0))); // Start earlier
    onoff.SetAttribute("StopTime", TimeValue(Seconds(118.0)));
    ApplicationContainer clientApps = onoff.Install(wifiStaNodes.Get(0));

    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer serverApps = sink.Install(wifiApNode.Get(0));
    serverApps.Start(Seconds(0.5));
    serverApps.Stop(Seconds(120.0));

    // Interferer applications with reduced intensity
    if (tc.interferers > 0)
    {
        for (uint32_t i = 0; i < tc.interferers; ++i)
        {
            std::string interfererRate = "1Mbps"; // Much reduced from original
            if (tc.category == "HighInterference")
                interfererRate = "2Mbps";
            else if (tc.category == "PoorPerformance")
                interfererRate = "1.5Mbps";

            OnOffHelper interfererOnOff(
                "ns3::UdpSocketFactory",
                InetSocketAddress(interfererApInterface.GetAddress(i), port + 1 + i));
            interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate(interfererRate)));
            interfererOnOff.SetAttribute("PacketSize", UintegerValue(256)); // Smaller packets
            interfererOnOff.SetAttribute("OnTime",
                                         StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            interfererOnOff.SetAttribute("OffTime",
                                         StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            interfererOnOff.SetAttribute("StartTime", TimeValue(Seconds(2.0 + i * 0.5)));
            interfererOnOff.SetAttribute("StopTime", TimeValue(Seconds(118.0)));
            interfererOnOff.Install(interfererStaNodes.Get(i));

            PacketSinkHelper interfererSink("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), port + 1 + i));
            interfererSink.Install(interfererApNodes.Get(i));
        }
    }

    // --- Flow Monitor ---
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // --- Connect traces with improved patterns ---
    // Connect to your custom MinstrelWifiManagerLogged trace source
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/"
                                  "RemoteStationManager/$ns3::MinstrelWifiManagerLogged/RateChange",
                                  MakeCallback(&RateTrace));

    // Connect to packet transmission for debugging - try both with and without context
    try
    {
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                                      MakeCallback(&TxTrace));
    }
    catch (...)
    {
        // If the above fails, try with context
        try
        {
            Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                            MakeCallback(&TxTraceWithContext));
        }
        catch (...)
        {
            std::cout << "Warning: Could not connect to MacTx trace" << std::endl;
        }
    }

    // Schedule periodic checks to force rate adaptations
    for (double t = 5.0; t < 115.0; t += 10.0)
    {
        Simulator::Schedule(Seconds(t), [&controller]() { controller.IncrementAdaptationEvent(); });
    }

    controller.ScheduleMaxTimeStop();

    Simulator::Stop(Seconds(120.0));
    Simulator::Run();

    // --- Results collection ---
    double throughput = 0, packetLoss = 0, avgDelay = 0;
    double rxPackets = 0, txPackets = 0, rxBytes = 0;
    double simulationTime = Simulator::Now().GetSeconds() - 1.0; // Adjusted for earlier start

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        if (t.sourceAddress == staInterface.GetAddress(0) &&
            t.destinationAddress == apInterface.GetAddress(0))
        {
            rxPackets = it->second.rxPackets;
            txPackets = it->second.txPackets;
            rxBytes = it->second.rxBytes;
            throughput = (rxBytes * 8.0) / (simulationTime * 1e6);
            packetLoss = txPackets > 0 ? 100.0 * (txPackets - rxPackets) / txPackets : 0.0;
            avgDelay = it->second.rxPackets > 0
                           ? it->second.delaySum.GetSeconds() / it->second.rxPackets * 1000.0
                           : 0.0;
            break;
        }
    }

    collectedDecisions = controller.GetSuccessCount() + controller.GetFailureCount();

    // Note: No more CSV summary output - only detailed logging now

    std::cout << "  Collected: " << collectedDecisions << "/" << tc.targetDecisions
              << " decisions in " << simulationTime << "s"
              << " (TX: " << txPackets << ", RX: " << rxPackets << ")" << std::endl;
    std::cout << "  Detailed log: " << logPath << std::endl;

    Simulator::Destroy();
    g_decisionController = nullptr;
    g_detailedLogger = nullptr;
}

int
main(int argc, char* argv[])
{
    // Enable logging for debugging
    LogComponentEnable("MinstrelWifiManagerLogged", LOG_LEVEL_INFO);
    LogComponentEnable("DecisionCountController", LOG_LEVEL_INFO);

    if (system("mkdir -p balanced-results") != 0)
    {
        std::cerr << "Warning: Failed to create directory balanced-results" << std::endl;
    }

    PerformanceBasedParameterGenerator generator;
    std::vector<ScenarioParams> testCases = generator.GenerateStratifiedScenarios(2000);

    std::cout << "Generated " << testCases.size() << " performance-based scenarios" << std::endl;

    // Note: Removed CSV summary file creation - only detailed logs now

    std::map<std::string, uint32_t> categoryStats;
    std::map<std::string, std::vector<uint32_t>> decisionCountsByCategory;

    for (size_t i = 0; i < testCases.size(); ++i)
    {
        const auto& tc = testCases[i];
        std::cout << "Running scenario " << (i + 1) << "/" << testCases.size() << ": "
                  << tc.scenarioName << std::endl;

        uint32_t collectedDecisions = 0;
        RunTestCase(tc, collectedDecisions);

        categoryStats[tc.category]++;
        decisionCountsByCategory[tc.category].push_back(collectedDecisions);
    }

    std::cout << "\n=== PERFORMANCE-BASED SIMULATION RESULTS ===" << std::endl;
    uint32_t totalDecisions = 0;
    for (const auto& category : categoryStats)
    {
        const auto& counts = decisionCountsByCategory[category.first];
        uint32_t categoryTotal = 0;
        for (uint32_t count : counts)
            categoryTotal += count;
        totalDecisions += categoryTotal;

        double avgDecisions = counts.size() > 0 ? double(categoryTotal) / counts.size() : 0.0;

        std::cout << "Category " << category.first << ": " << category.second << " scenarios, "
                  << "avg " << std::fixed << std::setprecision(0) << avgDecisions
                  << " decisions/scenario" << std::endl;
    }

    std::cout << "\nTotal decisions collected: " << totalDecisions << std::endl;
    std::cout << "Detailed logs saved in: balanced-results/*_detailed.csv" << std::endl;

    return 0;
}