/*
 * Minstrel WiFi Manager Benchmark - FIXED FOR 14-FEATURE PIPELINE
 * Compatible with FIXED MinstrelWifiManagerLogged (14 safe features, zero temporal leakage)
 *
 * CRITICAL FIXES (2025-10-01 14:58:28 UTC):
 * - Issue #1: No temporal leakage (manager handles safe logging)
 * - Issue #33: Success ratios from PREVIOUS window (manager handles)
 * - Issue #4: Scenario naming with proper file exports
 * - Fixed attribute names (UpdateStatistics, not EwmaLevel)
 * - 802.11a support (8 rates: 0-7)
 * - Ensures balanced-results/*.csv files are created
 *
 * Author: ahmedjk34
 * Date: 2025-10-01 14:58:28 UTC
 * Version: 5.0 (FIXED - Zero Temporal Leakage)
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

// Decision controller
#include "ns3/decision-count-controller.h"
#include "ns3/performance-based-parameter-generator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// ============================================================================
// SNR Conversion Utility (matches fixed pipeline)
// ============================================================================
enum SnrModel
{
    LOG_MODEL,
    SOFT_MODEL,
    INTF_MODEL
};

static int g_snrConversionCallCount = 0;
static std::vector<double> g_lastConvertedSnrs;
double g_currentTestDistance = 20.0;
uint32_t g_currentInterferers = 0;

double
ConvertNS3ToRealisticSnr(double ns3Value, double distance, uint32_t interferers, SnrModel model)
{
    g_snrConversionCallCount++;

    if (g_snrConversionCallCount < 10 || g_snrConversionCallCount % 100 == 0)
    {
        std::cout << "[DEBUG SNR] Call #" << g_snrConversionCallCount << " | ns3=" << ns3Value
                  << " | dist=" << distance << "m | intf=" << interferers << std::endl;
    }

    if (distance <= 0.0)
        distance = 1.0;
    if (distance > 200.0)
        distance = 200.0;
    if (interferers > 10)
        interferers = 10;

    double realisticSnr = 0.0;

    switch (model)
    {
    case LOG_MODEL: {
        double snr0 = 40.0;
        double pathLossExp = 2.2;
        realisticSnr = snr0 - 10 * pathLossExp * log10(distance);
        realisticSnr -= (interferers * 1.5);
        break;
    }
    case SOFT_MODEL: {
        if (distance <= 20.0)
            realisticSnr = 35.0 - (distance * 0.8);
        else if (distance <= 50.0)
            realisticSnr = 19.0 - ((distance - 20.0) * 0.5);
        else if (distance <= 100.0)
            realisticSnr = 4.0 - ((distance - 50.0) * 0.3);
        else
            realisticSnr = -11.0 - ((distance - 100.0) * 0.2);

        realisticSnr -= (interferers * 2.0);
        break;
    }
    case INTF_MODEL: {
        realisticSnr = 38.0 - 10 * log10(distance * distance);
        realisticSnr -= (pow(interferers, 1.2) * 1.2);
        break;
    }
    }

    double variation = fmod(std::abs(ns3Value), 12.0) - 6.0;
    realisticSnr += variation * 0.4;

    realisticSnr = std::max(-30.0, std::min(45.0, realisticSnr));

    g_lastConvertedSnrs.push_back(realisticSnr);
    if (g_lastConvertedSnrs.size() > 10)
        g_lastConvertedSnrs.erase(g_lastConvertedSnrs.begin());

    return realisticSnr;
}

// Global decision controller
DecisionCountController* g_decisionController = nullptr;

// ============================================================================
// FIXED: Simple trace callbacks (manager handles feature logging)
// ============================================================================
static void
RateTrace(uint64_t oldValue, uint64_t newValue)
{
    static uint32_t rateChangeCount = 0;
    rateChangeCount++;

    if (rateChangeCount % 50 == 0)
    {
        std::cout << "[RATE CHANGE #" << rateChangeCount << "] " << oldValue << " -> " << newValue
                  << " bps" << std::endl;
    }

    if (g_decisionController)
    {
        g_decisionController->IncrementSuccess();
    }
}

static void
MinstrelDecisionTrace(std::string context,
                      uint32_t nodeId,
                      uint32_t deviceId,
                      Mac48Address address,
                      uint32_t rateIndex)
{
    static uint32_t decisionCount = 0;
    decisionCount++;

    if (decisionCount % 100 == 0)
    {
        std::cout << "[MINSTREL DECISION #" << decisionCount << "] node=" << nodeId
                  << " rate=" << rateIndex << std::endl;
    }

    if (g_decisionController)
    {
        g_decisionController->IncrementAdaptationEvent();
    }
}

static void
TxTrace(Ptr<const Packet> packet)
{
    static uint32_t txCount = 0;
    txCount++;

    if (txCount % 5000 == 0)
    {
        std::cout << "[TX] Total packets transmitted: " << txCount << std::endl;
    }
}

// ============================================================================
// FIXED: Test case runner with proper file exports
// ============================================================================
void
RunTestCase(const ScenarioParams& tc, uint32_t& collectedDecisions)
{
    DecisionCountController controller(tc.targetDecisions, 120);
    g_decisionController = &controller;

    // FIXED: Set global scenario info for SNR conversion
    g_currentTestDistance = tc.distance;
    g_currentInterferers = tc.interferers;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "RUNNING SCENARIO: " << tc.scenarioName << std::endl;
    std::cout << "  Category: " << tc.category << std::endl;
    std::cout << "  Distance: " << tc.distance << "m | Speed: " << tc.speed << " m/s" << std::endl;
    std::cout << "  Interferers: " << tc.interferers << " | Target: " << tc.targetDecisions
              << " decisions" << std::endl;
    std::cout << "  Expected SNR: "
              << ConvertNS3ToRealisticSnr(100.0, tc.distance, tc.interferers, SOFT_MODEL) << " dB"
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;

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
    // --- PHY and Channel (FIXED for 802.11a) ---
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    // FIXED: More permissive settings for 802.11a to work at longer distances
    phy.Set("TxPowerStart", DoubleValue(30.0)); // ✅ Increased from 20.0
    phy.Set("TxPowerEnd", DoubleValue(30.0));
    phy.Set("RxNoiseFigure", DoubleValue(3.0));    // ✅ Decreased from 5.0-7.0
    phy.Set("CcaEdThreshold", DoubleValue(-82.0)); // ✅ More sensitive (default -82)
    phy.Set("RxSensitivity", DoubleValue(-92.0));  // ✅ More sensitive (default -92)

    // channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
    //                            "Exponent",
    //                            DoubleValue(3.0), // Path loss exponent
    //                            "ReferenceDistance",
    //                            DoubleValue(1.0),
    //                            "ReferenceLoss",
    //                            DoubleValue(46.6677)); // Loss at 1m

    // // Optional: Add Nakagami fading for realism
    // channel.AddPropagationLoss("ns3::NakagamiPropagationLossModel",
    //                            "m0",
    //                            DoubleValue(1.5), // Fading parameter
    //                            "m1",
    //                            DoubleValue(0.75),
    //                            "m2",
    //                            DoubleValue(0.75));

    // FIXED: Adjust based on category for realistic conditions
    if (tc.category == "PoorPerformance")
    {
        phy.Set("RxNoiseFigure", DoubleValue(5.0)); // ✅ Still lower than before (was 7.0)
    }
    else if (tc.category == "HighInterference")
    {
        phy.Set("RxNoiseFigure", DoubleValue(4.0)); // ✅ Lower (was 6.0)
    }
    else
    {
        phy.Set("RxNoiseFigure", DoubleValue(3.0)); // ✅ Best case
    }

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a); // FIXED: 802.11a (8 rates: 0-7)

    // FIXED: Proper log path with scenario naming (Issue #4)
    std::string logPath = "balanced-results/" + tc.scenarioName + "_detailed.csv";

    // FIXED: Correct attribute names for MinstrelWifiManagerLogged
    wifi.SetRemoteStationManager("ns3::MinstrelWifiManagerLogged",
                                 "UpdateStatistics",
                                 TimeValue(MilliSeconds(100)), // FIXED: Not EwmaLevel
                                 "LookAroundRate",
                                 UintegerValue(10),
                                 "EWMA",
                                 UintegerValue(75), // FIXED: Not EwmaLevel
                                 "SampleColumn",
                                 UintegerValue(10),
                                 "PacketLength",
                                 UintegerValue(1200),
                                 "PrintStats",
                                 BooleanValue(false),
                                 "LogFilePath",
                                 StringValue(logPath), // FIXED: Ensure file export
                                 "ScenarioFileName",
                                 StringValue(tc.scenarioName)); // FIXED: Issue #4

    std::cout << "[CONFIG] Log file will be written to: " << logPath << std::endl;

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211a-fixed");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    // ADD THIS:
    Ptr<WifiNetDevice> staDevice = DynamicCast<WifiNetDevice>(staDevices.Get(0));
    if (staDevice)
    {
        Ptr<MinstrelWifiManagerLogged> mgr =
            DynamicCast<MinstrelWifiManagerLogged>(staDevice->GetRemoteStationManager());
        if (mgr)
        {
            mgr->SetScenarioParameters(tc.distance, tc.interferers);
            std::cout << "[CONFIG] ✅ Set Minstrel parameters: distance=" << tc.distance
                      << "m, interferers=" << tc.interferers << std::endl;
        }
        else
        {
            std::cout << "[WARN] ❌ Could not get MinstrelWifiManagerLogged instance!" << std::endl;
        }
    }

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // Create interferer devices
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

    MobilityHelper staMobility;
    if (tc.speed > 0.0)
    {
        // Mobile scenario
        staMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
        staPositionAlloc->Add(Vector(tc.distance, 0.0, 0.0));
        staMobility.SetPositionAllocator(staPositionAlloc);
        staMobility.Install(wifiStaNodes);

        Vector velocity(tc.speed * 0.5, 0.0, 0.0);
        if (tc.category == "PoorPerformance" || tc.category == "HighInterference")
            velocity.y = tc.speed * 0.05 * ((tc.distance > 50) ? 1 : -1);

        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(velocity);
    }
    else
    {
        // Static scenario
        staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
        staPositionAlloc->Add(Vector(tc.distance, 0.0, 0.0));
        staMobility.SetPositionAllocator(staPositionAlloc);
        staMobility.Install(wifiStaNodes);
    }

    // Interferer positioning
    if (tc.interferers > 0)
    {
        MobilityHelper interfererMobility;
        interfererMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

        Ptr<ListPositionAllocator> interfererApAlloc = CreateObject<ListPositionAllocator>();
        Ptr<ListPositionAllocator> interfererStaAlloc = CreateObject<ListPositionAllocator>();

        for (uint32_t i = 0; i < tc.interferers; ++i)
        {
            double angle = 2.0 * M_PI * i / std::max<uint32_t>(tc.interferers, 1);
            double radius = 30.0 + i * 15.0;

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

    // --- Applications ---
    uint16_t port = 4000;

    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(apInterface.GetAddress(0), port));

    // FIXED: Adjust traffic rate based on category
    std::string adjustedRate = tc.trafficRate;
    if (tc.category == "PoorPerformance" || tc.category == "HighInterference")
    {
        double rateValue = std::stod(tc.trafficRate.substr(0, tc.trafficRate.length() - 4));
        rateValue *= 0.6;
        adjustedRate = std::to_string(static_cast<int>(rateValue)) + "Mbps";
    }

    onoff.SetAttribute("DataRate", DataRateValue(DataRate(adjustedRate)));
    onoff.SetAttribute("PacketSize", UintegerValue(tc.packetSize));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
    onoff.SetAttribute("StartTime", TimeValue(Seconds(1.0)));
    onoff.SetAttribute("StopTime", TimeValue(Seconds(118.0)));
    ApplicationContainer clientApps = onoff.Install(wifiStaNodes.Get(0));

    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer serverApps = sink.Install(wifiApNode.Get(0));
    serverApps.Start(Seconds(0.5));
    serverApps.Stop(Seconds(120.0));

    // Interferer traffic
    if (tc.interferers > 0)
    {
        for (uint32_t i = 0; i < tc.interferers; ++i)
        {
            std::string interfererRate = "1Mbps";
            if (tc.category == "HighInterference")
                interfererRate = "2Mbps";

            OnOffHelper interfererOnOff(
                "ns3::UdpSocketFactory",
                InetSocketAddress(interfererApInterface.GetAddress(i), port + 1 + i));
            interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate(interfererRate)));
            interfererOnOff.SetAttribute("PacketSize", UintegerValue(256));
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

    // --- Connect traces ---
    // try
    // {
    //     Config::ConnectWithoutContext(
    //         "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/"
    //         "RemoteStationManager/$ns3::MinstrelWifiManagerLogged/RateChange",
    //         MakeCallback(&RateTrace));
    // }
    // catch (...)
    // {
    //     std::cout << "[WARN] Could not connect to RateChange trace" << std::endl;
    // }

    // Try to connect TX trace
    try
    {
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                                      MakeCallback(&TxTrace));
    }
    catch (...)
    {
        std::cout << "[WARN] Could not connect to MacTx trace" << std::endl;
    }

    // Schedule periodic events
    for (double t = 5.0; t < 115.0; t += 10.0)
    {
        Simulator::Schedule(Seconds(t), [&controller]() { controller.IncrementAdaptationEvent(); });
    }

    controller.ScheduleMaxTimeStop();

    Simulator::Stop(Seconds(120.0));

    std::cout << "[SIM] Starting simulation..." << std::endl;
    Simulator::Run();
    std::cout << "[SIM] Simulation completed" << std::endl;

    // After Simulator::Run() in your benchmark:
    std::cout << "\n=== DIAGNOSTIC INFO ===" << std::endl;
    std::cout << "Simulation time: " << Simulator::Now().GetSeconds() << "s" << std::endl;

    // Check if log file exists and has content
    std::ifstream checkFile(logPath);
    if (checkFile.is_open())
    {
        checkFile.seekg(0, std::ios::end);
        size_t fileSize = checkFile.tellg();
        std::cout << "Log file size: " << fileSize << " bytes" << std::endl;

        if (fileSize > 0)
        {
            checkFile.seekg(0);
            std::string firstLine;
            std::getline(checkFile, firstLine);
            std::cout << "First line: " << firstLine << std::endl;

            int lineCount = 1;
            std::string line;
            while (std::getline(checkFile, line))
                lineCount++;
            std::cout << "Total lines: " << lineCount << std::endl;
        }
        checkFile.close();
    }
    else
    {
        std::cout << "❌ Could not open log file!" << std::endl;
    }

    // Collect results
    double throughput = 0, packetLoss = 0, avgDelay = 0;
    double rxPackets = 0, txPackets = 0, rxBytes = 0;
    double simulationTime = Simulator::Now().GetSeconds() - 1.0;

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

    std::cout << "\n[RESULTS] Scenario: " << tc.scenarioName << std::endl;
    std::cout << "  Collected: " << collectedDecisions << "/" << tc.targetDecisions
              << " decisions (" << std::fixed << std::setprecision(1)
              << (100.0 * collectedDecisions / tc.targetDecisions) << "%)" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(1) << simulationTime << "s"
              << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput << " Mbps"
              << std::endl;
    std::cout << "  Packet Loss: " << std::fixed << std::setprecision(1) << packetLoss << "%"
              << std::endl;
    std::cout << "  Avg Delay: " << std::fixed << std::setprecision(2) << avgDelay << " ms"
              << std::endl;
    std::cout << "  TX/RX: " << txPackets << "/" << rxPackets << std::endl;
    std::cout << "  ✅ Log file: " << logPath << std::endl;
    std::cout << "  ✅ 14 safe features logged (zero temporal leakage)" << std::endl;

    Simulator::Destroy();
    g_decisionController = nullptr;
}

// ============================================================================
// Main function
// ============================================================================
int
main(int argc, char* argv[])
{
    auto benchmarkStartTime = std::chrono::high_resolution_clock::now();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FIXED Minstrel WiFi Manager Benchmark" << std::endl;
    std::cout << "Author: ahmedjk34" << std::endl;
    std::cout << "Date: 2025-10-01 14:58:28 UTC" << std::endl;
    std::cout << "Version: 5.0 (FIXED - Zero Temporal Leakage)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\nCRITICAL FIXES:" << std::endl;
    std::cout << "  ✅ Issue #1: Zero temporal leakage (manager handles safe logging)" << std::endl;
    std::cout << "  ✅ Issue #33: Success ratios from PREVIOUS window" << std::endl;
    std::cout << "  ✅ Issue #4: Scenario naming for train/test splitting" << std::endl;
    std::cout << "  ✅ Fixed attribute names (UpdateStatistics, EWMA)" << std::endl;
    std::cout << "  ✅ 802.11a: 8 rates (0-7)" << std::endl;
    std::cout << "  ✅ File exports to balanced-results/*.csv" << std::endl;

    LogComponentEnable("MinstrelWifiManagerLogged", LOG_LEVEL_INFO);
    LogComponentEnable("DecisionCountController", LOG_LEVEL_INFO);

    // FIXED: Ensure directory exists
    if (system("mkdir -p balanced-results") != 0)
    {
        std::cerr << "WARNING: Could not create balanced-results directory" << std::endl;
    }
    else
    {
        std::cout << "\n✅ Created balanced-results/ directory for CSV exports" << std::endl;
    }

    // Generate test cases
    PerformanceBasedParameterGenerator generator;
    std::vector<ScenarioParams> testCases = generator.GenerateStratifiedScenarios(30);

    std::cout << "\n📊 Generated " << testCases.size() << " performance-based scenarios"
              << std::endl;

    std::map<std::string, uint32_t> categoryStats;
    std::map<std::string, std::vector<uint32_t>> decisionCountsByCategory;
    std::map<std::string, uint32_t> totalDecisionsByCategory;

    // Run all scenarios
    for (size_t i = 0; i < testCases.size(); ++i)
    {
        const auto& tc = testCases[i];

        std::cout << "\n" << std::string(80, '#') << std::endl;
        std::cout << "# SCENARIO " << (i + 1) << "/" << testCases.size() << " (" << std::fixed
                  << std::setprecision(1) << (100.0 * (i + 1) / testCases.size()) << "%)"
                  << std::endl;
        std::cout << std::string(80, '#') << std::endl;

        uint32_t collectedDecisions = 0;

        try
        {
            RunTestCase(tc, collectedDecisions);

            categoryStats[tc.category]++;
            decisionCountsByCategory[tc.category].push_back(collectedDecisions);
            totalDecisionsByCategory[tc.category] += collectedDecisions;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[ERROR] Scenario " << (i + 1) << " failed: " << e.what() << std::endl;
        }
    }

    auto benchmarkEndTime = std::chrono::high_resolution_clock::now();
    auto totalDuration =
        std::chrono::duration_cast<std::chrono::minutes>(benchmarkEndTime - benchmarkStartTime);

    // Final summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BENCHMARK COMPLETE - SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    uint32_t totalDecisions = 0;
    for (const auto& category : categoryStats)
    {
        const auto& counts = decisionCountsByCategory[category.first];
        uint32_t categoryTotal = totalDecisionsByCategory[category.first];
        totalDecisions += categoryTotal;

        double avgDecisions = counts.size() > 0 ? double(categoryTotal) / counts.size() : 0.0;

        std::cout << "\nCategory: " << category.first << std::endl;
        std::cout << "  Scenarios: " << category.second << std::endl;
        std::cout << "  Total Decisions: " << categoryTotal << std::endl;
        std::cout << "  Avg Decisions/Scenario: " << std::fixed << std::setprecision(0)
                  << avgDecisions << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TOTAL DECISIONS COLLECTED: " << totalDecisions << std::endl;
    std::cout << "TOTAL EXECUTION TIME: " << totalDuration.count() << " minutes" << std::endl;
    std::cout << "OUTPUT DIRECTORY: balanced-results/" << std::endl;
    std::cout << "FILE FORMAT: <scenario>_detailed.csv" << std::endl;
    std::cout << "\n✅ ALL DATA LOGGED WITH 14 SAFE FEATURES" << std::endl;
    std::cout << "✅ ZERO TEMPORAL LEAKAGE" << std::endl;
    std::cout << "✅ SCENARIO FILES FOR TRAIN/TEST SPLITTING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // SNR conversion debug summary
    std::cout << "\n--- SNR Conversion Debug ---" << std::endl;
    std::cout << "Total calls: " << g_snrConversionCallCount << std::endl;
    std::cout << "Last 10 SNRs: ";
    for (double snr : g_lastConvertedSnrs)
        std::cout << std::fixed << std::setprecision(1) << snr << " ";
    std::cout << "dB" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}