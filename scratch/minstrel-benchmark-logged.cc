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
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// Global decision controller
DecisionCountController* g_decisionController = nullptr;

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
}

// Fixed packet transmission callback - removed context parameter
static void
TxTrace(Ptr<const Packet> packet)
{
    static uint32_t txCount = 0;
    txCount++;
    if (txCount % 100 == 0) // Log every 100th packet
    {
        std::cout << "TX packets: " << txCount << std::endl;
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
}

void
RunTestCase(const ScenarioParams& tc, std::ofstream& csv, uint32_t& collectedDecisions)
{
    DecisionCountController controller(tc.targetDecisions, 120); // 2 min max
    g_decisionController = &controller;

    std::string logPath = "balanced-results/" + tc.scenarioName + ".csv";
    controller.SetLogFilePath(logPath);

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

    csv << "\"" << tc.scenarioName << "\"," << tc.category << "," << tc.distance << "," << tc.speed
        << "," << tc.interferers << "," << tc.packetSize << "," << adjustedRate << ","
        << tc.targetSnrMin << "," << tc.targetSnrMax << "," << tc.targetDecisions << ","
        << collectedDecisions << "," << simulationTime << "," << throughput << "," << packetLoss
        << "," << avgDelay << "," << rxPackets << "," << txPackets << "\n";

    std::cout << "  Collected: " << collectedDecisions << "/" << tc.targetDecisions
              << " decisions in " << simulationTime << "s"
              << " (TX: " << txPackets << ", RX: " << rxPackets << ")" << std::endl;

    Simulator::Destroy();
    g_decisionController = nullptr;
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
    std::vector<ScenarioParams> testCases = generator.GenerateStratifiedScenarios(800);

    std::cout << "Generated " << testCases.size() << " performance-based scenarios" << std::endl;

    std::ofstream csv("minstrel-benchmark-performance-based.csv");
    csv << "Scenario,Category,Distance,Speed,Interferers,PacketSize,TrafficRate,"
           "TargetSnrMin,TargetSnrMax,TargetDecisions,CollectedDecisions,SimTime(s),"
           "Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

    std::map<std::string, uint32_t> categoryStats;
    std::map<std::string, std::vector<uint32_t>> decisionCountsByCategory;

    for (size_t i = 0; i < testCases.size(); ++i)
    {
        const auto& tc = testCases[i];
        std::cout << "Running scenario " << (i + 1) << "/" << testCases.size() << ": "
                  << tc.scenarioName << std::endl;

        uint32_t collectedDecisions = 0;
        RunTestCase(tc, csv, collectedDecisions);

        categoryStats[tc.category]++;
        decisionCountsByCategory[tc.category].push_back(collectedDecisions);
    }

    csv.close();

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
    std::cout << "Results in: minstrel-benchmark-performance-based.csv" << std::endl;

    return 0;
}