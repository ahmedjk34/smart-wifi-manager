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

// Match SmartWifiManagerV3Logged trace signature
// expected=CallbackImpl<void, std::string, unsigned long, unsigned long>
static void
RateTrace(std::string context, unsigned long newRate, unsigned long oldRate)
{
    std::cout << "Rate adaptation event: context=" << context << " newRate=" << newRate
              << " oldRate=" << oldRate << std::endl;

    // Count each rate-change as a "decision" event
    if (g_decisionController)
    {
        g_decisionController->IncrementSuccess();
    }
}

void
RunTestCase(const ScenarioParams& tc, std::ofstream& csv, uint32_t& collectedDecisions)
{
    // Reset decision controller for this scenario
    DecisionCountController controller(tc.targetDecisions, 120); // 2 min max
    g_decisionController = &controller;

    // Per-scenario rich log path (SmartWifiManagerV3Logged writes here)
    std::string logPath = "balanced-results/" + tc.scenarioName + ".csv";
    controller.SetLogFilePath(logPath);

    std::cout << "  Target: " << tc.targetDecisions << " decisions (" << tc.category << ")"
              << std::endl;

    // --- Simulation Setup ---
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    NodeContainer interfererApNodes;
    NodeContainer interfererStaNodes;
    interfererApNodes.Create(tc.interferers);
    interfererStaNodes.Create(tc.interferers);

    // Channel and PHY with improved realism
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    // Slightly adjust noise figure based on scenario category
    if (tc.category == "PoorPerformance")
    {
        phy.Set("RxNoiseFigure", DoubleValue(10.0));
    }
    else
    {
        phy.Set("RxNoiseFigure", DoubleValue(7.0));
    }

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);

    // Use SmartWifiManagerV3Logged as in File 1 to keep tracing and per-scenario CSV format
    wifi.SetRemoteStationManager("ns3::SmartWifiManagerV3Logged",
                                 "LogFilePath",
                                 StringValue(logPath));

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    // Main STA and AP
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // Interferer STA/AP devices (use same manager so all events log to scenario file)
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer interfererStaDevices = wifi.Install(phy, mac, interfererStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer interfererApDevices = wifi.Install(phy, mac, interfererApNodes);

    // Mobility for AP
    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
    apPositionAlloc->Add(Vector(0.0, 0.0, 0.0));
    apMobility.SetPositionAllocator(apPositionAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(wifiApNode);

    // Mobility for STA
    if (tc.speed > 0.0)
    {
        MobilityHelper mobMove;
        mobMove.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> movingAlloc = CreateObject<ListPositionAllocator>();
        movingAlloc->Add(Vector(tc.distance, 0.0, 0.0));
        mobMove.SetPositionAllocator(movingAlloc);
        mobMove.Install(wifiStaNodes);

        // Introduce slight lateral motion for some challenging categories (optional)
        Vector velocity(tc.speed, 0.0, 0.0);
        if (tc.category == "PoorPerformance" || tc.category == "HighInterference")
        {
            velocity.y = tc.speed * 0.1 * ((tc.distance > 50) ? 1 : -1);
        }
        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(velocity);
    }
    else
    {
        MobilityHelper mobStill;
        mobStill.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        Ptr<ListPositionAllocator> stillAlloc = CreateObject<ListPositionAllocator>();
        stillAlloc->Add(Vector(tc.distance, 0.0, 0.0));
        mobStill.SetPositionAllocator(stillAlloc);
        mobStill.Install(wifiStaNodes);
    }

    // Interferers positioning (radial spread for diversity)
    if (tc.interferers > 0)
    {
        MobilityHelper interfererMobility;
        interfererMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

        Ptr<ListPositionAllocator> interfererApAlloc = CreateObject<ListPositionAllocator>();
        Ptr<ListPositionAllocator> interfererStaAlloc = CreateObject<ListPositionAllocator>();

        for (uint32_t i = 0; i < tc.interferers; ++i)
        {
            double angle = 2.0 * M_PI * i / std::max<uint32_t>(tc.interferers, 1);
            double radius = 20.0 + i * 10.0;

            interfererApAlloc->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 0.0));
            interfererStaAlloc->Add(
                Vector((radius + 5.0) * std::cos(angle), (radius + 5.0) * std::sin(angle), 0.0));
        }

        interfererMobility.SetPositionAllocator(interfererApAlloc);
        interfererMobility.Install(interfererApNodes);

        interfererMobility.SetPositionAllocator(interfererStaAlloc);
        interfererMobility.Install(interfererStaNodes);
    }

    // Internet stack
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);
    if (tc.interferers > 0)
    {
        stack.Install(interfererApNodes);
        stack.Install(interfererStaNodes);
    }

    // IP addressing
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

    // Applications
    uint16_t port = 4000;
    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(apInterface.GetAddress(0), port));
    onoff.SetAttribute("DataRate", DataRateValue(DataRate(tc.trafficRate)));
    onoff.SetAttribute("PacketSize", UintegerValue(tc.packetSize));
    onoff.SetAttribute("StartTime", TimeValue(Seconds(2.0)));
    onoff.SetAttribute("StopTime", TimeValue(Seconds(118.0)));
    ApplicationContainer clientApps = onoff.Install(wifiStaNodes.Get(0));

    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer serverApps = sink.Install(wifiApNode.Get(0));
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(120.0));

    // Interferer traffic with category-based intensity
    if (tc.interferers > 0)
    {
        for (uint32_t i = 0; i < tc.interferers; ++i)
        {
            std::string interfererRate = "2Mbps";
            if (tc.category == "HighInterference")
            {
                interfererRate = "5Mbps";
            }
            else if (tc.category == "PoorPerformance")
            {
                interfererRate = "3Mbps";
            }

            OnOffHelper interfererOnOff(
                "ns3::UdpSocketFactory",
                InetSocketAddress(interfererApInterface.GetAddress(i), port + 1 + i));
            interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate(interfererRate)));
            interfererOnOff.SetAttribute("PacketSize", UintegerValue(512));
            interfererOnOff.SetAttribute("StartTime", TimeValue(Seconds(2.0 + i * 0.5)));
            interfererOnOff.SetAttribute("StopTime", TimeValue(Seconds(118.0)));
            interfererOnOff.Install(interfererStaNodes.Get(i));

            PacketSinkHelper interfererSink("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), port + 1 + i));
            interfererSink.Install(interfererApNodes.Get(i));
        }
    }

    // Flow Monitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Tracing: keep File 1 style - connect to SmartWifiManagerV3Logged RateChange
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                    "$ns3::SmartWifiManagerV3Logged/RateChange",
                    MakeCallback(&RateTrace));

    // Early termination safety
    controller.ScheduleMaxTimeStop();

    Simulator::Stop(Seconds(120.0));
    Simulator::Run();

    // Results calculation (aggregate, keep File 1 CSV format)
    double throughput = 0;
    double packetLoss = 0;
    double avgDelay = 0;
    double rxPackets = 0, txPackets = 0;
    double rxBytes = 0;
    double simulationTime = Simulator::Now().GetSeconds() - 2.0;

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
        }
    }

    // Capture decision count consistent with File 1
    collectedDecisions = controller.GetSuccessCount() + controller.GetFailureCount();

    // Append to main CSV (exact File 1 format and order)
    csv << "\"" << tc.scenarioName << "\"," << tc.category << "," << tc.distance << "," << tc.speed
        << "," << tc.interferers << "," << tc.packetSize << "," << tc.trafficRate << ","
        << tc.targetSnrMin << "," << tc.targetSnrMax << "," << tc.targetDecisions << ","
        << collectedDecisions << "," << simulationTime << "," << throughput << "," << packetLoss
        << "," << avgDelay << "," << rxPackets << "," << txPackets << "\n";

    std::cout << "  Collected: " << collectedDecisions << "/" << tc.targetDecisions
              << " decisions in " << simulationTime << "s" << std::endl;

    Simulator::Destroy();
    g_decisionController = nullptr;
}

int
main(int argc, char* argv[])
{
    // Ensure output directory exists
    system("mkdir -p balanced-results");

    // Use the performance-based generator
    PerformanceBasedParameterGenerator generator;
    std::vector<ScenarioParams> testCases = generator.GenerateStratifiedScenarios(800);

    std::cout << "Generated " << testCases.size() << " performance-based scenarios" << std::endl;

    // Main CSV with File 1 header/format
    std::ofstream csv("smartv3-benchmark-performance-based.csv");
    csv << "Scenario,Category,Distance,Speed,Interferers,PacketSize,TrafficRate,"
           "TargetSnrMin,TargetSnrMax,TargetDecisions,CollectedDecisions,SimTime(s),"
           "Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

    // Statistics tracking
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

    // Final performance report (as in File 1)
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
    std::cout << "Results in: smartv3-benchmark-performance-based.csv" << std::endl;

    return 0;
}