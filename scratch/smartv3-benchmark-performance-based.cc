#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

// *** INCLUDE THE 3 NEW FILES HERE ***
#include "ns3/performance-based-parameter-generator.h"
#include "ns3/decision-count-controller.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace ns3;

// Global decision controller
DecisionCountController* g_decisionController = nullptr;

void RateTrace(std::string context, uint64_t rate, uint64_t oldRate)
{
    std::cout << "Rate adaptation event: context=" << context
              << " new datarate=" << rate << " old datarate=" << oldRate << std::endl;
    
    // *** COUNT DECISIONS ***
    if (g_decisionController) {
        g_decisionController->IncrementDecisionCount();
    }
}

void RunTestCase(const ScenarioParams& tc, std::ofstream& csv, uint32_t& collectedDecisions)  // *** FIXED: Added output parameter ***
{
    // Reset decision controller for this scenario
    DecisionCountController controller(tc.targetDecisions, 120); // 2 min max
    g_decisionController = &controller;
    
    std::string logPath = "balanced-results/" + tc.scenarioName + ".csv";
    controller.SetLogFilePath(logPath);
    
    std::cout << "  Target: " << tc.targetDecisions << " decisions (" << tc.category << ")" << std::endl;

    // *** REST OF YOUR ORIGINAL CODE WITH PARAMETER CHANGES ***
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    NodeContainer interfererApNodes;
    NodeContainer interfererStaNodes;
    interfererApNodes.Create(tc.interferers);  // *** CHANGED FROM tc.numInterferers ***
    interfererStaNodes.Create(tc.interferers);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::SmartWifiManagerV3Logged", 
        "LogFilePath", StringValue(logPath));  // *** USE PERFORMANCE-BASED LOG PATH ***
        
    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer interfererStaDevices = wifi.Install(phy, mac, interfererStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer interfererApDevices = wifi.Install(phy, mac, interfererApNodes);

    // Mobility - same logic but using performance-calculated distance
    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
    apPositionAlloc->Add(Vector(0.0, 0.0, 0.0));
    apMobility.SetPositionAllocator(apPositionAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(wifiApNode);

    if (tc.speed > 0.0)  // *** CHANGED FROM tc.staSpeed ***
    {
        MobilityHelper mobMove;
        mobMove.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> movingAlloc = CreateObject<ListPositionAllocator>();
        movingAlloc->Add(Vector(tc.distance, 0.0, 0.0)); // *** CHANGED FROM tc.staDistance ***
        mobMove.SetPositionAllocator(movingAlloc);
        mobMove.Install(wifiStaNodes);
        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(Vector(tc.speed, 0.0, 0.0));
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

    // Interferers
    MobilityHelper interfererMobility;
    Ptr<ListPositionAllocator> interfererApAlloc = CreateObject<ListPositionAllocator>();
    Ptr<ListPositionAllocator> interfererStaAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < tc.interferers; ++i)
    {
        interfererApAlloc->Add(Vector(50.0 + 40*i, 50.0, 0.0));
        interfererStaAlloc->Add(Vector(50.0 + 40*i, 55.0, 0.0));
    }
    interfererMobility.SetPositionAllocator(interfererApAlloc);
    interfererMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    interfererMobility.Install(interfererApNodes);
    interfererMobility.SetPositionAllocator(interfererStaAlloc);
    interfererMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    interfererMobility.Install(interfererStaNodes);

    // Internet stack
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);
    stack.Install(interfererApNodes);
    stack.Install(interfererStaNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface = address.Assign(apDevices);
    Ipv4InterfaceContainer staInterface = address.Assign(staDevices);

    address.SetBase("10.1.4.0", "255.255.255.0");
    Ipv4InterfaceContainer interfererApInterface = address.Assign(interfererApDevices);
    Ipv4InterfaceContainer interfererStaInterface = address.Assign(interfererStaDevices);

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

    // Interferer traffic
    for (uint32_t i = 0; i < tc.interferers; ++i)
    {
        OnOffHelper interfererOnOff("ns3::UdpSocketFactory", InetSocketAddress(interfererApInterface.GetAddress(i), port+1));
        interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate("2Mbps")));
        interfererOnOff.SetAttribute("PacketSize", UintegerValue(512));
        interfererOnOff.SetAttribute("StartTime", TimeValue(Seconds(2.0)));
        interfererOnOff.SetAttribute("StopTime", TimeValue(Seconds(118.0)));
        interfererOnOff.Install(interfererStaNodes.Get(i));

        PacketSinkHelper interfererSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port+1));
        interfererSink.Install(interfererApNodes.Get(i));
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/Rate",
                MakeCallback(&RateTrace));

    // *** SCHEDULE EARLY TERMINATION ***
    controller.ScheduleMaxTimeStop();

    Simulator::Stop(Seconds(120.0));
    Simulator::Run();

    // Results calculation (same as your original)
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
        if (t.sourceAddress == staInterface.GetAddress(0) && t.destinationAddress == apInterface.GetAddress(0))
        {
            rxPackets = it->second.rxPackets;
            txPackets = it->second.txPackets;
            rxBytes = it->second.rxBytes;
            throughput = (rxBytes * 8.0) / (simulationTime * 1e6);
            packetLoss = txPackets > 0 ? 100.0 * (txPackets - rxPackets) / txPackets : 0.0;
            avgDelay = it->second.rxPackets > 0 ? it->second.delaySum.GetSeconds() / it->second.rxPackets * 1000.0 : 0.0;
        }
    }

    // *** CAPTURE DECISION COUNT BEFORE NULLIFYING POINTER ***
    collectedDecisions = controller.GetDecisionCount();

    // *** ENHANCED CSV OUTPUT ***
    csv << "\"" << tc.scenarioName << "\","
        << tc.category << ","
        << tc.distance << ","
        << tc.speed << ","
        << tc.interferers << ","
        << tc.packetSize << ","
        << tc.trafficRate << ","
        << tc.targetSnrMin << ","
        << tc.targetSnrMax << ","
        << tc.targetDecisions << ","
        << collectedDecisions << ","  // *** USE CAPTURED VALUE ***
        << simulationTime << ","
        << throughput << ","
        << packetLoss << ","
        << avgDelay << ","
        << rxPackets << ","
        << txPackets << "\n";

    std::cout << "  Collected: " << collectedDecisions << "/" << tc.targetDecisions 
              << " decisions in " << simulationTime << "s" << std::endl;

    Simulator::Destroy();
    g_decisionController = nullptr;
}

int main(int argc, char *argv[])
{
    // *** CREATE OUTPUT DIRECTORY ***
    system("mkdir -p balanced-results");
    
    // *** USE THE PERFORMANCE-BASED GENERATOR ***
    PerformanceBasedParameterGenerator generator;
    std::vector<ScenarioParams> testCases = generator.GenerateStratifiedScenarios(200);

    std::cout << "Generated " << testCases.size() << " performance-based scenarios" << std::endl;

    // *** ENHANCED CSV HEADER ***
    std::ofstream csv("smartv3-benchmark-performance-based.csv");
    csv << "Scenario,Category,Distance,Speed,Interferers,PacketSize,TrafficRate,"
           "TargetSnrMin,TargetSnrMax,TargetDecisions,CollectedDecisions,SimTime(s),"
           "Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

    // *** STATISTICS TRACKING ***
    std::map<std::string, uint32_t> categoryStats;
    std::map<std::string, std::vector<uint32_t>> decisionCountsByCategory;

    for (size_t i = 0; i < testCases.size(); ++i)
    {
        const auto& tc = testCases[i];
        std::cout << "Running scenario " << (i + 1) << "/" << testCases.size() 
                  << ": " << tc.scenarioName << std::endl;
        
        // *** FIXED: Properly capture decision count ***
        uint32_t collectedDecisions = 0;
        RunTestCase(tc, csv, collectedDecisions);
        
        // Track statistics using the captured value
        categoryStats[tc.category]++;
        decisionCountsByCategory[tc.category].push_back(collectedDecisions);
    }

    csv.close();

    // *** FINAL PERFORMANCE REPORT ***
    std::cout << "\n=== PERFORMANCE-BASED SIMULATION RESULTS ===" << std::endl;
    uint32_t totalDecisions = 0;
    for (const auto& category : categoryStats) {
        const auto& counts = decisionCountsByCategory[category.first];
        uint32_t categoryTotal = 0;
        for (uint32_t count : counts) categoryTotal += count;
        totalDecisions += categoryTotal;
        
        double avgDecisions = counts.size() > 0 ? double(categoryTotal) / counts.size() : 0.0;  // *** FIXED: Added safety check ***
        
        std::cout << "Category " << category.first << ": " 
                  << category.second << " scenarios, "
                  << "avg " << std::fixed << std::setprecision(0) << avgDecisions << " decisions/scenario" << std::endl;
    }
    
    std::cout << "\nTotal decisions collected: " << totalDecisions << std::endl;
    std::cout << "Results in: smartv3-benchmark-performance-based.csv" << std::endl;
    
    return 0;
}