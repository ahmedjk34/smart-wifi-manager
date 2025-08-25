#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace ns3;

// Struct for describing a single test case
struct BenchmarkTestCase
{
    double staDistance;
    double staSpeed;
    uint32_t numInterferers;
    uint32_t packetSize;
    std::string trafficRate;
    std::string scenarioName;
};

void RateTrace(std::string context, uint64_t rate, uint64_t oldRate)
{
    std::cout << "Rate adaptation event: context=" << context
              << " new datarate=" << rate << " old datarate=" << oldRate << std::endl;
}

void RunTestCase(const BenchmarkTestCase& tc, std::ofstream& csv)
{
    // Create nodes
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    // Interferers
    NodeContainer interfererApNodes;
    NodeContainer interfererStaNodes;
    interfererApNodes.Create(tc.numInterferers);
    interfererStaNodes.Create(tc.numInterferers);

    // Channel and PHY
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::AarfWifiManager");

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // Interferer WiFi
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer interfererStaDevices = wifi.Install(phy, mac, interfererStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer interfererApDevices = wifi.Install(phy, mac, interfererApNodes);

    // AP at origin
    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
    apPositionAlloc->Add(Vector(0.0, 0.0, 0.0));
    apMobility.SetPositionAllocator(apPositionAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(wifiApNode);

    // STA at distance (ONLY ONE MOBILITY MODEL INSTALLED)
    if (tc.staSpeed > 0.0)
    {
        MobilityHelper mobMove;
        mobMove.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> movingAlloc = CreateObject<ListPositionAllocator>();
        movingAlloc->Add(Vector(tc.staDistance, 0.0, 0.0));
        mobMove.SetPositionAllocator(movingAlloc);
        mobMove.Install(wifiStaNodes);
        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(Vector(tc.staSpeed, 0.0, 0.0));
    }
    else
    {
        MobilityHelper mobStill;
        mobStill.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        Ptr<ListPositionAllocator> stillAlloc = CreateObject<ListPositionAllocator>();
        stillAlloc->Add(Vector(tc.staDistance, 0.0, 0.0));
        mobStill.SetPositionAllocator(stillAlloc);
        mobStill.Install(wifiStaNodes);
    }

    // Interferers placed far from main AP and STA
    MobilityHelper interfererMobility;
    Ptr<ListPositionAllocator> interfererApAlloc = CreateObject<ListPositionAllocator>();
    Ptr<ListPositionAllocator> interfererStaAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < tc.numInterferers; ++i)
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

    // Interferer IPs
    address.SetBase("10.1.4.0", "255.255.255.0");
    Ipv4InterfaceContainer interfererApInterface = address.Assign(interfererApDevices);
    Ipv4InterfaceContainer interfererStaInterface = address.Assign(interfererStaDevices);

    // Main Application: UDP bulk traffic
    uint16_t port = 4000;
    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(apInterface.GetAddress(0), port));
    onoff.SetAttribute("DataRate", DataRateValue(DataRate(tc.trafficRate)));
    onoff.SetAttribute("PacketSize", UintegerValue(tc.packetSize));
    onoff.SetAttribute("StartTime", TimeValue(Seconds(2.0)));
    onoff.SetAttribute("StopTime", TimeValue(Seconds(18.0)));
    ApplicationContainer clientApps = onoff.Install(wifiStaNodes.Get(0));

    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer serverApps = sink.Install(wifiApNode.Get(0));
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(20.0));

    // Interferer traffic
    for (uint32_t i = 0; i < tc.numInterferers; ++i)
    {
        OnOffHelper interfererOnOff("ns3::UdpSocketFactory", InetSocketAddress(interfererApInterface.GetAddress(i), port+1));
        interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate("2Mbps")));
        interfererOnOff.SetAttribute("PacketSize", UintegerValue(512));
        interfererOnOff.SetAttribute("StartTime", TimeValue(Seconds(2.0)));
        interfererOnOff.SetAttribute("StopTime", TimeValue(Seconds(18.0)));
        interfererOnOff.Install(interfererStaNodes.Get(i));

        PacketSinkHelper interfererSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port+1));
        interfererSink.Install(interfererApNodes.Get(i));
    }

    // FlowMonitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Enable Rate trace
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/Rate",
                MakeCallback(&RateTrace));

    // Run simulation
    Simulator::Stop(Seconds(20.0));
    Simulator::Run();

    // Results
    double throughput = 0;
    double packetLoss = 0;
    double avgDelay = 0;
    double rxPackets = 0, txPackets = 0;
    double rxBytes = 0;
    double simulationTime = 16.0; // from 2s to 18s

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        // Filter for main STA to AP flow
        if (t.sourceAddress == staInterface.GetAddress(0) && t.destinationAddress == apInterface.GetAddress(0))
        {
            rxPackets = it->second.rxPackets;
            txPackets = it->second.txPackets;
            rxBytes = it->second.rxBytes;
            throughput = (rxBytes * 8.0) / (simulationTime * 1e6); // Mbps
            packetLoss = txPackets > 0 ? 100.0 * (txPackets - rxPackets) / txPackets : 0.0;
            avgDelay = it->second.rxPackets > 0 ? it->second.delaySum.GetSeconds() / it->second.rxPackets * 1000.0 : 0.0;
        }
    }

    // Output to CSV
    csv << "\"" << tc.scenarioName << "\","
        << tc.staDistance << ","
        << tc.staSpeed << ","
        << tc.numInterferers << ","
        << tc.packetSize << ","
        << tc.trafficRate << ","
        << throughput << ","
        << packetLoss << ","
        << avgDelay << ","
        << rxPackets << ","
        << txPackets << "\n";

    Simulator::Destroy();
}

int main(int argc, char *argv[])
{
    // Many test cases in a vector
    std::vector<BenchmarkTestCase> testCases;

    // Fill test cases: distances, speeds, interferers, packet sizes, rates
    std::vector<double> distances = { 20.0, 40.0, 60.0 };      // 3
    std::vector<double> speeds = { 0.0, 10.0 };                // 2
    std::vector<uint32_t> interferers = { 0, 3 };              // 2
    std::vector<uint32_t> packetSizes = { 256, 1500 };         // 2
    std::vector<std::string> trafficRates = { "1Mbps", "11Mbps", "54Mbps" }; // 3

    for (double d : distances)
    {
        for (double s : speeds)
        {
            for (uint32_t i : interferers)
            {
                for (uint32_t p : packetSizes)
                {
                    for (const std::string& r : trafficRates)
                    {
                        std::ostringstream name;
                        name << "dist=" << d << "_speed=" << s << "_intf=" << i << "_pkt=" << p << "_rate=" << r;
                        BenchmarkTestCase tc;
                        tc.staDistance = d;
                        tc.staSpeed = s;
                        tc.numInterferers = i;
                        tc.packetSize = p;
                        tc.trafficRate = r;
                        tc.scenarioName = name.str();
                        testCases.push_back(tc);
                    }
                }
            }
        }
    }

    std::ofstream csv("aarf-benchmark.csv");
    csv << "Scenario,Distance,Speed,Interferers,PacketSize,TrafficRate,Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

    for (const auto& tc : testCases)
    {
        std::cout << "Running: " << tc.scenarioName << std::endl;
        RunTestCase(tc, csv);
    }

    csv.close();
    std::cout << "All tests complete. Results in aarf-benchmark.csv\n";
    return 0;
}