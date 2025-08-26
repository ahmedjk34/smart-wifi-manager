#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/propagation-module.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

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

// Struct for aggregating all case-level statistics
struct CaseStats {
    uint32_t totalTx = 0;
    uint32_t totalRx = 0;
    uint32_t totalDropped = 0;
    uint32_t retransmissions = 0;
    double totalDelay = 0.0;

    // SNR stats
    double sumSnr = 0.0;
    uint32_t snrSamples = 0;
    double minSnr = 1e9;
    double maxSnr = -1e9;

    // Timing
    double startTime = 0.0;
    double endTime = 0.0;
};

static CaseStats g_caseStats;
static uint32_t g_currentTestCase = 0;

// ====== PACKET-LEVEL CALLBACKS AGGREGATE INTO g_caseStats ======

// Rate adaptation tracing (kept as case-level event for debugging/tuning)
void RateTrace(std::string context, uint64_t rate, uint64_t oldRate)
{
    // Optional: Toggle by debug flag.
    // std::cout << "[TEST " << g_currentTestCase << "] RATE ADAPTATION: " 
    //           << "Context=" << context
    //           << " | NEW_RATE=" << rate/1000000.0 << "Mbps"
    //           << " | OLD_RATE=" << oldRate/1000000.0 << "Mbps" 
    //           << " | Time=" << std::fixed << std::setprecision(3) << Simulator::Now().GetSeconds() << "s"
    //           << std::endl;
}

// PHY state change tracing (optional for debugging)
void PhyStateTrace(std::string context, Time start, Time duration, WifiPhyState state)
{
    // (Optional: Remove or toggle for debugging)
}

// PHY RX start tracing - SNR stats
void PhyRxStartTrace(std::string context, Ptr<const Packet> packet, RxPowerWattPerChannelBand rxPowersW)
{
    double totalPower = 0.0;
    for (auto& power : rxPowersW)
    {
        totalPower += power.second;
    }
    double rssi_dbm = 10 * log10(totalPower * 1000); // Convert W to mW then to dBm

    // Aggregate SNR stats
    g_caseStats.sumSnr += rssi_dbm;
    g_caseStats.snrSamples++;
    g_caseStats.minSnr = std::min(g_caseStats.minSnr, rssi_dbm);
    g_caseStats.maxSnr = std::max(g_caseStats.maxSnr, rssi_dbm);
}

// PHY RX end tracing
void PhyRxEndTrace(std::string context, Ptr<const Packet> packet)
{
    g_caseStats.totalRx++;
}

// PHY RX drop tracing
void PhyRxDropTrace(std::string context, Ptr<const Packet> packet, WifiPhyRxfailureReason reason)
{
    g_caseStats.totalDropped++;
}

// PHY TX start tracing
void PhyTxStartTrace(std::string context, Ptr<const Packet> packet, double txPowerW)
{
    g_caseStats.totalTx++;
}

// PHY TX end tracing
void PhyTxEndTrace(std::string context, Ptr<const Packet> packet)
{
    // No need to aggregate; keep for debugging.
}

// MAC TX tracing
void MacTxTrace(std::string context, Ptr<const Packet> packet)
{
    // No need to aggregate; keep for debugging.
}

// MAC RX tracing
void MacRxTrace(std::string context, Ptr<const Packet> packet)
{
    // No need to aggregate; keep for debugging.
}

// MAC TX drop tracing
void MacTxDropTrace(std::string context, Ptr<const Packet> packet)
{
    // Consider incrementing retransmissions if desired.
    g_caseStats.retransmissions++;
}

// Queue tracing (optional)
void QueueTrace(std::string context, Ptr<const Packet> packet)
{
    // No need to aggregate; keep for debugging.
}

void QueueDropTrace(std::string context, Ptr<const Packet> packet)
{
    g_caseStats.totalDropped++;
}

// Application level tracing
void AppTxTrace(std::string context, Ptr<const Packet> packet)
{
    // No need to aggregate; keep for debugging.
}

void AppRxTrace(std::string context, Ptr<const Packet> packet, const Address& address)
{
    // No need to aggregate; keep for debugging.
}

// Position tracing for mobile nodes (optional aggregation)
void PositionTrace(std::string context, Ptr<const MobilityModel> model)
{
    // Optionally aggregate mobility stats here
}

// SNR monitoring function (optional periodic stats)
void MonitorSnr()
{
    // (Optional: Remove or toggle for debugging)
    if (Simulator::Now().GetSeconds() < 19.0) // Stop before simulation ends
    {
        Simulator::Schedule(Seconds(1.0), &MonitorSnr);
    }
}

// ====== CASE SUMMARY OUTPUT ======
void PrintCaseSummary(uint32_t testId, const BenchmarkTestCase& tc)
{
    double avgSnr = (g_caseStats.snrSamples > 0) ? g_caseStats.sumSnr / g_caseStats.snrSamples : 0.0;
    double pdr = (g_caseStats.totalTx > 0) ? (double)g_caseStats.totalRx / g_caseStats.totalTx * 100.0 : 0.0;
    double throughputMbps = (g_caseStats.totalRx * tc.packetSize * 8.0) / (1e6 * (g_caseStats.endTime - g_caseStats.startTime));
    double avgDelay = (g_caseStats.totalRx > 0) ? g_caseStats.totalDelay / g_caseStats.totalRx : 0.0;

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "[TEST " << testId << "] CASE SUMMARY\n";
    std::cout << "Scenario=" << tc.scenarioName
              << " | Distance=" << tc.staDistance << "m"
              << " | Speed=" << tc.staSpeed << "m/s"
              << " | Interferers=" << tc.numInterferers
              << " | PacketSize=" << tc.packetSize
              << " | TrafficRate=" << tc.trafficRate << "\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "TxPackets=" << g_caseStats.totalTx
              << " | RxPackets=" << g_caseStats.totalRx
              << " | Dropped=" << g_caseStats.totalDropped
              << " | Retransmissions=" << g_caseStats.retransmissions << "\n";
    std::cout << "AvgSNR=" << avgSnr << " dBm"
              << " | MinSNR=" << g_caseStats.minSnr << " dBm"
              << " | MaxSNR=" << g_caseStats.maxSnr << " dBm\n";
    std::cout << "PDR=" << pdr << "%"
              << " | Throughput=" << throughputMbps << " Mbps\n";
    std::cout << "AvgDelay=" << avgDelay << " s\n";
    std::cout << "SimulationTime=" << g_caseStats.endTime - g_caseStats.startTime << " s\n";
    std::cout << std::string(80, '=') << "\n";
}

// ====== TEST RUNNER ======
void RunTestCase(const BenchmarkTestCase& tc, std::ofstream& csv)
{
    // Reset stats
    g_caseStats = CaseStats();

    // Start time
    g_caseStats.startTime = Simulator::Now().GetSeconds();

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

    // Channel and PHY with explicit propagation model
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::FriisPropagationLossModel");

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.SetErrorRateModel("ns3::YansErrorRateModel");

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::SmartWifiManagerV1");

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    mac.SetType("ns3::StaWifiMac", 
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", 
                "Ssid", SsidValue(ssid),
                "EnableBeaconJitter", BooleanValue(false));
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

    // STA at distance
    if (tc.staSpeed > 0.0)
    {
        MobilityHelper mobMove;
        mobMove.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> movingAlloc = CreateObject<ListPositionAllocator>();
        movingAlloc->Add(Vector(tc.staDistance, 0.0, 0.0));
        mobMove.SetPositionAllocator(movingAlloc);
        mobMove.Install(wifiStaNodes);
        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(Vector(tc.staSpeed, 0.0, 0.0));
        // Optionally connect position tracing for mobile node
        // std::ostringstream mobilityContext;
        // mobilityContext << "/NodeList/" << wifiStaNodes.Get(0)->GetId() << "/$ns3::MobilityModel/CourseChange";
        // Config::Connect(mobilityContext.str(), MakeCallback(&PositionTrace));
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
        OnOffHelper interfererOnOff("ns3::UdpSocketFactory", InetSocketAddress(interfererApInterface.GetAddress(i), port+1+i));
        interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate("2Mbps")));
        interfererOnOff.SetAttribute("PacketSize", UintegerValue(512));
        interfererOnOff.SetAttribute("StartTime", TimeValue(Seconds(2.0)));
        interfererOnOff.SetAttribute("StopTime", TimeValue(Seconds(18.0)));
        interfererOnOff.Install(interfererStaNodes.Get(i));

        PacketSinkHelper interfererSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port+1+i));
        interfererSink.Install(interfererApNodes.Get(i));
    }

    // Enable only necessary traces (packet-level traces aggregate to g_caseStats)
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/Rate",
                    MakeCallback(&RateTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                    MakeCallback(&PhyTxStartTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxEnd",
                    MakeCallback(&PhyTxEndTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxEnd",
                    MakeCallback(&PhyRxEndTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop",
                    MakeCallback(&PhyRxDropTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxBegin",
                    MakeCallback(&PhyRxStartTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTxDrop",
                    MakeCallback(&MacTxDropTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
                    MakeCallback(&MacRxTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                    MakeCallback(&MacTxTrace));
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::OnOffApplication/Tx",
                    MakeCallback(&AppTxTrace));
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx",
                    MakeCallback(&AppRxTrace));

    // FlowMonitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Start periodic monitoring (optional)
    Simulator::Schedule(Seconds(3.0), &MonitorSnr);

    // Simulation run
    Simulator::Stop(Seconds(20.0));
    Simulator::Run();

    // End time
    g_caseStats.endTime = Simulator::Now().GetSeconds();

    // Collect FlowMonitor stats for delay, throughput, etc.
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    double throughput = 0;
    double packetLoss = 0;
    double avgDelay = 0;
    double rxPackets = 0, txPackets = 0;
    double rxBytes = 0;
    double simulationTime = 16.0; // from 2s to 18s

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
            avgDelay = it->second.rxPackets > 0 ? it->second.delaySum.GetSeconds() / it->second.rxPackets : 0.0;
            g_caseStats.totalDelay = it->second.delaySum.GetSeconds(); // Save for summary avg
        }
    }

    // Print single case-level summary for this test case
    PrintCaseSummary(g_currentTestCase, tc);

    // Output to CSV
    csv << "\"" << tc.scenarioName << "\","
        << tc.staDistance << ","
        << tc.staSpeed << ","
        << tc.numInterferers << ","
        << tc.packetSize << ","
        << tc.trafficRate << ","
        << throughput << ","
        << packetLoss << ","
        << avgDelay * 1000.0 << "," // ms
        << rxPackets << ","
        << txPackets << "\n";

    Simulator::Destroy();
}

int main(int argc, char *argv[])
{
    // Enable logging for even more detailed output
    LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
    LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
    LogComponentEnable("WifiPhy", LOG_LEVEL_FUNCTION);
    LogComponentEnable("YansWifiPhy", LOG_LEVEL_FUNCTION);
    LogComponentEnable("WifiMac", LOG_LEVEL_FUNCTION);
    LogComponentEnable("AarfWifiManager", LOG_LEVEL_INFO);

    std::cout << "\n" << std::string(100, '#') << std::endl;
    std::cout << "# COMPREHENSIVE NS3 WIFI BENCHMARK WITH CASE-LEVEL SUMMARY LOGGING" << std::endl;
    std::cout << "# All PHY, MAC, Queue, Application, and Mobility traces aggregate to per-case stats" << std::endl;
    std::cout << std::string(100, '#') << std::endl;

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

    std::ofstream csv("smartv1-benchmark.csv");
    csv << "Scenario,Distance,Speed,Interferers,PacketSize,TrafficRate,Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

    std::cout << "\nTOTAL TEST CASES TO RUN: " << testCases.size() << std::endl;
    std::cout << "Expected runtime: ~" << (testCases.size() * 20) << " seconds of simulation time\n" << std::endl;

    for (const auto& tc : testCases)
    {
        g_currentTestCase++;
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "*** EXECUTING TEST CASE " << g_currentTestCase << " of " << testCases.size() << " ***" << std::endl;
        std::cout << "*** " << tc.scenarioName << " ***" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        RunTestCase(tc, csv);

        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "*** TEST CASE " << g_currentTestCase << " COMPLETED ***" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
    }

    csv.close();

    std::cout << "\n" << std::string(100, '#') << std::endl;
    std::cout << "# ALL TESTS COMPLETE!" << std::endl;
    std::cout << "# Results saved to: smartv1-benchmark.csv" << std::endl;
    std::cout << "# Total test cases executed: " << testCases.size() << std::endl;
    std::cout << std::string(100, '#') << std::endl;

    return 0;
}

