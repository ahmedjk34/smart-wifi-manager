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

// --- HYBRID PATCH START ---
// Logging file (global to allow access in callbacks)
std::ofstream logFile;

// These extern "C" functions allow the manager to log features and picked rates.
extern "C" void LogFeaturesAndRate(const std::vector<double>& features, uint32_t rateIdx, uint64_t rate,
                                   std::string context, double risk, uint32_t ruleRate, double mlConfidence)
{
    logFile << "[ML DEBUG] Features to ML: ";
    for (size_t i = 0; i < features.size(); ++i)
        logFile << features[i] << " ";
    logFile << "-> ML Prediction: " << rateIdx
            << " (Rate: " << rate << "bps)"
            << " | Context: " << context
            << " | Risk: " << risk
            << " | RuleRate: " << ruleRate
            << " | ML Confidence: " << mlConfidence
            << std::endl;
}
// --- HYBRID PATCH END ---

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
    logFile << "[RATE ADAPT EVENT] context=" << context
            << " new datarate=" << rate
            << " old datarate=" << oldRate
            << std::endl;
}

void PhyRxEndTrace(std::string context, Ptr<const Packet> packet)
{
    logFile << "[DEBUG PHY] Packet received at " << context << std::endl;
}

void RunTestCase(const BenchmarkTestCase& tc, std::ofstream& csv, const std::string& modelType)
{
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);

    wifi.SetRemoteStationManager("ns3::SmartWifiManagerRf",
                                 "ModelType", StringValue(modelType));

    if (modelType == "oracle")
    {
        wifi.SetRemoteStationManager("ns3::SmartWifiManagerRf",
                                     "ModelPath", StringValue("step3_rf_oracle_best_rateIdx_model_FIXED.joblib"),
                                     "ModelType", StringValue("oracle"));
    }
    else if (modelType == "v3")
    {
        wifi.SetRemoteStationManager("ns3::SmartWifiManagerRf",
                                     "ModelPath", StringValue("step3_rf_v3_rateIdx_model_FIXED.joblib"),
                                     "ModelType", StringValue("v3"));
    }

    // --- HYBRID PATCH START ---
    // Optionally add context/risk/hybrid thresholds
    wifi.SetRemoteStationManager("ns3::SmartWifiManagerRf",
                                 "ConfidenceThreshold", DoubleValue(0.7),
                                 "RiskThreshold", DoubleValue(0.7),
                                 "FailureThreshold", UintegerValue(4));
    // --- HYBRID PATCH END ---

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
    apPositionAlloc->Add(Vector(0.0, 0.0, 0.0));
    apMobility.SetPositionAllocator(apPositionAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(wifiApNode);

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

    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface = address.Assign(apDevices);
    Ipv4InterfaceContainer staInterface = address.Assign(staDevices);

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

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // --- HYBRID PATCH START ---
    // Connect rate/context/risk tracing
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/Rate",
        MakeCallback(&RateTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxEnd",
        MakeCallback(&PhyRxEndTrace));
    // --- HYBRID PATCH END ---

    Simulator::Stop(Seconds(20.0));
    Simulator::Run();

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

    csv << "\"" << tc.scenarioName << "\","
        << modelType << ","
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
    // Open the log file for writing, will be used globally
    logFile.open("smartrf-logs.txt");
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open smartrf-logs.txt for writing logs." << std::endl;
        return 1;
    }
    logFile << "SmartRF Benchmark Logging Started\n";

    std::vector<BenchmarkTestCase> testCases;
    std::vector<double> distances = { 1.0, 40.0, 120.0 };
    std::vector<double> speeds = { 0.0, 10.0 };
    std::vector<uint32_t> interferers = { 0, 3 };
    std::vector<uint32_t> packetSizes = { 256, 1500 };
    std::vector<std::string> trafficRates = { "1Mbps", "11Mbps", "54Mbps" };

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

    std::vector<std::string> modelTypes = {"oracle", "v3"};

    for (const std::string& modelType : modelTypes)
    {
        std::string csvFilename = "smartrf-benchmark.csv";
        std::ofstream csv(csvFilename);
        csv << "Scenario,ModelType,Distance,Speed,Interferers,PacketSize,TrafficRate,Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

        for (const auto& tc : testCases)
        {
            logFile << "Running RF " << modelType << ": " << tc.scenarioName << std::endl;
            RunTestCase(tc, csv, modelType);
        }

        csv.close();
        logFile << "RF " << modelType << " tests complete. Results in " << csvFilename << "\n";
    }

    logFile << "All RF tests complete!\n";
    logFile.close();
    return 0;
}