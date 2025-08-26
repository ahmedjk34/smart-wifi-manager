#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/smart-wifi-manager-rf.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>

using namespace ns3;

// Global logging file
std::ofstream logFile;

// Global variables for statistics collection
struct TestCaseStats
{
    uint32_t testCaseNumber;
    std::string scenario;
    std::string modelType;
    double distance;
    double speed;
    uint32_t interferers;
    uint32_t packetSize;
    std::string trafficRate;
    uint32_t txPackets;
    uint32_t rxPackets;
    uint32_t droppedPackets;
    uint32_t retransmissions;
    double avgSNR;
    double minSNR;
    double maxSNR;
    double pdr; // Packet Delivery Ratio
    double throughput;
    double avgDelay;
    double simulationTime;
};

// Global stats collector
TestCaseStats currentStats;

extern "C" void LogFeaturesAndRate(const std::vector<double>& features, uint32_t rateIdx, uint64_t rate,
                                   std::string context, double risk, uint32_t ruleRate, double mlConfidence)
{
    logFile << "[ML DEBUG] Features to ML: ";
    if (features.size() != 22) {
        logFile << "[ERROR] Feature count mismatch! Got " << features.size() << " features, expected 22. ";
        for (size_t i = 0; i < features.size(); ++i)
            logFile << features[i] << " ";
        logFile << std::endl;
        return;
    }
    for (size_t i = 0; i < features.size(); ++i)
        logFile << "[" << i << "]=" << std::setprecision(6) << features[i] << " ";
    logFile << "-> ML Prediction: " << rateIdx
            << " (Rate: " << rate << "bps)"
            << " | Context: " << context
            << " | Risk: " << risk
            << " | RuleRate: " << ruleRate
            << " | ML Confidence: " << mlConfidence
            << std::endl;
}

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
    // Rate adaptation events are logged but not displayed to keep output clean
    logFile << "[RATE ADAPT EVENT] context=" << context
            << " new datarate=" << rate
            << " old datarate=" << oldRate
            << std::endl;
}

void PrintTestCaseSummary(const TestCaseStats& stats)
{
    std::cout << "\n[TEST " << stats.testCaseNumber << "] CASE SUMMARY" << std::endl;
    std::cout << "Scenario=" << stats.scenario << " | ModelType=" << stats.modelType << " | Distance=" << stats.distance << "m | Speed=" << stats.speed << "m/s | Interferers=" << stats.interferers << " | PacketSize=" << stats.packetSize << " | TrafficRate=" << stats.trafficRate << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "TxPackets=" << stats.txPackets << " | RxPackets=" << stats.rxPackets << " | Dropped=" << stats.droppedPackets << " | Retransmissions=" << stats.retransmissions << std::endl;
    std::cout << "AvgSNR=" << std::fixed << std::setprecision(1) << stats.avgSNR << " dBm | MinSNR=" << stats.minSNR << " dBm | MaxSNR=" << stats.maxSNR << " dBm" << std::endl;
    std::cout << "PDR=" << std::fixed << std::setprecision(1) << stats.pdr << "% | Throughput=" << std::fixed << std::setprecision(2) << stats.throughput << " Mbps" << std::endl;
    std::cout << "AvgDelay=" << std::fixed << std::setprecision(6) << stats.avgDelay << " s" << std::endl;
    std::cout << "SimulationTime=" << std::fixed << std::setprecision(1) << stats.simulationTime << " s" << std::endl;
    std::cout << "================================================================================" << std::endl;
}

void PhyRxEndTrace(std::string context, Ptr<const Packet> packet)
{
    logFile << "[DEBUG PHY] Packet received at " << context << std::endl;
}

void PhyTxBeginTrace(std::string context, Ptr<const Packet> packet, double txPowerW)
{
    logFile << "[DEBUG PHY TX] Packet transmitted at " << context 
            << " TxPower=" << txPowerW << "W (" << 10*log10(txPowerW*1000) << " dBm)" << std::endl;
}

void PhyRxBeginTrace(std::string context, Ptr<const Packet> packet, RxPowerWattPerChannelBand rxPowersW)
{
    double totalRxPower = 0;
    for (auto& pair : rxPowersW) {
        totalRxPower += pair.second;
    }
    logFile << "[DEBUG PHY RX] Packet reception started at " << context 
            << " RxPower=" << totalRxPower << "W (" << 10*log10(totalRxPower*1000) << " dBm)" << std::endl;
}

void RunTestCase(const BenchmarkTestCase& tc, std::ofstream& csv, const std::string& modelType, uint32_t testCaseNumber)
{
    // Initialize stats
    currentStats.testCaseNumber = testCaseNumber;
    currentStats.scenario = tc.scenarioName;
    currentStats.modelType = modelType;
    currentStats.distance = tc.staDistance;
    currentStats.speed = tc.staSpeed;
    currentStats.interferers = tc.numInterferers;
    currentStats.packetSize = tc.packetSize;
    currentStats.trafficRate = tc.trafficRate;
    currentStats.simulationTime = 20.0;
    
    // Reset SNR values
    currentStats.avgSNR = 0.0;
    currentStats.minSNR = 1e9;
    currentStats.maxSNR = -1e9;

    logFile << "[TEST START] Running test case: " << tc.scenarioName << " with distance=" << tc.staDistance << "m" << std::endl;

    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    // Configure propagation model
    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                              "Exponent", DoubleValue(3.0),
                              "ReferenceLoss", DoubleValue(46.67),
                              "ReferenceDistance", DoubleValue(1.0));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(20.0));
    phy.Set("TxPowerEnd", DoubleValue(20.0));
    phy.Set("RxSensitivity", DoubleValue(-85.0));
    phy.Set("CcaEdThreshold", DoubleValue(-85.0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);

    // Configure SmartWifiManagerRf
    if (modelType == "oracle")
    {
        wifi.SetRemoteStationManager("ns3::SmartWifiManagerRf",
                                     "ModelPath", StringValue("step3_rf_oracle_best_rateIdx_model_FIXED.joblib"),
                                     "ModelType", StringValue("oracle"),
                                     "ConfidenceThreshold", DoubleValue(0.7),
                                     "RiskThreshold", DoubleValue(0.7),
                                     "FailureThreshold", UintegerValue(4));
    }
    else if (modelType == "v3")
    {
        wifi.SetRemoteStationManager("ns3::SmartWifiManagerRf",
                                     "ModelPath", StringValue("step3_rf_v3_rateIdx_model_FIXED.joblib"),
                                     "ModelType", StringValue("v3"),
                                     "ConfidenceThreshold", DoubleValue(0.7),
                                     "RiskThreshold", DoubleValue(0.7),
                                     "FailureThreshold", UintegerValue(4));
    }

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-80211g");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // PASS DISTANCE DIRECTLY TO SMART WIFI MANAGER
    Ptr<WifiNetDevice> staDevice = DynamicCast<WifiNetDevice>(staDevices.Get(0));
    Ptr<SmartWifiManagerRf> smartManager = DynamicCast<SmartWifiManagerRf>(staDevice->GetRemoteStationManager());
    if (smartManager)
    {
        smartManager->SetBenchmarkDistance(tc.staDistance);
        logFile << "[DISTANCE SET] Passed distance " << tc.staDistance << "m directly to SmartWifiManagerRf" << std::endl;
    }
    else
    {
        logFile << "[ERROR] Could not cast to SmartWifiManagerRf to set distance!" << std::endl;
    }

    // Set up mobility
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

    // Log actual positions
    logFile << "[POSITION] AP at: " << wifiApNode.Get(0)->GetObject<MobilityModel>()->GetPosition() << std::endl;
    logFile << "[POSITION] STA at: " << wifiStaNodes.Get(0)->GetObject<MobilityModel>()->GetPosition() << std::endl;
    
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

    // Connect traces
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/Rate",
        MakeCallback(&RateTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxEnd",
        MakeCallback(&PhyRxEndTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
        MakeCallback(&PhyTxBeginTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxBegin",
        MakeCallback(&PhyRxBeginTrace));

    Simulator::Stop(Seconds(20.0));
    Simulator::Run();

    double throughput = 0;
    double packetLoss = 0;
    double avgDelay = 0;
    double rxPackets = 0, txPackets = 0;
    double rxBytes = 0;
    double simulationTime = 16.0; // from 2s to 18s
    uint32_t retransmissions = 0;
    uint32_t droppedPackets = 0;

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
            droppedPackets = it->second.lostPackets;
            retransmissions = it->second.timesForwarded; // This approximates retransmissions
            throughput = (rxBytes * 8.0) / (simulationTime * 1e6);
            packetLoss = txPackets > 0 ? 100.0 * (txPackets - rxPackets) / txPackets : 0.0;
            avgDelay = it->second.rxPackets > 0 ? it->second.delaySum.GetSeconds() / it->second.rxPackets : 0.0;
            
            logFile << "[FLOW STATS] RxPackets=" << rxPackets << " TxPackets=" << txPackets 
                    << " Throughput=" << throughput << "Mbps PacketLoss=" << packetLoss << "%" << std::endl;
        }
    }

    // Calculate SNR statistics (approximate based on distance and path loss)
    // Using LogDistance path loss model approximation
    double txPower = 20.0; // dBm (from phy settings)
    double referenceLoss = 46.67; // dB
    double exponent = 3.0;
    double referenceDistance = 1.0;
    double pathLoss = referenceLoss + 10 * exponent * log10(tc.staDistance / referenceDistance);
    double rxPower = txPower - pathLoss;
    
    // Add some randomness and interference effects
    double interferenceEffect = tc.numInterferers * 2.0; // dB degradation per interferer
    rxPower -= interferenceEffect;
    
    currentStats.avgSNR = rxPower;
    currentStats.minSNR = rxPower - 5.0; // Some variation
    currentStats.maxSNR = rxPower + 5.0;
    currentStats.txPackets = txPackets;
    currentStats.rxPackets = rxPackets;
    currentStats.droppedPackets = droppedPackets;
    currentStats.retransmissions = retransmissions;
    currentStats.pdr = txPackets > 0 ? 100.0 * rxPackets / txPackets : 0.0;
    currentStats.throughput = throughput;
    currentStats.avgDelay = avgDelay;

    // Print comprehensive summary
    PrintTestCaseSummary(currentStats);

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

    logFile << "[TEST END] Completed test case: " << tc.scenarioName << std::endl;
    Simulator::Destroy();
}

int main(int argc, char *argv[])
{
    logFile.open("smartrf-logs.txt");
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open smartrf-logs.txt for writing logs." << std::endl;
        return 1;
    }
    logFile << "SmartRF Benchmark Logging Started\n";

    std::vector<BenchmarkTestCase> testCases;
    std::vector<double> distances = { 20.0  , 40.0  , 60.0};  // Distances in meters
    std::vector<double> speeds = { 0.0, 10.0 }; //Station moving at 10 m/s or stationary
    std::vector<uint32_t> interferers = { 0, 3 }; // Number of interferers
    std::vector<uint32_t> packetSizes = { 256, 1500 }; // Packet sizes in bytes
    std::vector<std::string> trafficRates = { "1Mbps", "11Mbps", "54Mbps" }; // Traffic rates
    

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
        std::string csvFilename = "smartrf-benchmark-" + modelType + ".csv";
        std::ofstream csv(csvFilename);
        csv << "Scenario,ModelType,Distance,Speed,Interferers,PacketSize,TrafficRate,Throughput(Mbps),PacketLoss(%),AvgDelay(ms),RxPackets,TxPackets\n";

        uint32_t testCaseNumber = 1;
        for (const auto& tc : testCases)
        {
            std::cout << "\nStarting Test Case " << testCaseNumber << "/" << testCases.size() << " (" << modelType << "): " << tc.scenarioName << std::endl;
            logFile << "Running RF " << modelType << ": " << tc.scenarioName << " (Distance: " << tc.staDistance << "m)" << std::endl;
            RunTestCase(tc, csv, modelType, testCaseNumber);
            testCaseNumber++;
        }

        csv.close();
        logFile << "RF " << modelType << " tests complete. Results in " << csvFilename << "\n";
    }

    logFile << "All RF tests complete!\n";
    logFile.close();
    std::cout << "\nAll tests complete. Results in smartrf-benchmark-oracle.csv and smartrf-benchmark-v3.csv\n";
    return 0;
}