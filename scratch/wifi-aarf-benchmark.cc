/*
 * AARF WiFi Manager Benchmark - ENVIRONMENT MATCHED TO MINSTREL BASELINE
 * All physical parameters synchronized for fair comparison
 *
 * CRITICAL FIXES (2025-10-03):
 * ============================================================================
 * 1. PHY parameters: TxPower=30dBm, RxSensitivity=-92dBm, category-based noise
 * 2. Interferer placement: Circular distribution (30+i×15m) matching Minstrel
 * 3. Mobility: Velocity calculation with Y-component for poor conditions
 * 4. Traffic: 0.6× reduction for poor conditions, exponential interferer on/off
 * 5. Category determination: Same logic as Minstrel baseline
 *
 * Author: ahmedjk34 (https://github.com/ahmedjk34)
 * Date: 2025-10-03
 * Version: 3.0 (ENVIRONMENT MATCHED)
 * Baseline: AarfWifiManager (Auto Rate Fallback)
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// ============================================================================
// MATCHED: Realistic SNR conversion (IDENTICAL to Minstrel)
// ============================================================================
enum SnrModel
{
    LOG_MODEL,
    SOFT_MODEL,
    INTF_MODEL
};

double
ConvertNS3ToRealisticSnr(double ns3Value, double distance, uint32_t interferers, SnrModel model)
{
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
    return realisticSnr;
}

// ============================================================================
// MATCHED: Category determination (IDENTICAL to Minstrel)
// ============================================================================
std::string
DetermineCategory(double distance, uint32_t interferers, double speed)
{
    if (distance >= 70.0 || (distance >= 50.0 && speed >= 10.0))
        return "PoorPerformance";
    else if (interferers >= 3 || (interferers >= 2 && distance >= 40.0))
        return "HighInterference";
    else
        return "GoodConditions";
}

// ============================================================================
// Test case structure
// ============================================================================
struct BenchmarkTestCase
{
    double staDistance;
    double staSpeed;
    uint32_t numInterferers;
    uint32_t packetSize;
    std::string trafficRate;
    std::string scenarioName;
};

struct TestCaseStats
{
    uint32_t testCaseNumber;
    std::string scenario;
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
    double pdr;
    double throughput;
    double avgDelay;
    double jitter;
    double simulationTime;
    bool statsValid;
};

TestCaseStats currentStats;

void
RateTrace(std::string context, uint64_t rate, uint64_t oldRate)
{
    // Rate adaptation events (silent logging)
}

void
PrintTestCaseSummary(const TestCaseStats& stats)
{
    std::cout << "\n[TEST " << stats.testCaseNumber << "] AARF BASELINE SUMMARY (ENV MATCHED)"
              << std::endl;
    std::cout << "Scenario=" << stats.scenario << " | Distance=" << stats.distance
              << "m | Speed=" << stats.speed << "m/s | Interferers=" << stats.interferers
              << std::endl;
    std::cout << "TxPackets=" << stats.txPackets << " | RxPackets=" << stats.rxPackets
              << " | PDR=" << std::fixed << std::setprecision(1) << stats.pdr << "%" << std::endl;
    std::cout << "Throughput=" << std::fixed << std::setprecision(2) << stats.throughput
              << " Mbps | AvgDelay=" << std::fixed << std::setprecision(6) << stats.avgDelay << " s"
              << std::endl;
    std::cout << "AvgSNR=" << std::fixed << std::setprecision(1) << stats.avgSNR << " dB"
              << std::endl;
}

void
RunTestCase(const BenchmarkTestCase& tc, std::ofstream& csv, uint32_t testCaseNumber)
{
    currentStats.testCaseNumber = testCaseNumber;
    currentStats.scenario = tc.scenarioName;
    currentStats.distance = tc.staDistance;
    currentStats.speed = tc.staSpeed;
    currentStats.interferers = tc.numInterferers;
    currentStats.packetSize = tc.packetSize;
    currentStats.trafficRate = tc.trafficRate;
    currentStats.simulationTime = 20.0;
    currentStats.statsValid = false;

    // Determine category for environment adjustments
    std::string category = DetermineCategory(tc.staDistance, tc.numInterferers, tc.staSpeed);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "AARF TEST " << testCaseNumber << " | Category: " << category << std::endl;
    std::cout << "Scenario: " << tc.scenarioName << std::endl;
    std::cout << "Expected SNR: "
              << ConvertNS3ToRealisticSnr(100.0, tc.staDistance, tc.numInterferers, SOFT_MODEL)
              << " dB" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create nodes
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    NodeContainer interfererApNodes;
    NodeContainer interfererStaNodes;
    interfererApNodes.Create(tc.numInterferers);
    interfererStaNodes.Create(tc.numInterferers);

    // ============================================================
    // MATCHED PHY CONFIGURATION (IDENTICAL TO MINSTREL BASELINE)
    // ============================================================
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    // MATCHED: Baseline PHY parameters
    phy.Set("TxPowerStart", DoubleValue(30.0));
    phy.Set("TxPowerEnd", DoubleValue(30.0));
    phy.Set("RxNoiseFigure", DoubleValue(3.0)); // Base: 3dB
    phy.Set("CcaEdThreshold", DoubleValue(-82.0));
    phy.Set("RxSensitivity", DoubleValue(-92.0));

    // MATCHED: Category-specific adjustments
    if (category == "PoorPerformance")
    {
        phy.Set("RxNoiseFigure", DoubleValue(5.0));
    }
    else if (category == "HighInterference")
    {
        phy.Set("RxNoiseFigure", DoubleValue(4.0));
    }

    std::cout << "[PHY] Matched to baseline: TxPower=30dBm, RxNoise="
              << (category == "PoorPerformance" ? "5.0"
                                                : (category == "HighInterference" ? "4.0" : "3.0"))
              << "dB" << std::endl;

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::AarfWifiManager");

    WifiMacHelper mac;
    Ssid ssid = Ssid("aarf-baseline-matched");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // Interferer devices
    NetDeviceContainer interfererStaDevices, interfererApDevices;
    if (tc.numInterferers > 0)
    {
        mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("interferer-ssid")));
        interfererStaDevices = wifi.Install(phy, mac, interfererStaNodes);

        mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("interferer-ssid")));
        interfererApDevices = wifi.Install(phy, mac, interfererApNodes);
    }

    // ============================================================
    // MATCHED MOBILITY (IDENTICAL TO MINSTREL BASELINE)
    // ============================================================

    // AP at origin
    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apPositionAlloc = CreateObject<ListPositionAllocator>();
    apPositionAlloc->Add(Vector(0.0, 0.0, 0.0));
    apMobility.SetPositionAllocator(apPositionAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(wifiApNode);

    // STA mobility - MATCHED
    MobilityHelper staMobility;
    if (tc.staSpeed > 0.0)
    {
        // Mobile scenario
        staMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
        staPositionAlloc->Add(Vector(tc.staDistance, 0.0, 0.0));
        staMobility.SetPositionAllocator(staPositionAlloc);
        staMobility.Install(wifiStaNodes);

        // MATCHED: Minstrel velocity calculation
        Vector velocity(tc.staSpeed * 0.5, 0.0, 0.0);
        if (category == "PoorPerformance" || category == "HighInterference")
        {
            velocity.y = tc.staSpeed * 0.05 * ((tc.staDistance > 50) ? 1 : -1);
        }

        wifiStaNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(velocity);
        std::cout << "[MOBILITY] Speed=" << tc.staSpeed << "m/s, Velocity=(" << velocity.x << ","
                  << velocity.y << ",0)" << std::endl;
    }
    else
    {
        // Static scenario
        staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        Ptr<ListPositionAllocator> staPositionAlloc = CreateObject<ListPositionAllocator>();
        staPositionAlloc->Add(Vector(tc.staDistance, 0.0, 0.0));
        staMobility.SetPositionAllocator(staPositionAlloc);
        staMobility.Install(wifiStaNodes);
    }

    // ============================================================
    // MATCHED INTERFERER PLACEMENT (CIRCULAR DISTRIBUTION)
    // ============================================================
    if (tc.numInterferers > 0)
    {
        MobilityHelper interfererMobility;
        interfererMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

        Ptr<ListPositionAllocator> interfererApAlloc = CreateObject<ListPositionAllocator>();
        Ptr<ListPositionAllocator> interfererStaAlloc = CreateObject<ListPositionAllocator>();

        for (uint32_t i = 0; i < tc.numInterferers; ++i)
        {
            double angle = 2.0 * M_PI * i / std::max<uint32_t>(tc.numInterferers, 1);
            double radius = 30.0 + i * 15.0; // MATCHED: staggered

            interfererApAlloc->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 0.0));
            interfererStaAlloc->Add(
                Vector((radius + 10.0) * std::cos(angle), (radius + 10.0) * std::sin(angle), 0.0));
        }

        interfererMobility.SetPositionAllocator(interfererApAlloc);
        interfererMobility.Install(interfererApNodes);

        interfererMobility.SetPositionAllocator(interfererStaAlloc);
        interfererMobility.Install(interfererStaNodes);

        std::cout << "[INTERFERERS] Circular placement: " << tc.numInterferers << " nodes at 30-"
                  << (30 + (tc.numInterferers - 1) * 15) << "m" << std::endl;
    }

    // Internet stack
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);
    if (tc.numInterferers > 0)
    {
        stack.Install(interfererApNodes);
        stack.Install(interfererStaNodes);
    }

    Ipv4AddressHelper address;
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface = address.Assign(apDevices);
    Ipv4InterfaceContainer staInterface = address.Assign(staDevices);

    Ipv4InterfaceContainer interfererApInterface, interfererStaInterface;
    if (tc.numInterferers > 0)
    {
        address.SetBase("10.1.4.0", "255.255.255.0");
        interfererApInterface = address.Assign(interfererApDevices);
        interfererStaInterface = address.Assign(interfererStaDevices);
    }

    // ============================================================
    // MATCHED TRAFFIC CONFIGURATION
    // ============================================================
    uint16_t port = 4000;

    // MATCHED: Category-based traffic adjustment
    std::string adjustedRate = tc.trafficRate;
    if (category == "PoorPerformance" || category == "HighInterference")
    {
        double rateValue = std::stod(tc.trafficRate.substr(0, tc.trafficRate.length() - 4));
        rateValue *= 0.6; // MATCHED: 60% reduction
        adjustedRate = std::to_string(static_cast<int>(rateValue)) + "Mbps";
        std::cout << "[TRAFFIC] Adjusted rate: " << tc.trafficRate << " -> " << adjustedRate
                  << " (poor conditions)" << std::endl;
    }

    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(apInterface.GetAddress(0), port));
    onoff.SetAttribute("DataRate", DataRateValue(DataRate(adjustedRate)));
    onoff.SetAttribute("PacketSize", UintegerValue(tc.packetSize));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
    onoff.SetAttribute("StartTime", TimeValue(Seconds(3.0)));
    onoff.SetAttribute("StopTime", TimeValue(Seconds(17.0)));
    ApplicationContainer clientApps = onoff.Install(wifiStaNodes.Get(0));

    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer serverApps = sink.Install(wifiApNode.Get(0));
    serverApps.Start(Seconds(2.0));
    serverApps.Stop(Seconds(18.0));

    // MATCHED: Interferer traffic
    if (tc.numInterferers > 0)
    {
        for (uint32_t i = 0; i < tc.numInterferers; ++i)
        {
            std::string interfererRate = "1Mbps";
            if (category == "HighInterference")
                interfererRate = "2Mbps"; // MATCHED

            OnOffHelper interfererOnOff(
                "ns3::UdpSocketFactory",
                InetSocketAddress(interfererApInterface.GetAddress(i), port + 1 + i));
            interfererOnOff.SetAttribute("DataRate", DataRateValue(DataRate(interfererRate)));
            interfererOnOff.SetAttribute("PacketSize", UintegerValue(256)); // MATCHED: 256
            interfererOnOff.SetAttribute("OnTime",
                                         StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            interfererOnOff.SetAttribute("OffTime",
                                         StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            interfererOnOff.SetAttribute("StartTime", TimeValue(Seconds(3.5)));
            interfererOnOff.SetAttribute("StopTime", TimeValue(Seconds(16.5)));
            interfererOnOff.Install(interfererStaNodes.Get(i));

            PacketSinkHelper interfererSink("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), port + 1 + i));
            interfererSink.Install(interfererApNodes.Get(i));
        }
    }

    // Flow monitoring
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/Rate",
                    MakeCallback(&RateTrace));

    Simulator::Stop(Seconds(20.0));
    std::cout << "Starting simulation (20 seconds - ENVIRONMENT MATCHED)..." << std::endl;
    Simulator::Run();

    // Collect results
    double throughput = 0, packetLoss = 0, avgDelay = 0, jitter = 0;
    double rxPackets = 0, txPackets = 0, rxBytes = 0;
    double simulationTime = 14.0;
    uint32_t retransmissions = 0, droppedPackets = 0;
    bool flowStatsFound = false;

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        if (t.sourceAddress == staInterface.GetAddress(0) &&
            t.destinationAddress == apInterface.GetAddress(0))
        {
            flowStatsFound = true;
            rxPackets = it->second.rxPackets;
            txPackets = it->second.txPackets;
            rxBytes = it->second.rxBytes;
            droppedPackets = it->second.lostPackets;
            retransmissions = it->second.timesForwarded;
            throughput = (rxBytes * 8.0) / (simulationTime * 1e6);
            packetLoss = txPackets > 0 ? 100.0 * (txPackets - rxPackets) / txPackets : 0.0;
            avgDelay = it->second.rxPackets > 0
                           ? it->second.delaySum.GetSeconds() / it->second.rxPackets
                           : 0.0;
            jitter = it->second.rxPackets > 1
                         ? it->second.jitterSum.GetSeconds() / (it->second.rxPackets - 1)
                         : 0.0;
            break;
        }
    }

    // MATCHED: Same SNR calculation
    double avgSnr = ConvertNS3ToRealisticSnr(100.0, tc.staDistance, tc.numInterferers, SOFT_MODEL);
    currentStats.avgSNR = avgSnr;
    currentStats.minSNR = avgSnr - 3.0;
    currentStats.maxSNR = avgSnr + 3.0;

    currentStats.txPackets = txPackets;
    currentStats.rxPackets = rxPackets;
    currentStats.droppedPackets = droppedPackets;
    currentStats.retransmissions = retransmissions;
    currentStats.pdr = txPackets > 0 ? 100.0 * rxPackets / txPackets : 0.0;
    currentStats.throughput = throughput;
    currentStats.avgDelay = avgDelay;
    currentStats.jitter = jitter;
    currentStats.statsValid = flowStatsFound;

    PrintTestCaseSummary(currentStats);

    // CSV output
    csv << "\"" << tc.scenarioName << "\"," << tc.staDistance << "," << tc.staSpeed << ","
        << tc.numInterferers << "," << tc.packetSize << "," << tc.trafficRate << "," << throughput
        << "," << packetLoss << "," << avgDelay << "," << jitter << "," << rxPackets << ","
        << txPackets << "," << avgSnr << "," << (flowStatsFound ? "TRUE" : "FALSE") << "\n";

    Simulator::Destroy();
}

int
main(int argc, char* argv[])
{
    std::vector<BenchmarkTestCase> testCases;

    // Same test matrix as before
    std::vector<double> distances = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
    std::vector<double> speeds = {0.0, 5.0, 10.0, 15.0};
    std::vector<uint32_t> interferers = {0, 1, 2, 3};
    std::vector<uint32_t> packetSizes = {256, 1500};
    std::vector<std::string> trafficRates = {"1Mbps", "11Mbps", "54Mbps"};

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
                        if (s >= 10.0 && d >= 60.0)
                            continue;
                        if (r == "1Mbps" && p == 1500 && d >= 70.0)
                            continue;
                        if (i >= 3 && s >= 15.0)
                            continue;

                        std::ostringstream name;
                        name << "dist=" << d << "_speed=" << s << "_intf=" << i << "_pkt=" << p
                             << "_rate=" << r;

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

    std::cout << "AARF Baseline Benchmark v3.0 (ENVIRONMENT MATCHED TO MINSTREL)" << std::endl;
    std::cout << "Total test cases: " << testCases.size() << std::endl;
    std::cout << "Physical environment: MATCHED (PHY, mobility, interferers, traffic)" << std::endl;
    std::cout << "Category-based adjustments: ENABLED" << std::endl;

    std::ofstream csv("aarf-benchmark-environment-matched.csv");
    csv << "Scenario,Distance,Speed,Interferers,PacketSize,TrafficRate,Throughput(Mbps),"
        << "PacketLoss(%),AvgDelay(s),Jitter(s),RxPackets,TxPackets,AvgSNR,StatsValid\n";

    uint32_t testCaseNumber = 1;
    for (const auto& tc : testCases)
    {
        std::cout << "\nStarting Test " << testCaseNumber << "/" << testCases.size() << ": "
                  << tc.scenarioName << std::endl;
        RunTestCase(tc, csv, testCaseNumber);
        testCaseNumber++;
    }

    csv.close();
    std::cout << "\nAll tests complete. Results in aarf-benchmark-environment-matched.csv\n";
    std::cout << "Environment now IDENTICAL to Minstrel baseline for fair comparison.\n";
    return 0;
}