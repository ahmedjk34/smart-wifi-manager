#include "performance-based-parameter-generator.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

namespace ns3
{

std::vector<ScenarioParams>
PerformanceBasedParameterGenerator::GenerateStratifiedScenarios(uint32_t totalScenarios)
{
    std::vector<ScenarioParams> scenarios;

    // Distribute across multiple richer categories
    uint32_t catPoor = totalScenarios * 0.20; // More poor cases
    uint32_t catMedium = totalScenarios * 0.20;
    uint32_t catHighInt = totalScenarios * 0.15;
    uint32_t catGood = totalScenarios * 0.15;
    uint32_t catNearIdeal = totalScenarios * 0.10;
    uint32_t catExtreme = totalScenarios * 0.10;
    uint32_t catEdgeStress = totalScenarios * 0.05;
    uint32_t catRandomChaos = totalScenarios - (catPoor + catMedium + catHighInt + catGood +
                                                catNearIdeal + catExtreme + catEdgeStress);

    for (uint32_t i = 0; i < catPoor; ++i)
        scenarios.push_back(GeneratePoorPerformanceScenario(i));
    for (uint32_t i = 0; i < catMedium; ++i)
        scenarios.push_back(GenerateMediumPerformanceScenario(i));
    for (uint32_t i = 0; i < catHighInt; ++i)
        scenarios.push_back(GenerateHighInterferenceScenario(i));
    for (uint32_t i = 0; i < catGood; ++i)
        scenarios.push_back(GenerateGoodPerformanceScenario(i));
    for (uint32_t i = 0; i < catNearIdeal; ++i)
        scenarios.push_back(GenerateNearIdealScenario(i));
    for (uint32_t i = 0; i < catExtreme; ++i)
        scenarios.push_back(GenerateExtremeScenario(i));
    for (uint32_t i = 0; i < catEdgeStress; ++i)
        scenarios.push_back(GenerateEdgeStressScenario(i));
    for (uint32_t i = 0; i < catRandomChaos; ++i)
        scenarios.push_back(GenerateRandomChaosScenario(i));

    return scenarios;
}

/* ------------------- Existing Scenarios (slightly expanded) ------------------- */

ScenarioParams
PerformanceBasedParameterGenerator::GeneratePoorPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "PoorPerformance";
    params.targetDecisions = 50 + (index % 150); // 50–200 decisions

    // Include **negative & very low SNRs**
    std::vector<std::pair<double, double>> snrRanges = {{-5.0, 0.0},
                                                        {0.0, 5.0},
                                                        {5.0, 10.0},
                                                        {8.0, 12.0},
                                                        {10.0, 15.0}};
    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 1);

    params.speed = 0.5 + (index % 6); // up to 5.5 m/s
    params.interferers = (index % 4); // 0–3 interferers

    std::vector<uint32_t> packetSizes = {64, 128, 256, 512, 768};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"1Mbps", "3Mbps", "5Mbps", "8Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Poor_" << std::setw(3) << index << "_snr" << std::fixed << std::setprecision(1)
         << targetSnr << "_spd" << params.speed << "_if" << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateMediumPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "MediumPerformance";
    params.targetDecisions = 150 + (index % 250);

    std::vector<std::pair<double, double>> snrRanges = {{12.0, 18.0},
                                                        {15.0, 20.0},
                                                        {18.0, 22.0},
                                                        {16.0, 21.0},
                                                        {14.0, 19.0}};
    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 7) / 7.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.5 + (index % 8); // 0.5–8 m/s
    params.interferers = (index % 3); // 0–2 interferers

    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024, 1280};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"8Mbps", "12Mbps", "16Mbps", "20Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Medium_" << std::setw(3) << index << "_snr" << std::fixed << std::setprecision(1)
         << targetSnr << "_spd" << params.speed << "_if" << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateHighInterferenceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "HighInterference";
    params.targetDecisions = 70 + (index % 180);

    std::vector<std::pair<double, double>> snrRanges = {{15.0, 22.0},
                                                        {18.0, 25.0},
                                                        {20.0, 27.0},
                                                        {16.0, 23.0}};
    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 2);

    params.speed = 1.0 + (index % 10);    // 1–10 m/s
    params.interferers = 2 + (index % 4); // 2–5 interferers

    std::vector<uint32_t> packetSizes = {128, 256, 512, 768};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"12Mbps", "15Mbps", "18Mbps", "20Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "HighInt_" << std::setw(3) << index << "_if" << params.interferers << "_spd"
         << params.speed;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateGoodPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "GoodPerformance";
    params.targetDecisions = 200 + (index % 400);

    std::vector<std::pair<double, double>> snrRanges = {{20.0, 26.0},
                                                        {22.0, 28.0},
                                                        {24.0, 30.0},
                                                        {25.0, 32.0}};
    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = (index % 3); // 0–2 m/s
    params.interferers = (index % 4 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {512, 768, 1024, 1280, 1500};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"20Mbps", "25Mbps", "30Mbps", "40Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Good_" << std::setw(3) << index << "_snr" << std::fixed << std::setprecision(1)
         << targetSnr;
    params.scenarioName = name.str();

    return params;
}

/* ------------------- New Scenarios ------------------- */

ScenarioParams
PerformanceBasedParameterGenerator::GenerateNearIdealScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "NearIdeal";
    params.targetDecisions = 500 + (index % 500); // 500–1000 decisions

    params.targetSnrMin = 28.0;
    params.targetSnrMax = 35.0;
    params.distance = CalculateDistanceForSnr(34.0, 0);

    params.speed = 0.0; // Stationary
    params.interferers = 0;

    params.packetSize = 1500;      // Max MTU
    params.trafficRate = "54Mbps"; // Ideal throughput

    std::ostringstream name;
    name << "NearIdeal_" << std::setw(3) << index << "_54Mbps_stationary";
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateExtremeScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "Extreme";
    params.targetDecisions = 20 + (index % 50);

    params.targetSnrMin = -20.0;
    params.targetSnrMax = -5.0;
    params.distance = CalculateDistanceForSnr(-10.0, 5);

    params.speed = 20.0 + (index % 30);     // 20–50 m/s
    params.interferers = 10 + (index % 10); // 10–20 interferers

    params.packetSize = 64; // Minimal
    params.trafficRate = "500Kbps";

    std::ostringstream name;
    name << "Extreme_" << std::setw(3) << index << "_negSNR_if" << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateEdgeStressScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "EdgeStress";
    params.targetDecisions = 100 + (index % 100);

    params.targetSnrMin = 0.0;
    params.targetSnrMax = 35.0;
    params.distance = CalculateDistanceForSnr((index % 2 == 0) ? 1.0 : 30.0, index % 5);

    params.speed = (index % 2 == 0) ? 0.0 : 50.0; // stationary vs ultra fast
    params.interferers = (index % 7);

    params.packetSize = (index % 2 == 0) ? 64 : 1500;
    params.trafficRate = (index % 2 == 0) ? "100Kbps" : "100Mbps";

    std::ostringstream name;
    name << "EdgeStress_" << std::setw(3) << index << "_snrVar_spd" << params.speed << "_if"
         << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateRandomChaosScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "RandomChaos";
    std::mt19937 gen(index);
    std::uniform_real_distribution<> snrDist(-15.0, 35.0);
    std::uniform_int_distribution<> interfererDist(0, 10);
    std::uniform_int_distribution<> pktDist(64, 1500);
    std::uniform_real_distribution<> speedDist(0.0, 50.0);

    double targetSnr = snrDist(gen);
    params.targetSnrMin = targetSnr - 1.5;
    params.targetSnrMax = targetSnr + 1.5;

    params.interferers = interfererDist(gen);
    params.speed = speedDist(gen);

    // Traffic rate chosen based on SNR
    double maxRate;
    if (targetSnr < 0)
        maxRate = 2.0;
    else if (targetSnr < 10)
        maxRate = 10.0;
    else if (targetSnr < 20)
        maxRate = 30.0;
    else if (targetSnr < 30)
        maxRate = 48.0;
    else
        maxRate = 54.0;

    std::uniform_real_distribution<> rateDist(0.5, maxRate);
    double chosenRate = rateDist(gen);

    params.packetSize = (chosenRate > 10.0) ? std::max(512, pktDist(gen)) : pktDist(gen);
    params.trafficRate = std::to_string((int)chosenRate) + "Mbps";

    params.targetDecisions = 50 + (index % 500);
    params.distance = CalculateDistanceForSnr(targetSnr, params.interferers);

    std::ostringstream name;
    name << "Chaos_" << std::setw(3) << index << "_snr" << std::fixed << std::setprecision(1)
         << targetSnr << "_spd" << params.speed << "_if" << params.interferers << "_rate"
         << chosenRate << "Mbps";
    params.scenarioName = name.str();

    return params;
}

/* ------------------- SNR to Distance ------------------- */

double
PerformanceBasedParameterGenerator::CalculateDistanceForSnr(double targetSnr, uint32_t interferers)
{
    double snr = targetSnr + (interferers * 2.0);
    if (snr >= 35.0)
        return 1.0;
    if (snr > 19.0)
        return (35.0 - snr) / 0.8;
    if (snr > 4.0)
        return 20.0 + (19.0 - snr) / 0.5;
    if (snr > -11.0)
        return 50.0 + (4.0 - snr) / 0.3;
    return 100.0 + (-11.0 - snr) / 0.2;
}

} // namespace ns3
