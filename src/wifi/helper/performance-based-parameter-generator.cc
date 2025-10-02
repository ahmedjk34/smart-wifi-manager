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

    // FIXED: Balanced distribution + EQUAL TIME for all scenarios
    // 20% each category = balanced representation
    uint32_t catPoor = std::round(totalScenarios * 0.20);      // Rates 0-2
    uint32_t catMedium = std::round(totalScenarios * 0.20);    // Rates 2-4
    uint32_t catGood = std::round(totalScenarios * 0.20);      // Rates 4-5
    uint32_t catExcellent = std::round(totalScenarios * 0.20); // Rates 5-7
    uint32_t catChaos = std::round(totalScenarios * 0.20);     // Mixed

    // Adjust to match total
    uint32_t sum = catPoor + catMedium + catGood + catExcellent + catChaos;
    if (sum < totalScenarios)
        catChaos += (totalScenarios - sum);
    else if (sum > totalScenarios)
        catChaos -= (sum - totalScenarios);

    // Generate scenarios - ALL with same target decisions (time-controlled)
    for (uint32_t i = 0; i < catPoor; ++i)
        scenarios.push_back(GeneratePoorPerformanceScenario(i));

    for (uint32_t i = 0; i < catMedium; ++i)
        scenarios.push_back(GenerateMediumPerformanceScenario(i));

    for (uint32_t i = 0; i < catGood; ++i)
        scenarios.push_back(GenerateGoodPerformanceScenario(i));

    for (uint32_t i = 0; i < catExcellent; ++i)
        scenarios.push_back(GenerateExcellentPerformanceScenario(i));

    for (uint32_t i = 0; i < catChaos; ++i)
        scenarios.push_back(GenerateRandomChaosScenario(i));

    return scenarios;
}

ScenarioParams
PerformanceBasedParameterGenerator::GeneratePoorPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "PoorPerformance";

    // FIXED: Time-based (not packet-based)
    params.targetDecisions = 999999; // Unlimited - time controls termination

    // SNR range: 8-15 dB (rates 0-2)
    std::vector<std::pair<double, double>> snrRanges = {
        {8.0, 10.0}, // Rate 0-1
        {9.0, 11.0},
        {10.0, 12.0}, // Rate 1-2
        {11.0, 13.0},
        {12.0, 15.0} // Rate 2
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.5 + (index % 4);
    params.interferers = (index % 3 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"3Mbps", "5Mbps", "8Mbps", "10Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Poor_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_spd" << params.speed << "_if"
         << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateMediumPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "MediumPerformance";

    // FIXED: Time-based
    params.targetDecisions = 999999;

    // SNR range: 15-22 dB (rates 2-4)
    std::vector<std::pair<double, double>> snrRanges = {{15.0, 17.0},
                                                        {16.0, 18.0},
                                                        {17.0, 19.0},
                                                        {18.0, 20.0},
                                                        {19.0, 22.0}};

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 8) / 8.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.5 + (index % 6);
    params.interferers = (index % 4 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {512, 768, 1024, 1280};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"10Mbps", "15Mbps", "20Mbps", "25Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Medium_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_if" << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateGoodPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "GoodPerformance";

    // FIXED: Time-based
    params.targetDecisions = 999999;

    // SNR range: 22-30 dB (rates 4-5)
    std::vector<std::pair<double, double>> snrRanges = {{22.0, 24.0},
                                                        {23.0, 25.0},
                                                        {24.0, 26.0},
                                                        {25.0, 28.0},
                                                        {26.0, 30.0}};

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0 + (index % 3);
    params.interferers = (index % 5 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {768, 1024, 1280, 1500};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"25Mbps", "30Mbps", "35Mbps", "40Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Good_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateExcellentPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "ExcellentPerformance";

    // FIXED: Time-based
    params.targetDecisions = 999999;

    // SNR range: 30-45 dB (rates 5-7)
    std::vector<std::pair<double, double>> snrRanges = {
        {30.0, 33.0}, // Rate 5-6
        {32.0, 36.0}, // Rate 6
        {34.0, 38.0}, // Rate 6-7
        {36.0, 40.0}, // Rate 7
        {38.0, 45.0}  // High rate 7
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0;
    params.interferers = 0;
    params.packetSize = 1500;

    std::vector<std::string> trafficRates = {"40Mbps", "48Mbps", "54Mbps", "60Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Excellent_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_" << params.trafficRate;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateRandomChaosScenario(uint32_t index)
{
    static std::mt19937 rng(12345 + index);

    std::uniform_real_distribution<double> snrDist(8.0, 45.0);
    double targetSnr = snrDist(rng);
    double minSnr = std::max(8.0, targetSnr - 3.0);
    double maxSnr = std::min(45.0, targetSnr + 3.0);

    // Match traffic rate to SNR
    std::string trafficRate;
    if (targetSnr < 12.0)
        trafficRate = "3Mbps";
    else if (targetSnr < 18.0)
        trafficRate = "8Mbps";
    else if (targetSnr < 25.0)
        trafficRate = "18Mbps";
    else if (targetSnr < 32.0)
        trafficRate = "35Mbps";
    else
        trafficRate = "54Mbps";

    std::uniform_real_distribution<double> speedDist(0.0, 15.0);
    std::uniform_int_distribution<uint32_t> interfererDist(0, 5);
    std::uniform_int_distribution<uint32_t> pktDist(256, 1500);

    ScenarioParams params;
    params.category = "RandomChaos";

    // FIXED: Time-based
    params.targetDecisions = 999999;

    params.targetSnrMin = minSnr;
    params.targetSnrMax = maxSnr;
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    params.speed = speedDist(rng);
    params.interferers = interfererDist(rng);
    params.packetSize = (pktDist(rng) / 256) * 256;
    params.trafficRate = trafficRate;

    std::ostringstream name;
    name << "RandomChaos_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_spd" << params.speed << "_if"
         << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateHighInterferenceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "HighInterference";
    params.targetDecisions = 999999;

    std::vector<std::pair<double, double>> snrRanges = {{18.0, 24.0},
                                                        {20.0, 26.0},
                                                        {19.0, 25.0},
                                                        {17.0, 23.0}};

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 1.0 + (index % 6);
    params.interferers = 2 + (index % 2);

    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"12Mbps", "18Mbps", "25Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "HighInt_" << std::setfill('0') << std::setw(3) << index << "_if" << params.interferers
         << "_spd" << params.speed;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateNearIdealScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "NearIdeal";
    params.targetDecisions = 999999;

    double minSNR = 36.0;
    double maxSNR = 40.0;
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0;
    params.interferers = 0;
    params.packetSize = 1500;
    std::vector<std::string> rates = {"54Mbps", "60Mbps"};
    params.trafficRate = rates[index % rates.size()];

    std::ostringstream name;
    name << "NearIdeal_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_" << params.trafficRate;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateExtremeScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "Extreme";
    params.targetDecisions = 999999;

    double minSNR = 6.0;
    double maxSNR = 10.0;
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 3) / 3.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 4);

    params.speed = 10.0 + (index % 8);
    params.interferers = 4 + (index % 3);
    std::vector<uint32_t> pktSizes = {256, 512};
    params.packetSize = pktSizes[index % pktSizes.size()];
    params.trafficRate = "2Mbps";

    std::ostringstream name;
    name << "Extreme_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_if" << params.interferers << "_spd"
         << params.speed;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateEdgeStressScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "EdgeStress";
    params.targetDecisions = 999999;

    double minSNR = 12.0;
    double maxSNR = 16.0;
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 2);

    params.speed = 5.0 + (index % 6);
    params.interferers = 2;
    std::vector<uint32_t> pktSizes = {512, 768, 1024};
    params.packetSize = pktSizes[index % pktSizes.size()];
    params.trafficRate = "5Mbps";

    std::ostringstream name;
    name << "EdgeStress_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_spd" << params.speed;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateForceHighRateScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "ForceHighRate";
    params.targetDecisions = 999999;

    params.targetSnrMin = 38.0;
    params.targetSnrMax = 45.0;

    double targetSnr =
        params.targetSnrMin + (params.targetSnrMax - params.targetSnrMin) * ((index % 5) / 5.0);
    params.distance = 0.5 + (index % 3) * 0.5;

    params.speed = 0.0;
    params.interferers = 0;
    params.packetSize = 1500;

    std::vector<std::string> trafficRates = {"54Mbps", "60Mbps", "65Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "ForceHighRate_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_" << params.trafficRate;
    params.scenarioName = name.str();

    return params;
}

double
PerformanceBasedParameterGenerator::CalculateDistanceForSnr(double targetSnr, uint32_t interferers)
{
    double adjustedSnr = targetSnr + (interferers * 2.0);

    double distance;
    if (adjustedSnr >= 35.0)
        distance = 1.0;
    else if (adjustedSnr > 19.0)
        distance = (35.0 - adjustedSnr) / 0.8;
    else if (adjustedSnr > 4.0)
        distance = 20.0 + (19.0 - adjustedSnr) / 0.5;
    else
        distance = 50.0 + (4.0 - adjustedSnr) / 0.3;

    return std::clamp(distance, 1.0, 80.0);
}

} // namespace ns3