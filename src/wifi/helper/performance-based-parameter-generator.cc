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

    // Rebalanced for more challenging scenarios
    // Category A: Poor Performance Scenarios (45% - increased from 40%)
    uint32_t categoryA = totalScenarios * 0.45;
    for (uint32_t i = 0; i < categoryA; ++i)
    {
        scenarios.push_back(GeneratePoorPerformanceScenario(i));
    }

    // Category B: Medium Performance/Boundary Scenarios (30%)
    uint32_t categoryB = totalScenarios * 0.30;
    for (uint32_t i = 0; i < categoryB; ++i)
    {
        scenarios.push_back(GenerateMediumPerformanceScenario(i));
    }

    // Category C: High Interference/Stress Testing (15% - reduced from 20%)
    uint32_t categoryC = totalScenarios * 0.15;
    for (uint32_t i = 0; i < categoryC; ++i)
    {
        scenarios.push_back(GenerateHighInterferenceScenario(i));
    }

    // Category D: Good Performance/Baseline (10% - reduced from 10%)
    uint32_t categoryD = totalScenarios - (categoryA + categoryB + categoryC);
    for (uint32_t i = 0; i < categoryD; ++i)
    {
        scenarios.push_back(GenerateGoodPerformanceScenario(i));
    }

    return scenarios;
}

ScenarioParams
PerformanceBasedParameterGenerator::GeneratePoorPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "PoorPerformance";
    params.targetDecisions = 800 + (index % 400); // 800-1200 decisions

    // Very low SNR ranges for poor performance
    std::vector<std::pair<double, double>> snrRanges = {
        {3.0, 6.0},  // Very poor
        {6.0, 9.0},  // Poor
        {9.0, 12.0}, // Below average
        {5.0, 8.0},  // Overlapping poor range
        {7.0, 11.0}  // Mixed poor-medium
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // High mobility for more challenging conditions
    params.speed = 5.0 + (index % 15); // 5-20 m/s

    // More interferers for challenging scenarios
    params.interferers = 2 + (index % 4); // 2-5 interferers

    // Varied packet sizes favoring larger packets (more challenging)
    std::vector<uint32_t> packetSizes = {512, 1024, 1536, 2048, 2560};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Higher traffic rates to stress the system
    std::vector<std::string> trafficRates = {"35Mbps", "40Mbps", "45Mbps", "50Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Poor_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_spd" << params.speed;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateMediumPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "MediumPerformance";
    params.targetDecisions = 1000 + (index % 500); // 1000-1500 decisions

    // Medium SNR ranges - transition zones
    std::vector<std::pair<double, double>> snrRanges = {
        {12.0, 16.0}, // Medium-low
        {15.0, 19.0}, // Medium
        {18.0, 22.0}, // Medium-high
        {14.0, 17.0}, // Overlapping medium-low
        {16.0, 20.0}, // Overlapping medium
        {13.0, 18.0}  // Wide medium range
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 8) / 8.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Moderate mobility
    params.speed = 2.0 + (index % 8); // 2-10 m/s

    // Moderate interference
    params.interferers = 1 + (index % 3); // 1-3 interferers

    // Mixed packet sizes
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024, 1280};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Moderate traffic rates
    std::vector<std::string> trafficRates = {"25Mbps", "30Mbps", "35Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Medium_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_if" << params.interferers;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateHighInterferenceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "HighInterference";
    params.targetDecisions =
        600 + (index % 400); // 600-1000 decisions (fewer due to harsh conditions)

    // Wide SNR range but focus on interference impact
    std::vector<std::pair<double, double>> snrRanges = {
        {8.0, 15.0},  // Poor to medium with high interference
        {12.0, 20.0}, // Medium with high interference
        {6.0, 18.0},  // Wide range with interference
        {10.0, 16.0}  // Focused medium range
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // High mobility for additional stress
    params.speed = 8.0 + (index % 12); // 8-20 m/s

    // Very high interference - this is the key distinguishing factor
    params.interferers = 4 + (index % 4); // 4-7 interferers

    // Large packet sizes to increase collision probability
    std::vector<uint32_t> packetSizes = {1024, 1536, 2048, 2560, 3072};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // High traffic rates to maximize interference
    std::vector<std::string> trafficRates = {"40Mbps", "45Mbps", "50Mbps", "55Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "HighInt_" << std::setfill('0') << std::setw(3) << index << "_if" << params.interferers
         << "_spd" << params.speed;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateGoodPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "GoodPerformance";
    params.targetDecisions = 1200 + (index % 300); // 1200-1500 decisions

    // High SNR ranges for good performance
    std::vector<std::pair<double, double>> snrRanges = {
        {22.0, 28.0}, // Good
        {25.0, 30.0}, // Very good
        {20.0, 25.0}, // Good-medium
        {23.0, 27.0}  // Stable good
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Low to moderate mobility
    params.speed = 0.5 + (index % 6); // 0.5-6 m/s

    // Low interference
    params.interferers = (index % 2) ? 1 : 2; // 1-2 interferers

    // Smaller packet sizes for better performance
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Moderate traffic rates
    std::vector<std::string> trafficRates = {"20Mbps", "25Mbps", "30Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Good_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr;
    params.scenarioName = name.str();

    return params;
}

double
PerformanceBasedParameterGenerator::CalculateDistanceForSnr(double targetSnr)
{
    const double TX_POWER_DBM = 20.0;
    const double NOISE_FLOOR_DBM = -90.0;
    const double FREQUENCY_OFFSET = 32.44 + 20.0 * log10(2400.0);

    double requiredPathLoss = TX_POWER_DBM - NOISE_FLOOR_DBM - targetSnr;
    double logDistance = (requiredPathLoss - FREQUENCY_OFFSET) / 40.0;
    double distance = pow(10.0, logDistance);

    return std::max(1.0, std::min(150.0, distance)); // Extended max distance to 150m
}

} // namespace ns3