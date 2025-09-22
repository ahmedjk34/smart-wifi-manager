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

    // Rebalanced for more challenging scenarios but with guaranteed data collection
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
    params.targetDecisions = 600 + (index % 300); // 600-900 decisions (reduced from 800-1200)

    // Adjusted SNR ranges - still poor but not so harsh that no packets get through
    std::vector<std::pair<double, double>> snrRanges = {
        {8.0, 12.0},  // Poor but workable (raised from 3.0-6.0)
        {10.0, 14.0}, // Poor-medium transition (raised from 6.0-9.0)
        {12.0, 16.0}, // Below average but functional (raised from 9.0-12.0)
        {9.0, 13.0},  // Overlapping poor range (raised from 5.0-8.0)
        {11.0, 15.0}  // Mixed poor-medium (raised from 7.0-11.0)
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Reduced mobility for better connectivity
    params.speed = 2.0 + (index % 8); // 2-10 m/s (reduced from 5-20)

    // Fewer interferers to allow some packets through
    params.interferers = 1 + (index % 2); // 1-2 interferers (reduced from 2-5)

    // Smaller packet sizes for better success rate
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024, 1280}; // Reduced max size
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Reduced traffic rates to prevent overwhelming the poor channel
    std::vector<std::string> trafficRates = {"15Mbps",
                                             "20Mbps",
                                             "25Mbps",
                                             "30Mbps"}; // Reduced from 35-50
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
    params.targetDecisions = 800 + (index % 400); // 800-1200 decisions (reduced from 1000-1500)

    // Medium SNR ranges - adjusted for reliable packet transmission
    std::vector<std::pair<double, double>> snrRanges = {
        {15.0, 19.0}, // Medium-low (raised from 12.0-16.0)
        {17.0, 21.0}, // Medium (raised from 15.0-19.0)
        {19.0, 23.0}, // Medium-high (raised from 18.0-22.0)
        {16.0, 20.0}, // Overlapping medium-low (raised from 14.0-17.0)
        {18.0, 22.0}, // Overlapping medium (raised from 16.0-20.0)
        {14.0, 20.0}  // Wide medium range (slightly raised from 13.0-18.0)
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 8) / 8.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Reduced mobility for more stable connections
    params.speed = 1.0 + (index % 6); // 1-7 m/s (reduced from 2-10)

    // Controlled interference
    params.interferers = 1 + (index % 2); // 1-2 interferers (reduced from 1-3)

    // Conservative packet sizes
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024}; // Reduced max size
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Moderate traffic rates
    std::vector<std::string> trafficRates = {"20Mbps", "25Mbps", "30Mbps"}; // Reduced from 25-35
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
    params.targetDecisions = 500 + (index % 300); // 500-800 decisions (reduced from 600-1000)

    // Better SNR ranges to compensate for high interference
    std::vector<std::pair<double, double>> snrRanges = {
        {12.0, 18.0}, // Decent SNR with high interference (raised from 8.0-15.0)
        {15.0, 22.0}, // Good SNR with high interference (raised from 12.0-20.0)
        {10.0, 20.0}, // Wide range with interference (raised from 6.0-18.0)
        {14.0, 19.0}  // Focused range (raised from 10.0-16.0)
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Reduced mobility to help with interference challenges
    params.speed = 3.0 + (index % 8); // 3-11 m/s (reduced from 8-20)

    // High interference but not overwhelming
    params.interferers = 2 + (index % 3); // 2-4 interferers (reduced from 4-7)

    // Smaller packet sizes to improve success rate in interference
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024, 1280}; // Reduced max size
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Controlled traffic rates despite high interference
    std::vector<std::string> trafficRates = {"25Mbps", "30Mbps", "35Mbps"}; // Reduced from 40-55
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
    params.targetDecisions = 1000 + (index % 400); // 1000-1400 decisions (reduced from 1200-1500)

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

    // Low mobility for stable performance
    params.speed = 0.5 + (index % 4); // 0.5-4 m/s (reduced from 0.5-6)

    // Minimal interference
    params.interferers = (index % 2) ? 0 : 1; // 0-1 interferers (reduced from 1-2)

    // Optimal packet sizes for good performance
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Conservative traffic rates for reliable performance
    std::vector<std::string> trafficRates = {"15Mbps", "20Mbps", "25Mbps"}; // Reduced from 20-30
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

    return std::max(
        1.0,
        std::min(100.0, distance)); // Reduced max distance to 100m for better connectivity
}

} // namespace ns3