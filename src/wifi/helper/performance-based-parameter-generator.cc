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

    // Adjusted distribution to ensure better data collection
    // Category A: Poor Performance Scenarios (25% - reduced for better overall success)
    uint32_t categoryA = totalScenarios * 0.25;
    for (uint32_t i = 0; i < categoryA; ++i)
    {
        scenarios.push_back(GeneratePoorPerformanceScenario(i));
    }

    // Category B: Medium Performance/Boundary Scenarios (45% - increased)
    uint32_t categoryB = totalScenarios * 0.45;
    for (uint32_t i = 0; i < categoryB; ++i)
    {
        scenarios.push_back(GenerateMediumPerformanceScenario(i));
    }

    // Category C: High Interference/Stress Testing (15%)
    uint32_t categoryC = totalScenarios * 0.15;
    for (uint32_t i = 0; i < categoryC; ++i)
    {
        scenarios.push_back(GenerateHighInterferenceScenario(i));
    }

    // Category D: Good Performance/Baseline (15% - increased)
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

    // Reduced targets to more achievable levels
    params.targetDecisions = 100 + (index % 200); // 100-300 decisions

    // More achievable SNR ranges - still poor but functional
    std::vector<std::pair<double, double>> snrRanges = {
        {10.0, 15.0}, // Low but workable
        {12.0, 18.0}, // Poor-medium transition
        {8.0, 14.0},  // Variable poor range
        {11.0, 16.0}, // Overlapping range
        {9.0, 13.0}   // Consistently poor
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Much reduced mobility
    params.speed = 0.5 + (index % 4); // 0.5-4 m/s

    // Minimal interference
    params.interferers = (index % 3 == 0) ? 0 : 1; // 0-1 interferers

    // Conservative packet sizes
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Very conservative traffic rates
    std::vector<std::string> trafficRates = {"5Mbps", "8Mbps", "10Mbps", "12Mbps"};
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

    // Moderate targets
    params.targetDecisions = 150 + (index % 250); // 150-400 decisions

    // Solid medium SNR ranges
    std::vector<std::pair<double, double>> snrRanges = {
        {15.0, 20.0}, // Medium-low
        {18.0, 23.0}, // Medium
        {16.0, 21.0}, // Medium range
        {17.0, 22.0}, // Overlapping medium
        {14.0, 19.0}, // Lower medium
        {19.0, 24.0}  // Higher medium
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 8) / 8.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Low to moderate mobility
    params.speed = 0.5 + (index % 5); // 0.5-5 m/s

    // Limited interference
    params.interferers = (index % 4 == 0) ? 0 : 1; // Mostly 1, sometimes 0

    // Standard packet sizes
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024, 1280};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Moderate traffic rates
    std::vector<std::string> trafficRates = {"10Mbps", "15Mbps", "20Mbps", "25Mbps"};
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

    // Lower targets due to interference challenges
    params.targetDecisions = 80 + (index % 150); // 80-230 decisions

    // Good SNR to compensate for high interference
    std::vector<std::pair<double, double>> snrRanges = {
        {18.0, 25.0}, // Good SNR with interference
        {20.0, 27.0}, // Very good SNR with interference
        {16.0, 23.0}, // Decent SNR with interference
        {19.0, 24.0}  // Stable good SNR
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Low mobility to help with interference
    params.speed = 1.0 + (index % 6); // 1-7 m/s

    // Significant but not overwhelming interference
    params.interferers = 2 + (index % 2); // 2-3 interferers

    // Smaller packet sizes for better success in interference
    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Moderate traffic rates
    std::vector<std::string> trafficRates = {"15Mbps", "20Mbps", "25Mbps"};
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

    // Higher targets for good performance scenarios
    params.targetDecisions = 200 + (index % 300); // 200-500 decisions

    // High SNR ranges for excellent performance
    std::vector<std::pair<double, double>> snrRanges = {
        {22.0, 28.0}, // Good
        {25.0, 30.0}, // Very good
        {20.0, 26.0}, // Good-medium high
        {24.0, 29.0}, // Excellent
        {21.0, 27.0}  // Consistently good
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr);

    // Very low mobility for stable performance
    params.speed = 0.0 + (index % 3); // 0-2 m/s (including stationary)

    // No or minimal interference
    params.interferers = (index % 5 == 0) ? 1 : 0; // Mostly 0, occasionally 1

    // Optimal packet sizes
    std::vector<uint32_t> packetSizes = {512, 768, 1024, 1280};
    params.packetSize = packetSizes[index % packetSizes.size()];

    // Higher traffic rates for good performance scenarios
    std::vector<std::string> trafficRates = {"20Mbps", "25Mbps", "30Mbps", "35Mbps"};
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

    // More conservative distance limits for better connectivity
    return std::max(1.0, std::min(80.0, distance)); // Max 80m for reliable communication
}

} // namespace ns3