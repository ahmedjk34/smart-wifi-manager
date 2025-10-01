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

    // Redistribute based on your target class distribution analysis
    // Focus on balancing the underrepresented high-rate classes
    uint32_t catPoor = std::round(totalScenarios * 0.25);          // Reduced from 30% to 25%
    uint32_t catMedium = std::round(totalScenarios * 0.25);        // Reduced from 35% to 25%
    uint32_t catHighInt = std::round(totalScenarios * 0.15);       // Keep 15%
    uint32_t catGood = std::round(totalScenarios * 0.12);          // Increased from 7% to 12%
    uint32_t catExcellent = std::round(totalScenarios * 0.10);     // Increased from 5% to 10%
    uint32_t catNearIdeal = std::round(totalScenarios * 0.03);     // Keep 3%
    uint32_t catExtreme = std::round(totalScenarios * 0.05);       // Increased from 3% to 5%
    uint32_t catEdgeStress = std::round(totalScenarios * 0.03);    // Increased from 2% to 3%
    uint32_t catForceHighRate = std::round(totalScenarios * 0.02); // NEW: 2% for high rates

    // Rebalance to ensure total = totalScenarios
    uint32_t sum = catPoor + catMedium + catHighInt + catGood + catExcellent + catNearIdeal +
                   catExtreme + catEdgeStress + catForceHighRate;

    // Adjust the largest group to match total
    if (sum < totalScenarios)
        catMedium += (totalScenarios - sum);
    else if (sum > totalScenarios)
        catMedium -= (sum - totalScenarios);

    // Generate scenarios with the new distribution
    for (uint32_t i = 0; i < catPoor; ++i)
        scenarios.push_back(GeneratePoorPerformanceScenario(i));
    for (uint32_t i = 0; i < catMedium; ++i)
        scenarios.push_back(GenerateMediumPerformanceScenario(i));
    for (uint32_t i = 0; i < catHighInt; ++i)
        scenarios.push_back(GenerateHighInterferenceScenario(i));
    for (uint32_t i = 0; i < catGood; ++i)
        scenarios.push_back(GenerateGoodPerformanceScenario(i));
    for (uint32_t i = 0; i < catExcellent; ++i)
        scenarios.push_back(GenerateExcellentPerformanceScenario(i));
    for (uint32_t i = 0; i < catNearIdeal; ++i)
        scenarios.push_back(GenerateNearIdealScenario(i));
    for (uint32_t i = 0; i < catExtreme; ++i)
        scenarios.push_back(GenerateExtremeScenario(i));
    for (uint32_t i = 0; i < catEdgeStress; ++i)
        scenarios.push_back(GenerateEdgeStressScenario(i));
    for (uint32_t i = 0; i < catForceHighRate; ++i) // NEW: Include high-rate scenarios
        scenarios.push_back(GenerateForceHighRateScenario(i));

    // Add some random chaos scenarios for diversity (5% of total)
    uint32_t chaosScenarios = std::round(totalScenarios * 0.05);
    for (uint32_t i = 0; i < chaosScenarios; ++i)
        scenarios.push_back(GenerateRandomChaosScenario(i));

    return scenarios;
}

ScenarioParams
PerformanceBasedParameterGenerator::GeneratePoorPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "PoorPerformance";
    params.targetDecisions = 80 + (index % 120); // 80-200 decisions

    // SNR range: 8-15 dB (functional but challenging)
    std::vector<std::pair<double, double>> snrRanges = {
        {8.0, 12.0},  // Very poor but functional
        {10.0, 14.0}, // Poor
        {9.0, 13.0},  // Consistently poor
        {11.0, 15.0}, // Poor-medium boundary
        {8.5, 12.5}   // Low variability poor
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.5 + (index % 4);              // 0.5-4 m/s
    params.interferers = (index % 3 == 0) ? 1 : 0; // Mostly 0, sometimes 1

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
    params.targetDecisions = 150 + (index % 200); // 150-350 decisions

    // SNR range: 15-22 dB (moderate performance)
    std::vector<std::pair<double, double>> snrRanges = {
        {15.0, 19.0}, // Medium-low
        {16.0, 20.0}, // Medium
        {17.0, 21.0}, // Medium-high
        {15.5, 19.5}, // Stable medium
        {16.5, 22.0}, // Medium with variance
        {14.5, 18.5}  // Lower medium
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 8) / 8.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.5 + (index % 6);              // 0.5-6 m/s
    params.interferers = (index % 4 == 0) ? 1 : 0; // Mostly 0, occasionally 1

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
PerformanceBasedParameterGenerator::GenerateHighInterferenceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "HighInterference";
    params.targetDecisions = 100 + (index % 150); // 100-250 decisions

    // Good base SNR (18-26 dB) to compensate for interference
    std::vector<std::pair<double, double>> snrRanges = {
        {18.0, 24.0}, // Good with interference
        {20.0, 26.0}, // Very good with interference
        {19.0, 25.0}, // Stable good
        {17.0, 23.0}  // Decent with interference
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 1.0 + (index % 6);     // 1-7 m/s
    params.interferers = 2 + (index % 2); // 2-3 interferers

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
PerformanceBasedParameterGenerator::GenerateGoodPerformanceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "GoodPerformance";
    params.targetDecisions = 200 + (index % 300); // 200-500 decisions

    // SNR range: 22-30 dB (good performance)
    std::vector<std::pair<double, double>> snrRanges = {
        {22.0, 27.0}, // Good
        {24.0, 29.0}, // Very good
        {23.0, 28.0}, // Consistently good
        {25.0, 30.0}, // High good
        {21.0, 26.0}  // Lower good
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0 + (index % 3);              // 0-2 m/s (including stationary)
    params.interferers = (index % 5 == 0) ? 1 : 0; // Mostly 0, rarely 1

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
    params.targetDecisions = 300 + (index % 400); // 300-700 decisions

    // SNR range: 30-40 dB (excellent/ideal)
    std::vector<std::pair<double, double>> snrRanges = {
        {30.0, 35.0}, // Excellent
        {32.0, 38.0}, // Near-ideal
        {31.0, 36.0}, // Consistently excellent
        {33.0, 40.0}  // Ideal
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0; // Stationary for maximum stability
    params.interferers = 0;

    params.packetSize = 1500; // Max MTU

    std::vector<std::string> trafficRates = {"40Mbps", "48Mbps", "54Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "Excellent_" << std::setfill('0') << std::setw(3) << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_" << params.trafficRate;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateNearIdealScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "NearIdeal";
    params.targetDecisions = 500 + (index % 300); // 500-800 decisions

    // SNR: Very high, stable (36-40 dB)
    double minSNR = 36.0;
    double maxSNR = 40.0;
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0; // Stationary
    params.interferers = 0;
    params.packetSize = 1500; // Max MTU
    std::vector<std::string> rates = {"54Mbps", "60Mbps"};
    params.trafficRate = rates[index % rates.size()];

    std::ostringstream name;
    name << "NearIdeal_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_" << params.trafficRate;
    params.scenarioName = name.str();

    return params;
}

// Extreme scenario: SNR at margin, high speed, lots of interferers, challenging
ScenarioParams
PerformanceBasedParameterGenerator::GenerateExtremeScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "Extreme";
    params.targetDecisions = 50 + (index % 50); // 50-100 decisions

    // SNR: At the lower boundary (6-10 dB)
    double minSNR = 6.0;
    double maxSNR = 10.0;
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 3) / 3.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 4); // Many interferers

    params.speed = 10.0 + (index % 8);    // 10-17 m/s (fast)
    params.interferers = 4 + (index % 3); // 4-6 interferers
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

// Edge-stress: SNR at edge, lots of mobility, moderate interferers, test robustness
ScenarioParams
PerformanceBasedParameterGenerator::GenerateEdgeStressScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "EdgeStress";
    params.targetDecisions = 80 + (index % 80); // 80-160 decisions

    // SNR: 12-16 dB (functional edge)
    double minSNR = 12.0;
    double maxSNR = 16.0;
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 2);

    params.speed = 5.0 + (index % 6); // 5-10 m/s
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

// Random chaos: Randomized all parameters, for stress/fuzz testing
ScenarioParams
PerformanceBasedParameterGenerator::GenerateRandomChaosScenario(uint32_t index)
{
    static std::mt19937 rng(12345 + index);

    // Step 1: Select SNR
    std::uniform_real_distribution<double> snrDist(8.0, 40.0);
    double minSnr = snrDist(rng);
    double maxSnr = std::min(40.0, minSnr + snrDist(rng) * 0.25);
    double targetSnr = (minSnr + maxSnr) / 2.0;

    // Step 2: Select appropriate traffic rates for SNR
    std::vector<std::string> possibleRates;
    if (targetSnr < 13.0)
    {
        possibleRates = {"2Mbps", "5Mbps"};
    }
    else if (targetSnr < 18.0)
    {
        possibleRates = {"5Mbps", "10Mbps"};
    }
    else if (targetSnr < 25.0)
    {
        possibleRates = {"10Mbps", "20Mbps", "25Mbps"};
    }
    else if (targetSnr < 32.0)
    {
        possibleRates = {"20Mbps", "25Mbps", "35Mbps", "40Mbps"};
    }
    else
    {
        possibleRates = {"40Mbps", "48Mbps", "54Mbps"};
    }
    std::uniform_int_distribution<size_t> rateIdx(0, possibleRates.size() - 1);
    std::string trafficRate = possibleRates[rateIdx(rng)];

    // Step 3: Restrict packet sizes for low SNR
    std::vector<uint32_t> pktSizes;
    if (targetSnr < 13.0)
    {
        pktSizes = {256, 512};
    }
    else if (targetSnr < 18.0)
    {
        pktSizes = {256, 512, 768};
    }
    else if (targetSnr < 25.0)
    {
        pktSizes = {512, 768, 1024};
    }
    else
    {
        pktSizes = {768, 1024, 1280, 1500};
    }
    std::uniform_int_distribution<size_t> pktIdx(0, pktSizes.size() - 1);
    uint32_t packetSize = pktSizes[pktIdx(rng)];

    // Step 4: Other parameters random but sensible
    std::uniform_real_distribution<double> speedDist(0.0, 15.0);
    std::uniform_int_distribution<uint32_t> interfererDist(0, 6);

    ScenarioParams params;
    params.category = "RandomChaos";
    params.targetSnrMin = minSnr;
    params.targetSnrMax = maxSnr;
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    params.speed = speedDist(rng);
    params.interferers = interfererDist(rng);
    params.packetSize = packetSize;
    params.trafficRate = trafficRate;
    params.targetDecisions = 50 + (rng() % 701); // 50-750

    std::ostringstream name;
    name << "RandomChaos_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_spd" << params.speed << "_if"
         << params.interferers;
    params.scenarioName = name.str();

    return params;
}

double
PerformanceBasedParameterGenerator::CalculateDistanceForSnr(double targetSnr, uint32_t interferers)
{
    // Compensate for interferer penalty in SNR calculation
    double adjustedSnr = targetSnr + (interferers * 2.0);

    // Inverse of the SOFT_MODEL in ConvertNS3ToRealisticSnr
    double distance;
    if (adjustedSnr >= 35.0)
        distance = 1.0;
    else if (adjustedSnr > 19.0)
        distance = (35.0 - adjustedSnr) / 0.8;
    else if (adjustedSnr > 4.0)
        distance = 20.0 + (19.0 - adjustedSnr) / 0.5;
    else
        distance = 50.0 + (4.0 - adjustedSnr) / 0.3;

    // Clamp to reasonable ranges for ns-3
    return std::clamp(distance, 1.0, 80.0);
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateForceHighRateScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "ForceHighRate";
    params.targetDecisions = 400 + (index % 300); // 400-700 decisions

    // Perfect conditions for high rates
    params.targetSnrMin = 38.0;
    params.targetSnrMax = 45.0;

    double targetSnr =
        params.targetSnrMin + (params.targetSnrMax - params.targetSnrMin) * ((index % 5) / 5.0);
    params.distance = 0.5 + (index % 3) * 0.5; // 0.5-2.0m (very close)

    params.speed = 0.0;       // Stationary
    params.interferers = 0;   // No interference
    params.packetSize = 1500; // Maximum MTU

    // Very high traffic to force aggressive rate selection
    std::vector<std::string> trafficRates = {"54Mbps", "60Mbps", "65Mbps"};
    params.trafficRate = trafficRates[index % trafficRates.size()];

    std::ostringstream name;
    name << "ForceHighRate_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_" << params.trafficRate;
    params.scenarioName = name.str();

    return params;
}

} // namespace ns3