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

    // ✅ OPTIMIZED: Shifted towards better scenarios for balanced training data
    // This compensates for File 1b's power law balancing (POWER=0.5)
    // which heavily downsamples high-SNR scenarios
    uint32_t catPoor = std::round(totalScenarios * 0.10);      // 10% Poor (was 20%)
    uint32_t catMedium = std::round(totalScenarios * 0.15);    // 15% Medium (was 20%)
    uint32_t catGood = std::round(totalScenarios * 0.30);      // 30% Good (was 20%)
    uint32_t catExcellent = std::round(totalScenarios * 0.30); // 30% Excellent (was 20%)
    uint32_t catChaos = std::round(totalScenarios * 0.15);     // 15% Chaos (was 20%)

    // Adjust to match total
    uint32_t sum = catPoor + catMedium + catGood + catExcellent + catChaos;
    if (sum < totalScenarios)
        catExcellent += (totalScenarios - sum);
    else if (sum > totalScenarios)
        catExcellent -= (sum - totalScenarios);

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

    // Time-based termination
    params.targetDecisions = 999999;

    // ✅ FIXED: Increased SNR targets (better initial distances)
    // SNR range: 10-17 dB (rates 0-3) - was 8-15 dB
    std::vector<std::pair<double, double>> snrRanges = {
        {10.0, 12.0}, // Rate 0-1 (was 8-10)
        {11.0, 13.0},
        {12.0, 14.0}, // Rate 1-2 (was 10-12)
        {13.0, 15.0},
        {14.0, 17.0} // Rate 2-3 (was 12-15)
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 10) / 10.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    // Expected: 25-40m (was 30-50m)

    // ✅ FIXED: Reduced speed to prevent hitting 80m cap
    params.speed = 0.0 + (index % 4) * 0.3; // 0.0-0.9 m/s (was 0.5-3.5)
    params.interferers = (index % 3 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {256, 512, 768, 1024};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"5Mbps", "8Mbps", "10Mbps", "12Mbps"};
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

    params.targetDecisions = 999999;

    // ✅ FIXED: Increased SNR targets
    // SNR range: 17-24 dB (rates 3-5) - was 15-22 dB
    std::vector<std::pair<double, double>> snrRanges = {
        {17.0, 19.0}, // Was {15.0, 17.0}
        {18.0, 20.0}, // Was {16.0, 18.0}
        {19.0, 21.0}, // Was {17.0, 19.0}
        {20.0, 22.0}, // Was {18.0, 20.0}
        {21.0, 24.0}  // Was {19.0, 22.0}
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 8) / 8.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    // Expected: 10-20m (was 15-30m)

    // ✅ FIXED: Reduced speed
    params.speed = 0.0 + (index % 6) * 0.4; // 0.0-2.0 m/s (was 0.5-5.5)
    params.interferers = (index % 4 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {768, 1024, 1280, 1500};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"15Mbps", "20Mbps", "25Mbps", "30Mbps"};
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

    params.targetDecisions = 999999;

    // ✅ SLIGHTLY ADJUSTED: Better coverage of rate 5-6 range
    // SNR range: 24-32 dB (rates 5-6) - was 22-30 dB
    std::vector<std::pair<double, double>> snrRanges = {
        {24.0, 26.0}, // Was {22.0, 24.0}
        {25.0, 27.0}, // Was {23.0, 25.0}
        {26.0, 28.0}, // Was {24.0, 26.0}
        {27.0, 30.0}, // Was {25.0, 28.0}
        {28.0, 32.0}  // Was {26.0, 30.0}
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    // Expected: 2-8m

    // ✅ LOW MOBILITY: Keep scenarios near optimal distance
    params.speed = 0.0 + (index % 3) * 0.2; // 0.0-0.4 m/s (was 0.0-2.0)
    params.interferers = (index % 5 == 0) ? 1 : 0;

    std::vector<uint32_t> packetSizes = {1024, 1280, 1500};
    params.packetSize = packetSizes[index % packetSizes.size()];

    std::vector<std::string> trafficRates = {"30Mbps", "35Mbps", "40Mbps", "45Mbps"};
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

    params.targetDecisions = 999999;

    // ✅ OPTIMIZED: Better coverage of rate 7 range
    // SNR range: 32-45 dB (rate 7) - was 30-45 dB
    std::vector<std::pair<double, double>> snrRanges = {
        {32.0, 35.0}, // Rate 7 (was 30-33)
        {34.0, 37.0}, // Rate 7 (was 32-36)
        {36.0, 39.0}, // Rate 7 (was 34-38)
        {38.0, 42.0}, // Rate 7 (was 36-40)
        {40.0, 45.0}  // Rate 7 (was 38-45)
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    // Expected: 0.5-3m

    // STATIC: No mobility for excellent scenarios
    params.speed = 0.0;
    params.interferers = 0;
    params.packetSize = 1500;

    std::vector<std::string> trafficRates = {"48Mbps", "52Mbps", "54Mbps"};
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

    // ✅ FIXED: Increased lower bound + triangular distribution
    // Triangular distribution peaks at mid-range SNR (25 dB)
    std::vector<double> snrBreakpoints{10.0, 25.0, 45.0}; // SNR breakpoints (was uniform 8-45)
    std::vector<double> snrWeights{0.5, 1.0, 0.3};        // Weights: low at extremes, high at 25 dB
    std::piecewise_linear_distribution<double> snrDist(snrBreakpoints.begin(),
                                                       snrBreakpoints.end(),
                                                       snrWeights.begin());
    double targetSnr = snrDist(rng);

    double minSnr = std::max(10.0, targetSnr - 3.0);
    double maxSnr = std::min(45.0, targetSnr + 3.0);

    // Randomized traffic rate (realistic spread)
    std::vector<std::string> allRates = {"5Mbps",
                                         "8Mbps",
                                         "12Mbps",
                                         "18Mbps",
                                         "25Mbps",
                                         "30Mbps",
                                         "35Mbps",
                                         "40Mbps",
                                         "48Mbps",
                                         "54Mbps"};
    std::uniform_int_distribution<size_t> rateDist(0, allRates.size() - 1);
    std::string trafficRate = allRates[rateDist(rng)];

    // ✅ FIXED: Reduced max speed + lower interferers
    std::uniform_real_distribution<double> speedDist(0.0, 5.0);   // Was 15.0
    std::uniform_int_distribution<uint32_t> interfererDist(0, 3); // Was 0-5
    std::uniform_int_distribution<uint32_t> pktDist(256, 1500);

    ScenarioParams params;
    params.category = "RandomChaos";
    params.targetDecisions = 999999;

    params.targetSnrMin = minSnr;
    params.targetSnrMax = maxSnr;
    params.distance = CalculateDistanceForSnr(targetSnr, 0);
    // Expected: 5-60m (was 5-80m)

    params.speed = speedDist(rng);
    params.interferers = interfererDist(rng);
    params.packetSize = (pktDist(rng) / 256) * 256;
    params.trafficRate = trafficRate;

    std::ostringstream name;
    name << "RandomChaos_" << std::setw(3) << std::setfill('0') << index << "_snr" << std::fixed
         << std::setprecision(1) << targetSnr << "_spd" << params.speed << "_if"
         << params.interferers << "_" << trafficRate;
    params.scenarioName = name.str();

    return params;
}

ScenarioParams
PerformanceBasedParameterGenerator::GenerateHighInterferenceScenario(uint32_t index)
{
    ScenarioParams params;
    params.category = "HighInterference";
    params.targetDecisions = 999999;

    // ✅ SLIGHTLY INCREASED: Better coverage with interference
    std::vector<std::pair<double, double>> snrRanges = {
        {20.0, 24.0}, // Was {18.0, 24.0}
        {22.0, 26.0}, // Was {20.0, 26.0}
        {21.0, 25.0}, // Was {19.0, 25.0}
        {19.0, 23.0}  // Was {17.0, 23.0}
    };

    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;

    double targetSnr = range.first + (range.second - range.first) * ((index % 6) / 6.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    // ✅ FIXED: Reduced speed
    params.speed = 0.5 + (index % 4) * 0.4; // 0.5-2.1 m/s (was 1.0-6.0)
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

    // ✅ OPTIMIZED: Peak rate 7 performance
    double minSNR = 38.0; // Was 36.0
    double maxSNR = 45.0; // Was 40.0
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 5) / 5.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 0);

    params.speed = 0.0;
    params.interferers = 0;
    params.packetSize = 1500;

    std::vector<std::string> rates = {"52Mbps", "54Mbps"}; // Removed 60Mbps (exceeds 54)
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

    // ✅ SLIGHTLY ADJUSTED: Better coverage of worst-case scenarios
    double minSNR = 8.0;  // Was 6.0
    double maxSNR = 12.0; // Was 10.0
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 3) / 3.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 4);

    // ✅ FIXED: Reduced speed
    params.speed = 5.0 + (index % 6); // 5-10 m/s (was 10-17)
    params.interferers = 4 + (index % 3);

    std::vector<uint32_t> pktSizes = {256, 512};
    params.packetSize = pktSizes[index % pktSizes.size()];
    params.trafficRate = "3Mbps"; // Was 2Mbps

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

    // ✅ SLIGHTLY INCREASED: Better edge case coverage
    double minSNR = 14.0; // Was 12.0
    double maxSNR = 18.0; // Was 16.0
    params.targetSnrMin = minSNR;
    params.targetSnrMax = maxSNR;

    double targetSnr = minSNR + (maxSNR - minSNR) * ((index % 4) / 4.0);
    params.distance = CalculateDistanceForSnr(targetSnr, 2);

    // ✅ FIXED: Reduced speed
    params.speed = 2.0 + (index % 4); // 2-5 m/s (was 5-10)
    params.interferers = 2;

    std::vector<uint32_t> pktSizes = {512, 768, 1024};
    params.packetSize = pktSizes[index % pktSizes.size()];
    params.trafficRate = "8Mbps"; // Was 5Mbps

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

    // ✅ OPTIMIZED: Force rate 7 usage
    params.targetSnrMin = 40.0; // Was 38.0
    params.targetSnrMax = 45.0;

    double targetSnr =
        params.targetSnrMin + (params.targetSnrMax - params.targetSnrMin) * ((index % 5) / 5.0);
    params.distance = 0.5 + (index % 3) * 0.3; // 0.5-1.1m (was 0.5-1.5m)

    params.speed = 0.0;
    params.interferers = 0;
    params.packetSize = 1500;

    std::vector<std::string> trafficRates = {"50Mbps", "52Mbps", "54Mbps"};
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
    // ============================================================================
    // INVERSE SNR-TO-DISTANCE MAPPING
    // Compensates for interference subtraction at runtime
    // ============================================================================
    double compensatedSnr = targetSnr + (interferers * 2.0);

    double distance;

    // Inverse of SOFT_MODEL branches
    if (compensatedSnr >= 35.0)
    {
        // SNR ≥ 35 dB → distance ≤ 0.5m
        distance = std::max(0.5, (35.0 - compensatedSnr) / 0.8);
    }
    else if (compensatedSnr > 19.0)
    {
        // 19 < SNR ≤ 35 → distance 0.5-20m
        distance = (35.0 - compensatedSnr) / 0.8;
    }
    else if (compensatedSnr > 4.0)
    {
        // 4 < SNR ≤ 19 → distance 20-50m
        distance = 20.0 + (19.0 - compensatedSnr) / 0.5;
    }
    else if (compensatedSnr > 1.0)
    {
        // 1 < SNR ≤ 4 → distance 50-80m
        distance = 50.0 + (4.0 - compensatedSnr) / 0.1;
    }
    else
    {
        // SNR ≤ 1 → distance 80m+ (will be clamped)
        distance = 50.0 + (4.0 - compensatedSnr) / 0.1;
    }

    // Clamp to [0.5, 80.0] range
    return std::clamp(distance, 0.5, 80.0);
}

} // namespace ns3