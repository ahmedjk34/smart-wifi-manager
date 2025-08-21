#include "performance-based-parameter-generator.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace ns3 {

std::vector<ScenarioParams> 
PerformanceBasedParameterGenerator::GenerateStratifiedScenarios(uint32_t totalScenarios) {
    std::vector<ScenarioParams> scenarios;
    
    // Category A: Tier Transition Scenarios (40%)
    uint32_t categoryA = totalScenarios * 0.4;
    for (uint32_t i = 0; i < categoryA; ++i) {
        scenarios.push_back(GenerateTierTransitionScenario(i));
    }
    
    // Category B: Confidence Boundary Scenarios (30%)
    uint32_t categoryB = totalScenarios * 0.3;
    for (uint32_t i = 0; i < categoryB; ++i) {
        scenarios.push_back(GenerateConfidenceBoundaryScenario(i));
    }
    
    // Category C: Severity Testing Scenarios (20%)
    uint32_t categoryC = totalScenarios * 0.2;
    for (uint32_t i = 0; i < categoryC; ++i) {
        scenarios.push_back(GenerateSeverityTestingScenario(i));
    }
    
    // Category D: Trend Analysis Scenarios (10%)
    uint32_t categoryD = totalScenarios - (categoryA + categoryB + categoryC);
    for (uint32_t i = 0; i < categoryD; ++i) {
        scenarios.push_back(GenerateTrendAnalysisScenario(i));
    }
    
    return scenarios;
}

ScenarioParams 
PerformanceBasedParameterGenerator::GenerateTierTransitionScenario(uint32_t index) {
    ScenarioParams params;
    params.category = "TierTransition";
    params.targetDecisions = 1000;
    
    std::vector<std::pair<double, double>> snrRanges = {
        {T1_TYPICAL - 2.0, T1_TYPICAL + 2.0},
        {T2_TYPICAL - 2.0, T2_TYPICAL + 2.0},
        {T3_TYPICAL - 2.0, T3_TYPICAL + 2.0}
    };
    
    auto range = snrRanges[index % snrRanges.size()];
    params.targetSnrMin = range.first;
    params.targetSnrMax = range.second;
    
    double targetSnr = (range.first + range.second) / 2.0;
    params.distance = CalculateDistanceForSnr(targetSnr);
    params.speed = 1.0 + (index % 5);
    params.interferers = (index % 2) ? 1 : 2;
    params.packetSize = (index % 2) ? 512 : 1024;
    params.trafficRate = (index % 2) ? "20Mbps" : "30Mbps";
    
    std::ostringstream name;
    name << "TierTrans_" << std::setfill('0') << std::setw(3) << index 
         << "_snr" << std::fixed << std::setprecision(1) << targetSnr;
    params.scenarioName = name.str();
    
    return params;
}

ScenarioParams 
PerformanceBasedParameterGenerator::GenerateConfidenceBoundaryScenario(uint32_t index) {
    ScenarioParams params;
    params.category = "ConfidenceBoundary";
    params.targetDecisions = 1200;
    
    params.targetSnrMin = 17.0;
    params.targetSnrMax = 20.0;
    params.distance = CalculateDistanceForSnr(18.5);
    params.speed = 0.5 + (index % 4) * 0.5;
    params.interferers = 1 + (index % 2);
    params.packetSize = 256 + (index % 3) * 512;
    params.trafficRate = "25Mbps";
    
    std::ostringstream name;
    name << "ConfBound_" << std::setfill('0') << std::setw(3) << index;
    params.scenarioName = name.str();
    
    return params;
}

ScenarioParams 
PerformanceBasedParameterGenerator::GenerateSeverityTestingScenario(uint32_t index) {
    ScenarioParams params;
    params.category = "SeverityTesting";
    params.targetDecisions = 800;
    
    params.targetSnrMin = 7.0;
    params.targetSnrMax = 12.0;
    params.distance = CalculateDistanceForSnr(9.5);
    params.speed = 3.0 + (index % 8);
    params.interferers = 2 + (index % 3);
    params.packetSize = 1024 + (index % 2) * 512;
    params.trafficRate = "40Mbps";
    
    std::ostringstream name;
    name << "Severity_" << std::setfill('0') << std::setw(3) << index;
    params.scenarioName = name.str();
    
    return params;
}

ScenarioParams 
PerformanceBasedParameterGenerator::GenerateTrendAnalysisScenario(uint32_t index) {
    ScenarioParams params;
    params.category = "TrendAnalysis";
    params.targetDecisions = 1500;
    
    params.targetSnrMin = 12.0;
    params.targetSnrMax = 27.0;
    params.distance = CalculateDistanceForSnr(19.5);
    params.speed = 5.0 + (index % 10);
    params.interferers = 1 + (index % 2);
    params.packetSize = 512 + (index % 3) * 256;
    params.trafficRate = "35Mbps";
    
    std::ostringstream name;
    name << "Trend_" << std::setfill('0') << std::setw(3) << index;
    params.scenarioName = name.str();
    
    return params;
}

double 
PerformanceBasedParameterGenerator::CalculateDistanceForSnr(double targetSnr) {
    const double TX_POWER_DBM = 20.0;
    const double NOISE_FLOOR_DBM = -90.0;
    const double FREQUENCY_OFFSET = 32.44 + 20.0 * log10(2400.0);
    
    double requiredPathLoss = TX_POWER_DBM - NOISE_FLOOR_DBM - targetSnr;
    double logDistance = (requiredPathLoss - FREQUENCY_OFFSET) / 40.0;
    double distance = pow(10.0, logDistance);
    
    return std::max(1.0, std::min(100.0, distance));
}

} // namespace ns3