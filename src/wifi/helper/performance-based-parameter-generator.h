#ifndef PERFORMANCE_BASED_PARAMETER_GENERATOR_H
#define PERFORMANCE_BASED_PARAMETER_GENERATOR_H

#include "ns3/core-module.h"
#include <vector>
#include <map>
#include <cmath>

namespace ns3 {

struct ScenarioParams {
    double distance;
    double speed;
    uint32_t interferers;
    uint32_t packetSize;
    std::string trafficRate;
    std::string category;
    double targetSnrMin;
    double targetSnrMax;
    uint32_t targetDecisions;
    std::string scenarioName;
};

class PerformanceBasedParameterGenerator {
private:
    const double T1_TYPICAL = 10.0;  
    const double T2_TYPICAL = 15.0;  
    const double T3_TYPICAL = 25.0;  
    
public:
    std::vector<ScenarioParams> GenerateStratifiedScenarios(uint32_t totalScenarios = 200);
    
private:
    ScenarioParams GenerateTierTransitionScenario(uint32_t index);
    ScenarioParams GenerateConfidenceBoundaryScenario(uint32_t index);
    ScenarioParams GenerateSeverityTestingScenario(uint32_t index);
    ScenarioParams GenerateTrendAnalysisScenario(uint32_t index);
    double CalculateDistanceForSnr(double targetSnr);
};

} // namespace ns3

#endif