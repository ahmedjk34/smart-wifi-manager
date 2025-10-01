#ifndef PERFORMANCE_BASED_PARAMETER_GENERATOR_H
#define PERFORMANCE_BASED_PARAMETER_GENERATOR_H

#include "ns3/core-module.h"

#include <cmath>
#include <map>
#include <string> // needed for std::string
#include <vector>

namespace ns3
{

struct ScenarioParams
{
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

class PerformanceBasedParameterGenerator
{
  private:
    // Expanded tier definitions for better coverage
    // in performance-based-parameter-generator.h
    // const double POOR_SNR_MIN = -15.0;
    // const double POOR_SNR_MAX = 12.0;
    // const double MEDIUM_SNR_MIN = 12.0;
    // const double MEDIUM_SNR_MAX = 22.0;
    // const double GOOD_SNR_MIN = 22.0;
    // const double GOOD_SNR_MAX = 35.0; // expanded
    // const double EXCELLENT_SNR_MIN = 35.0;
    // const double EXCELLENT_SNR_MAX = 45.0; // new tier

  public:
    std::vector<ScenarioParams> GenerateStratifiedScenarios(uint32_t totalScenarios = 200);

  private:
    // Scenario generators
    ScenarioParams GeneratePoorPerformanceScenario(uint32_t index);
    ScenarioParams GenerateMediumPerformanceScenario(uint32_t index);
    ScenarioParams GenerateHighInterferenceScenario(uint32_t index);
    ScenarioParams GenerateGoodPerformanceScenario(uint32_t index);
    ScenarioParams GenerateNearIdealScenario(uint32_t index);
    ScenarioParams GenerateExtremeScenario(uint32_t index);
    ScenarioParams GenerateEdgeStressScenario(uint32_t index);
    ScenarioParams GenerateRandomChaosScenario(uint32_t index);
    ScenarioParams GenerateExcellentPerformanceScenario(uint32_t index);
    ScenarioParams GenerateForceHighRateScenario(uint32_t index);

    // Utility
    double CalculateDistanceForSnr(double targetSnr, uint32_t interferers);
};

} // namespace ns3

#endif // PERFORMANCE_BASED_PARAMETER_GENERATOR_H
