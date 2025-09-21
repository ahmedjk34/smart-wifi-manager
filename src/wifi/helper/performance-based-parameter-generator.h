#ifndef PERFORMANCE_BASED_PARAMETER_GENERATOR_H
#define PERFORMANCE_BASED_PARAMETER_GENERATOR_H

#include "ns3/core-module.h"

#include <cmath>
#include <map>
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
    const double POOR_SNR_MIN = 3.0;
    const double POOR_SNR_MAX = 12.0;
    const double MEDIUM_SNR_MIN = 12.0;
    const double MEDIUM_SNR_MAX = 22.0;
    const double GOOD_SNR_MIN = 22.0;
    const double GOOD_SNR_MAX = 30.0;

  public:
    std::vector<ScenarioParams> GenerateStratifiedScenarios(uint32_t totalScenarios = 200);

  private:
    // Renamed and refocused scenario generators
    ScenarioParams GeneratePoorPerformanceScenario(uint32_t index);
    ScenarioParams GenerateMediumPerformanceScenario(uint32_t index);
    ScenarioParams GenerateHighInterferenceScenario(uint32_t index);
    ScenarioParams GenerateGoodPerformanceScenario(uint32_t index);
    double CalculateDistanceForSnr(double targetSnr);
};

} // namespace ns3

#endif