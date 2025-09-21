#ifndef DECISION_COUNT_CONTROLLER_H
#define DECISION_COUNT_CONTROLLER_H

#include "ns3/core-module.h"

#include <fstream>
#include <map>
#include <string>

namespace ns3
{

class DecisionCountController
{
  private:
    uint32_t m_targetSuccesses;
    uint32_t m_targetFailures;
    uint32_t m_currentSuccesses;
    uint32_t m_currentFailures;
    uint32_t m_maxSimulationTime;
    std::string m_logFilePath;
    bool m_simulationComplete;

    // Enhanced tracking for better data collection
    uint32_t m_adaptationEvents;
    double m_lastCheckTime;

  public:
    DecisionCountController(uint32_t targetSuccesses,
                            uint32_t targetFailures,
                            uint32_t maxTimeSeconds = 60);

    // Core functionality
    void SetLogFilePath(const std::string& path);
    std::string GetLogFilePath() const; // *** NEW METHOD ***
    void IncrementSuccess();
    void IncrementFailure();
    void IncrementAdaptationEvent(); // New method for rate adaptation events
    void ScheduleMaxTimeStop();

    // Status and metrics
    bool IsComplete() const;
    uint32_t GetSuccessCount() const;
    uint32_t GetFailureCount() const;
    uint32_t GetAdaptationEventCount() const;   // New method
    double GetDataCollectionEfficiency() const; // New method

    // Analysis and control
    std::map<int, uint32_t> AnalyzeDecisionDistribution();
    void ForceStop();
    void WriteSummaryRowIfNeeded();

  private:
    void CheckTerminationCondition(); // Enhanced termination logic
};

} // namespace ns3

#endif // DECISION_COUNT_CONTROLLER_H