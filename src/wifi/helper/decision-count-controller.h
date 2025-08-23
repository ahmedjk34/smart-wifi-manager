#ifndef DECISION_COUNT_CONTROLLER_H
#define DECISION_COUNT_CONTROLLER_H

#include "ns3/core-module.h"
#include <map>
#include <fstream>
#include <string>

namespace ns3 {

class DecisionCountController {
private:
    uint32_t m_targetSuccesses;
    uint32_t m_targetFailures;
    uint32_t m_currentSuccesses;
    uint32_t m_currentFailures;
    uint32_t m_maxSimulationTime;
    std::string m_logFilePath;
    bool m_simulationComplete;

public:
    DecisionCountController(uint32_t targetSuccesses, uint32_t targetFailures, uint32_t maxTimeSeconds = 60);

    void SetLogFilePath(const std::string& path);
    void IncrementSuccess();
    void IncrementFailure();
    void ScheduleMaxTimeStop();
    bool IsComplete() const;
    uint32_t GetSuccessCount() const;
    uint32_t GetFailureCount() const;
    std::map<int, uint32_t> AnalyzeDecisionDistribution();
    void ForceStop();

    // Diagnostic summary; does not write any dummy/blank CSV rows
    void WriteSummaryRowIfNeeded();
};

} // namespace ns3

#endif // DECISION_COUNT_CONTROLLER_H