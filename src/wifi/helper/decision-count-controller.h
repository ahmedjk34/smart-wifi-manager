#ifndef DECISION_COUNT_CONTROLLER_H
#define DECISION_COUNT_CONTROLLER_H

#include "ns3/core-module.h"
#include <map>
#include <fstream>

namespace ns3 {

class DecisionCountController {
private:
    uint32_t m_targetDecisions;
    uint32_t m_currentDecisions;
    uint32_t m_maxSimulationTime;
    std::string m_logFilePath;
    bool m_simulationComplete;
    
public:
    DecisionCountController(uint32_t targetDecisions, uint32_t maxTimeSeconds = 60);
    
    void SetLogFilePath(const std::string& path);
    void IncrementDecisionCount();
    void ScheduleMaxTimeStop();
    bool IsComplete() const;
    uint32_t GetDecisionCount() const;
    std::map<int, uint32_t> AnalyzeDecisionDistribution();
    
private:
    void ForceStop();
};

} // namespace ns3

#endif