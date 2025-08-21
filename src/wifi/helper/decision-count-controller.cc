#include "decision-count-controller.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include <sstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DecisionCountController");

DecisionCountController::DecisionCountController(uint32_t targetDecisions, uint32_t maxTimeSeconds)
    : m_targetDecisions(targetDecisions),
      m_currentDecisions(0),
      m_maxSimulationTime(maxTimeSeconds),
      m_simulationComplete(false) {
}

void DecisionCountController::SetLogFilePath(const std::string& path) {
    m_logFilePath = path;
}

void DecisionCountController::IncrementDecisionCount() {
    m_currentDecisions++;
    
    if (m_currentDecisions >= m_targetDecisions) {
        NS_LOG_INFO("Target decisions (" << m_targetDecisions << ") reached. Stopping simulation.");
        Simulator::Stop();
        m_simulationComplete = true;
    }
}

void DecisionCountController::ScheduleMaxTimeStop() {
    Simulator::Schedule(Seconds(m_maxSimulationTime), &DecisionCountController::ForceStop, this);
}

bool DecisionCountController::IsComplete() const {
    return m_simulationComplete;
}

uint32_t DecisionCountController::GetDecisionCount() const {
    return m_currentDecisions;
}

std::map<int, uint32_t> DecisionCountController::AnalyzeDecisionDistribution() {
    std::map<int, uint32_t> decisionCounts;
    
    if (m_logFilePath.empty()) return decisionCounts;
    
    std::ifstream file(m_logFilePath);
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        int decisionReason = 0;
        
        for (int i = 0; i <= 16; ++i) {
            std::getline(iss, token, ',');
            if (i == 16) {
                decisionReason = std::stoi(token);
                decisionCounts[decisionReason]++;
                break;
            }
        }
    }
    
    return decisionCounts;
}

void DecisionCountController::ForceStop() {
    if (!m_simulationComplete) {
        NS_LOG_WARN("Maximum simulation time reached. Only " << m_currentDecisions 
                   << " of " << m_targetDecisions << " decisions collected.");
        Simulator::Stop();
        m_simulationComplete = true;
    }
}

} // namespace ns3