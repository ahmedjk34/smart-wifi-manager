#include "decision-count-controller.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include <sstream>
#include <fstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DecisionCountController");

DecisionCountController::DecisionCountController(uint32_t targetSuccesses, uint32_t targetFailures, uint32_t maxTimeSeconds)
    : m_targetSuccesses(targetSuccesses),
      m_targetFailures(targetFailures),
      m_currentSuccesses(0),
      m_currentFailures(0),
      m_maxSimulationTime(maxTimeSeconds),
      m_logFilePath(""),
      m_simulationComplete(false) {
}

void DecisionCountController::SetLogFilePath(const std::string& path) {
    m_logFilePath = path;
}

void DecisionCountController::IncrementSuccess() {
    m_currentSuccesses++;
    if (m_currentSuccesses >= m_targetSuccesses && m_currentFailures >= m_targetFailures) {
        NS_LOG_INFO("Target samples reached. Stopping simulation.");
        Simulator::Stop();
        m_simulationComplete = true;
    }
}

void DecisionCountController::IncrementFailure() {
    m_currentFailures++;
    if (m_currentSuccesses >= m_targetSuccesses && m_currentFailures >= m_targetFailures) {
        NS_LOG_INFO("Target samples reached. Stopping simulation.");
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

uint32_t DecisionCountController::GetSuccessCount() const {
    return m_currentSuccesses;
}
uint32_t DecisionCountController::GetFailureCount() const {
    return m_currentFailures;
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
                try {
                    decisionReason = std::stoi(token);
                    decisionCounts[decisionReason]++;
                } catch (...) {
                    // If token is not an integer, skip this row
                }
                break;
            }
        }
    }
    return decisionCounts;
}

void DecisionCountController::ForceStop() {
    if (!m_simulationComplete) {
        NS_LOG_WARN("Maximum simulation time reached. Only " << m_currentSuccesses 
                   << " successes and " << m_currentFailures << " failures collected.");
        Simulator::Stop();
        m_simulationComplete = true;

        // Optionally: write diagnostic summary (no dummy CSV row)
        WriteSummaryRowIfNeeded();
    }
}

void DecisionCountController::WriteSummaryRowIfNeeded() {
    // This function does NOT write a CSV row.
    // If you want to log a summary, use NS_LOG or write a comment line (NOT a blank data row).
    // if (!m_logFilePath.empty()) {
    //     std::ofstream ofs(m_logFilePath, std::ios::app);
    //     ofs << "# SUMMARY: successes=" << m_currentSuccesses
    //         << ", failures=" << m_currentFailures << "\n";
    //     ofs.close();
    // }
}

} // namespace ns3