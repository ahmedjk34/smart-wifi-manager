#include "decision-count-controller.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <fstream>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DecisionCountController");

DecisionCountController::DecisionCountController(uint32_t targetSuccesses,
                                                 uint32_t targetFailures,
                                                 uint32_t maxTimeSeconds)
    : m_targetSuccesses(targetSuccesses),
      m_targetFailures(targetFailures),
      m_currentSuccesses(0),
      m_currentFailures(0),
      m_maxSimulationTime(maxTimeSeconds),
      m_logFilePath(""),
      m_simulationComplete(false),
      m_adaptationEvents(0),
      m_lastCheckTime(0)
{
}

void
DecisionCountController::SetLogFilePath(const std::string& path)
{
    m_logFilePath = path;
}

void
DecisionCountController::IncrementSuccess()
{
    m_currentSuccesses++;
    m_adaptationEvents++;
    CheckTerminationCondition();
}

void
DecisionCountController::IncrementFailure()
{
    m_currentFailures++;
    m_adaptationEvents++;
    CheckTerminationCondition();
}

void
DecisionCountController::IncrementAdaptationEvent()
{
    m_adaptationEvents++;
    // For scenarios with fewer explicit success/failure classifications,
    // treat adaptation events as data points
    m_currentSuccesses++;
    CheckTerminationCondition();
}

void
DecisionCountController::CheckTerminationCondition()
{
    double currentTime = Simulator::Now().GetSeconds();

    // Enhanced termination logic for better data collection
    bool targetReached =
        (m_currentSuccesses >= m_targetSuccesses) ||
        (m_adaptationEvents >= m_targetSuccesses * 1.2); // 20% buffer for adaptation events

    bool sufficientSampling =
        (currentTime - m_lastCheckTime > 5.0) &&         // At least 5 seconds between checks
        (m_adaptationEvents >= m_targetSuccesses * 0.8); // At least 80% of target

    if (targetReached || sufficientSampling)
    {
        NS_LOG_INFO("Target samples reached. Successes: "
                    << m_currentSuccesses << ", Failures: " << m_currentFailures
                    << ", Total adaptations: " << m_adaptationEvents);
        Simulator::Stop();
        m_simulationComplete = true;
    }

    m_lastCheckTime = currentTime;
}

void
DecisionCountController::ScheduleMaxTimeStop()
{
    Simulator::Schedule(Seconds(m_maxSimulationTime), &DecisionCountController::ForceStop, this);
}

bool
DecisionCountController::IsComplete() const
{
    return m_simulationComplete;
}

uint32_t
DecisionCountController::GetSuccessCount() const
{
    return m_currentSuccesses;
}

uint32_t
DecisionCountController::GetFailureCount() const
{
    return m_currentFailures;
}

uint32_t
DecisionCountController::GetAdaptationEventCount() const
{
    return m_adaptationEvents;
}

std::map<int, uint32_t>
DecisionCountController::AnalyzeDecisionDistribution()
{
    std::map<int, uint32_t> decisionCounts;
    if (m_logFilePath.empty())
        return decisionCounts;

    std::ifstream file(m_logFilePath);
    if (!file.is_open())
        return decisionCounts;

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;
        int decisionReason = 0;

        // Parse CSV to extract decision reason (assuming it's in column 16)
        for (int i = 0; i <= 16; ++i)
        {
            std::getline(iss, token, ',');
            if (i == 16)
            {
                try
                {
                    decisionReason = std::stoi(token);
                    decisionCounts[decisionReason]++;
                }
                catch (...)
                {
                    // Skip invalid entries
                }
                break;
            }
        }
    }

    file.close();
    return decisionCounts;
}

void
DecisionCountController::ForceStop()
{
    if (!m_simulationComplete)
    {
        NS_LOG_WARN("Maximum simulation time reached. Collected "
                    << m_currentSuccesses << " successes, " << m_currentFailures << " failures, "
                    << m_adaptationEvents << " total adaptations.");
        Simulator::Stop();
        m_simulationComplete = true;
        WriteSummaryRowIfNeeded();
    }
}

void
DecisionCountController::WriteSummaryRowIfNeeded()
{
    // Write summary as comment, not as data row
    if (!m_logFilePath.empty())
    {
        std::ofstream ofs(m_logFilePath, std::ios::app);
        if (ofs.is_open())
        {
            ofs << "# SIMULATION_SUMMARY: successes=" << m_currentSuccesses
                << ", failures=" << m_currentFailures << ", adaptations=" << m_adaptationEvents
                << ", sim_time=" << Simulator::Now().GetSeconds() << "s\n";
            ofs.close();
        }
    }
}

double
DecisionCountController::GetDataCollectionEfficiency() const
{
    if (m_targetSuccesses == 0)
        return 0.0;
    return std::min(1.0, double(m_adaptationEvents) / double(m_targetSuccesses));
}

} // namespace ns3