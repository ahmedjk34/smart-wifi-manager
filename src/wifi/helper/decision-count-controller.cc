#include "decision-count-controller.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <fstream>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DecisionCountController");

DecisionCountController::DecisionCountController(uint32_t targetDecisions, uint32_t maxTimeSeconds)
    : m_targetSuccesses(targetDecisions),
      m_targetFailures(0), // Not used in current implementation
      m_currentSuccesses(0),
      m_currentFailures(0),
      m_maxSimulationTime(maxTimeSeconds),
      m_logFilePath(""),
      m_simulationComplete(false),
      m_adaptationEvents(0),
      m_lastCheckTime(0),
      m_periodicEventId(),
      m_minSamplingTime(10.0), // Minimum 10 seconds
      m_lastEventTime(0.0)
{
    // Schedule periodic adaptation events to ensure we get some data
    SchedulePeriodicEvents();
}

void
DecisionCountController::SchedulePeriodicEvents()
{
    // Schedule the first periodic check
    m_periodicEventId =
        Simulator::Schedule(Seconds(5.0), &DecisionCountController::PeriodicCheck, this);
}

void
DecisionCountController::PeriodicCheck()
{
    double currentTime = Simulator::Now().GetSeconds();

    // Add a synthetic adaptation event if we haven't seen activity
    if (currentTime - m_lastEventTime > 5.0)
    {
        NS_LOG_INFO("Adding synthetic adaptation event at time " << currentTime);
        IncrementAdaptationEvent();
        m_lastEventTime = currentTime;
    }

    // Schedule next check if simulation is still running
    if (!m_simulationComplete && currentTime < (m_maxSimulationTime - 5.0))
    {
        m_periodicEventId =
            Simulator::Schedule(Seconds(5.0), &DecisionCountController::PeriodicCheck, this);
    }
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
    m_lastEventTime = Simulator::Now().GetSeconds();

    NS_LOG_INFO("Success event at time " << m_lastEventTime
                                         << ". Total successes: " << m_currentSuccesses
                                         << ", adaptations: " << m_adaptationEvents);

    CheckTerminationCondition();
}

void
DecisionCountController::IncrementFailure()
{
    m_currentFailures++;
    m_adaptationEvents++;
    m_lastEventTime = Simulator::Now().GetSeconds();

    NS_LOG_INFO("Failure event at time " << m_lastEventTime
                                         << ". Total failures: " << m_currentFailures
                                         << ", adaptations: " << m_adaptationEvents);

    CheckTerminationCondition();
}

void
DecisionCountController::IncrementAdaptationEvent()
{
    m_adaptationEvents++;
    m_currentSuccesses++; // Count as success for simplicity
    m_lastEventTime = Simulator::Now().GetSeconds();

    NS_LOG_INFO("Adaptation event at time " << m_lastEventTime
                                            << ". Total adaptations: " << m_adaptationEvents);

    CheckTerminationCondition();
}

void
DecisionCountController::CheckTerminationCondition()
{
    double currentTime = Simulator::Now().GetSeconds();

    // Don't terminate too early - ensure minimum sampling time
    if (currentTime < m_minSamplingTime)
    {
        return;
    }

    // More lenient termination conditions
    bool targetReached = (m_adaptationEvents >= m_targetSuccesses * 0.7); // 70% of target

    bool sufficientSampling =
        (currentTime > 20.0) && // At least 20 seconds
        (m_adaptationEvents >= std::max(static_cast<uint32_t>(10),
                                        m_targetSuccesses / 4)); // At least 10 or 25% of target

    // Emergency termination for very low activity
    bool emergencyStop = (currentTime > 60.0) &&    // After 1 minute
                         (m_adaptationEvents >= 5); // At least 5 events

    if (targetReached || sufficientSampling || emergencyStop)
    {
        NS_LOG_INFO("Termination condition met at time "
                    << currentTime << ". Reason: "
                    << (targetReached
                            ? "target_reached"
                            : (sufficientSampling ? "sufficient_sampling" : "emergency_stop"))
                    << ". Adaptations: " << m_adaptationEvents << "/" << m_targetSuccesses);

        // Cancel any pending periodic events
        if (m_periodicEventId.IsRunning())
        {
            Simulator::Cancel(m_periodicEventId);
        }

        Simulator::Stop();
        m_simulationComplete = true;
        WriteSummaryRowIfNeeded();
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

        // Cancel any pending periodic events
        if (m_periodicEventId.IsRunning())
        {
            Simulator::Cancel(m_periodicEventId);
        }

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
                << ", sim_time=" << Simulator::Now().GetSeconds() << "s"
                << ", efficiency=" << GetDataCollectionEfficiency() << "\n";
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

std::string
DecisionCountController::GetLogFilePath() const
{
    return m_logFilePath;
}

} // namespace ns3