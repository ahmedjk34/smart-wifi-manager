#ifndef DECISION_COUNT_CONTROLLER_H
#define DECISION_COUNT_CONTROLLER_H

#include "ns3/core-module.h"
#include "ns3/event-id.h"

#include <map>
#include <string>

namespace ns3
{

/**
 * \brief Controller for managing decision collection in WiFi rate adaptation simulations
 *
 * This class tracks the number of rate adaptation decisions made during simulation
 * and controls when to terminate the simulation based on collected data points.
 */
class DecisionCountController
{
  public:
    /**
     * \brief Constructor
     * \param targetDecisions Target number of decisions to collect
     * \param maxTimeSeconds Maximum simulation time in seconds
     */
    DecisionCountController(uint32_t targetDecisions, uint32_t maxTimeSeconds = 120);

    /**
     * \brief Set the log file path for writing decision data
     * \param path Path to the log file
     */
    void SetLogFilePath(const std::string& path);

    /**
     * \brief Increment the success counter
     */
    void IncrementSuccess();

    /**
     * \brief Increment the failure counter
     */
    void IncrementFailure();

    /**
     * \brief Increment the adaptation event counter
     */
    void IncrementAdaptationEvent();

    /**
     * \brief Schedule the maximum time stop event
     */
    void ScheduleMaxTimeStop();

    /**
     * \brief Check if simulation is complete
     * \return true if complete, false otherwise
     */
    bool IsComplete() const;

    /**
     * \brief Get the number of successful decisions collected
     * \return Number of successes
     */
    uint32_t GetSuccessCount() const;

    /**
     * \brief Get the number of failed decisions collected
     * \return Number of failures
     */
    uint32_t GetFailureCount() const;

    /**
     * \brief Get the total number of adaptation events
     * \return Number of adaptation events
     */
    uint32_t GetAdaptationEventCount() const;

    /**
     * \brief Analyze the distribution of decisions from log file
     * \return Map of decision types to counts
     */
    std::map<int, uint32_t> AnalyzeDecisionDistribution();

    /**
     * \brief Get data collection efficiency (0.0 to 1.0)
     * \return Efficiency ratio
     */
    double GetDataCollectionEfficiency() const;

    /**
     * \brief Get the log file path
     * \return Log file path
     */
    std::string GetLogFilePath() const;

  private:
    /**
     * \brief Check if termination conditions are met
     */
    void CheckTerminationCondition();

    /**
     * \brief Force stop the simulation
     */
    void ForceStop();

    /**
     * \brief Write simulation summary to log file
     */
    void WriteSummaryRowIfNeeded();

    /**
     * \brief Schedule periodic events to ensure data collection
     */
    void SchedulePeriodicEvents();

    /**
     * \brief Periodic check function
     */
    void PeriodicCheck();

    uint32_t m_targetSuccesses;   //!< Target number of successful decisions
    uint32_t m_targetFailures;    //!< Target number of failed decisions
    uint32_t m_currentSuccesses;  //!< Current number of successful decisions
    uint32_t m_currentFailures;   //!< Current number of failed decisions
    uint32_t m_maxSimulationTime; //!< Maximum simulation time in seconds
    std::string m_logFilePath;    //!< Path to log file
    bool m_simulationComplete;    //!< Flag indicating if simulation is complete
    uint32_t m_adaptationEvents;  //!< Total adaptation events
    double m_lastCheckTime;       //!< Last time termination was checked
    EventId m_periodicEventId;    //!< Event ID for periodic checks
    double m_minSamplingTime;     //!< Minimum sampling time before termination
    double m_lastEventTime;       //!< Time of last adaptation event
};

} // namespace ns3

#endif // DECISION_COUNT_CONTROLLER_H