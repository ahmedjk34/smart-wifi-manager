/*
 * Copyright (c) 2004,2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Modified for SmartWifiManagerV2: ahmedjk34
 */

#include "smart-wifi-manager-v2.h"

#include "ns3/log.h"
#include "ns3/wifi-tx-vector.h"

#define Min(a, b) ((a < b) ? a : b)
#define Max(a, b) ((a > b) ? a : b)

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SmartWifiManagerV2");

/**
 * \brief hold per-remote-station state for Smart Wifi manager V2.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the Smart Wifi manager V2.
 * V2 adds history tracking for adaptive rate selection.
 */
struct SmartWifiRemoteStationV2 : public WifiRemoteStation
{
   uint32_t m_timer;
   uint32_t m_success;
   uint32_t m_failed;
   bool     m_recovery;
   uint32_t m_timerTimeout;
   uint32_t m_successThreshold;

   //V1
   uint8_t  m_rate;
   double   m_lastSnr;

   // V2: Track recent transmission outcomes for adaptive logic
   static const int HISTORY_LEN = 5;
   bool m_txHistory[HISTORY_LEN]; // true = success, false = failure
   int m_historyIdx;

   SmartWifiRemoteStationV2() : m_historyIdx(0) {
       for (int i = 0; i < HISTORY_LEN; ++i) m_txHistory[i] = true;
   }
};

NS_OBJECT_ENSURE_REGISTERED(SmartWifiManagerV2);

TypeId
SmartWifiManagerV2::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SmartWifiManagerV2")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<SmartWifiManagerV2>()
            .AddAttribute("SuccessK",
                          "Multiplication factor for the success threshold in the Smart algorithm.",
                          DoubleValue(2.0),
                          MakeDoubleAccessor(&SmartWifiManagerV2::m_successK),
                          MakeDoubleChecker<double>())
            .AddAttribute("TimerK",
                          "Multiplication factor for the timer threshold in the Smart algorithm.",
                          DoubleValue(2.0),
                          MakeDoubleAccessor(&SmartWifiManagerV2::m_timerK),
                          MakeDoubleChecker<double>())
            .AddAttribute("MaxSuccessThreshold",
                          "Maximum value of the success threshold in the Smart algorithm.",
                          UintegerValue(60),
                          MakeUintegerAccessor(&SmartWifiManagerV2::m_maxSuccessThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MinTimerThreshold",
                          "The minimum value for the 'timer' threshold in the Smart algorithm.",
                          UintegerValue(15),
                          MakeUintegerAccessor(&SmartWifiManagerV2::m_minTimerThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MinSuccessThreshold",
                          "The minimum value for the success threshold in the Smart algorithm.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&SmartWifiManagerV2::m_minSuccessThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddTraceSource("Rate",
                            "Traced value for rate changes (b/s)",
                            MakeTraceSourceAccessor(&SmartWifiManagerV2::m_currentRate),
                            "ns3::TracedValueCallback::Uint64");
    return tid;
}

SmartWifiManagerV2::SmartWifiManagerV2()
    : WifiRemoteStationManager(),
      m_currentRate(0)
{
    NS_LOG_FUNCTION(this);
}

SmartWifiManagerV2::~SmartWifiManagerV2()
{
    NS_LOG_FUNCTION(this);
}

void
SmartWifiManagerV2::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    if (GetHtSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support HT rates");
    }
    if (GetVhtSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support VHT rates");
    }
    if (GetHeSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support HE rates");
    }
}

WifiRemoteStation*
SmartWifiManagerV2::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    auto station = new SmartWifiRemoteStationV2();

    station->m_successThreshold = m_minSuccessThreshold;
    station->m_timerTimeout = m_minTimerThreshold;
    station->m_rate = 0;
    station->m_success = 0;
    station->m_failed = 0;
    station->m_recovery = false;
    station->m_timer = 0;
    station->m_lastSnr = 0.0;
    station->m_historyIdx = 0;
    for (int i = 0; i < SmartWifiRemoteStationV2::HISTORY_LEN; ++i) station->m_txHistory[i] = true;

    return station;
}

void
SmartWifiManagerV2::DoReportRtsFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

void
SmartWifiManagerV2::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<SmartWifiRemoteStationV2*>(st);
    station->m_timer++;
    station->m_failed++;
    station->m_success = 0;

    // V2: Record transmission failure in history
    station->m_txHistory[station->m_historyIdx] = false;
    station->m_historyIdx = (station->m_historyIdx + 1) % SmartWifiRemoteStationV2::HISTORY_LEN;

    if (station->m_recovery)
    {
        NS_ASSERT(station->m_failed >= 1);
        if (station->m_failed == 1)
        {
            station->m_successThreshold =
                (int)(Min(station->m_successThreshold * m_successK, m_maxSuccessThreshold));
            station->m_timerTimeout =
                (int)(Max(station->m_timerTimeout * m_timerK, m_minSuccessThreshold));
            if (station->m_rate != 0)
            {
                station->m_rate--;
            }
        }
        station->m_timer = 0;
    }
    else
    {
        NS_ASSERT(station->m_failed >= 1);
        if (((station->m_failed - 1) % 2) == 1)
        {
            station->m_timerTimeout = m_minTimerThreshold;
            station->m_successThreshold = m_minSuccessThreshold;
            if (station->m_rate != 0)
            {
                station->m_rate--;
            }
        }
        if (station->m_failed >= 2)
        {
            station->m_timer = 0;
        }
    }
}

void
SmartWifiManagerV2::DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << station << rxSnr << txMode);
    auto st = static_cast<SmartWifiRemoteStationV2*>(station);
    st->m_lastSnr = rxSnr;
}

void
SmartWifiManagerV2::DoReportRtsOk(WifiRemoteStation* station,
                               double ctsSnr,
                               WifiMode ctsMode,
                               double rtsSnr)
{
    NS_LOG_FUNCTION(this << station << ctsSnr << ctsMode << rtsSnr);
    NS_LOG_DEBUG("station=" << station << " rts ok");
}

void
SmartWifiManagerV2::DoReportDataOk(WifiRemoteStation* st,
                                double ackSnr,
                                WifiMode ackMode,
                                double dataSnr,
                                uint16_t dataChannelWidth,
                                uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
    auto station = static_cast<SmartWifiRemoteStationV2*>(st);
    station->m_timer++;
    station->m_success++;
    station->m_failed = 0;
    station->m_recovery = false;
    station->m_lastSnr = dataSnr;

    // V2: Record transmission success in history
    station->m_txHistory[station->m_historyIdx] = true;
    station->m_historyIdx = (station->m_historyIdx + 1) % SmartWifiRemoteStationV2::HISTORY_LEN;

    uint8_t maxRateIdx = GetNSupported(station) - 1;

    // V2: Adaptive decision tree 
    // Use SNR for initial rate, but also consider recent history
    int recentFailures = 0;
    for (int i = 0; i < SmartWifiRemoteStationV2::HISTORY_LEN; ++i)
        if (!station->m_txHistory[i]) recentFailures++;


    if (station->m_lastSnr < 10.0 || recentFailures >= 4) {
        // Rule 1: Poor SNR OR excessive recent failures → conservative step-down
        station->m_rate = Max(0, station->m_rate - 1); // reduce, not force to lowest
    } else if (station->m_lastSnr < 15.0) {
        // Rule 2: Low SNR band → assign rate index 1
        station->m_rate = Min(1, maxRateIdx);
    } else if (station->m_lastSnr < 25.0) {
        // Rule 3: Mid SNR band → assign rate index 2
        station->m_rate = Min(2, maxRateIdx);
    } else {
        // Rule 4: High SNR band → history-gated maximum rate selection
        if (recentFailures == 0)
            // Rule 4a: Perfect recent history → immediate jump to maximum rate
            station->m_rate = maxRateIdx;
         else
            // Rule 4b: Some recent failures → cautious incremental increase
            station->m_rate = Min(station->m_rate + 1, maxRateIdx); // cautious step up
    }

    NS_LOG_DEBUG("station=" << station << " data ok success=" << station->m_success
                            << ", timer=" << station->m_timer << ", SNR=" << station->m_lastSnr << ", rate=" << (int)station->m_rate
                            << ", recentFailures=" << recentFailures);
}

void
SmartWifiManagerV2::DoReportFinalRtsFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

void
SmartWifiManagerV2::DoReportFinalDataFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

WifiTxVector
SmartWifiManagerV2::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    auto station = static_cast<SmartWifiRemoteStationV2*>(st);
    uint16_t channelWidth = GetChannelWidth(station);
    if (channelWidth > 20 && channelWidth != 22)
    {
        channelWidth = 20;
    }
    WifiMode mode = GetSupported(station, station->m_rate);
    uint64_t rate = mode.GetDataRate(channelWidth);
    if (m_currentRate != rate)
    {
        NS_LOG_DEBUG("New datarate: " << rate);
        m_currentRate = rate;
    }
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        800,
        1,
        1,
        0,
        channelWidth,
        GetAggregation(station));
}

WifiTxVector
SmartWifiManagerV2::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<SmartWifiRemoteStationV2*>(st);
    uint16_t channelWidth = GetChannelWidth(station);
    if (channelWidth > 20 && channelWidth != 22)
    {
        channelWidth = 20;
    }
    WifiMode mode;
    if (!GetUseNonErpProtection())
    {
        mode = GetSupported(station, 0);
    }
    else
    {
        mode = GetNonErpSupported(station, 0);
    }
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        800,
        1,
        1,
        0,
        channelWidth,
        GetAggregation(station));
}

} // namespace ns3

