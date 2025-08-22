/*
 * Copyright (c) 2004,2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "smart-wifi-manager-v1.h"

#include "ns3/log.h"
#include "ns3/wifi-tx-vector.h"

#define Min(a, b) ((a < b) ? a : b)
#define Max(a, b) ((a > b) ? a : b)

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SmartWifiManagerV1");

/**
 * \brief hold per-remote-station state for Smart Wifi manager V1.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the Smart Wifi manager V1
 */
struct SmartWifiRemoteStationV1 : public WifiRemoteStation
{
   uint32_t m_timer;            // counts transmissions since last rate change attempt
   uint32_t m_success;          // consecutive successes (resets on failure or rate change)
   uint32_t m_failed;           // consecutive failures since last success
   bool     m_recovery;         // true after we’ve just increased rate until the first success
   uint32_t m_timerTimeout;     // dynamic timer threshold to trigger rate-up
   uint32_t m_successThreshold; // dynamic success threshold to trigger rate-up
   uint8_t  m_rate;             // index into the supported-rate table (0 = lowest)
   double   m_lastSnr;          // last SNR measurement (for decision tree)
};

NS_OBJECT_ENSURE_REGISTERED(SmartWifiManagerV1);

TypeId
SmartWifiManagerV1::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SmartWifiManagerV1")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<SmartWifiManagerV1>()
            .AddAttribute("SuccessK",
                          "Multiplication factor for the success threshold in the Smart algorithm.",
                          DoubleValue(2.0),
                          MakeDoubleAccessor(&SmartWifiManagerV1::m_successK),
                          MakeDoubleChecker<double>())
            .AddAttribute("TimerK",
                          "Multiplication factor for the timer threshold in the Smart algorithm.",
                          DoubleValue(2.0),
                          MakeDoubleAccessor(&SmartWifiManagerV1::m_timerK),
                          MakeDoubleChecker<double>())
            .AddAttribute("MaxSuccessThreshold",
                          "Maximum value of the success threshold in the Smart algorithm.",
                          UintegerValue(60),
                          MakeUintegerAccessor(&SmartWifiManagerV1::m_maxSuccessThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MinTimerThreshold",
                          "The minimum value for the 'timer' threshold in the Smart algorithm.",
                          UintegerValue(15),
                          MakeUintegerAccessor(&SmartWifiManagerV1::m_minTimerThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MinSuccessThreshold",
                          "The minimum value for the success threshold in the Smart algorithm.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&SmartWifiManagerV1::m_minSuccessThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddTraceSource("Rate",
                            "Traced value for rate changes (b/s)",
                            MakeTraceSourceAccessor(&SmartWifiManagerV1::m_currentRate),
                            "ns3::TracedValueCallback::Uint64");
    return tid;
}

SmartWifiManagerV1::SmartWifiManagerV1()
    : WifiRemoteStationManager(),
      m_currentRate(0)
{
    NS_LOG_FUNCTION(this);
}

SmartWifiManagerV1::~SmartWifiManagerV1()
{
    NS_LOG_FUNCTION(this);
}

void
SmartWifiManagerV1::DoInitialize()
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
SmartWifiManagerV1::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    auto station = new SmartWifiRemoteStationV1();

    station->m_successThreshold = m_minSuccessThreshold;
    station->m_timerTimeout = m_minTimerThreshold;
    station->m_rate = 0;
    station->m_success = 0;
    station->m_failed = 0;
    station->m_recovery = false;
    station->m_timer = 0;
    station->m_lastSnr = 0.0;

    return station;
}

void
SmartWifiManagerV1::DoReportRtsFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

void
SmartWifiManagerV1::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<SmartWifiRemoteStationV1*>(st);
    station->m_timer++;
    station->m_failed++;
    station->m_success = 0;

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
SmartWifiManagerV1::DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << station << rxSnr << txMode);
    auto st = static_cast<SmartWifiRemoteStationV1*>(station);
    st->m_lastSnr = rxSnr;
}

void
SmartWifiManagerV1::DoReportRtsOk(WifiRemoteStation* station,
                               double ctsSnr,
                               WifiMode ctsMode,
                               double rtsSnr)
{
    NS_LOG_FUNCTION(this << station << ctsSnr << ctsMode << rtsSnr);
    NS_LOG_DEBUG("station=" << station << " rts ok");
}

void
SmartWifiManagerV1::DoReportDataOk(WifiRemoteStation* st,
                                double ackSnr,
                                WifiMode ackMode,
                                double dataSnr,
                                uint16_t dataChannelWidth,
                                uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
    auto station = static_cast<SmartWifiRemoteStationV1*>(st);
    station->m_timer++;
    station->m_success++;
    station->m_failed = 0;
    station->m_recovery = false;
    station->m_lastSnr = dataSnr;

    
//     // Decision Tree (improved for realistic SNR mapping)
// if (station->m_lastSnr < 8.0) {
//     station->m_rate = 0; // safest, lowest rate
// } else if (station->m_lastSnr < 14.0) {
//     station->m_rate = Min(1, maxRateIdx); // moderate rate
// } else if (station->m_lastSnr < 22.0) {
//     station->m_rate = Min(2, maxRateIdx); // higher rate
// } else {
//     station->m_rate = maxRateIdx; // best rate
// }


    // V1: Simple Decision Tree for Rate Selection
    // Example: 4 rates
    uint8_t maxRateIdx = GetNSupported(station) - 1;
    // Decision tree (feel free to tune SNR thresholds and supported rates count)
    if (station->m_lastSnr < 10.0)
    {
        station->m_rate = 0; // lowest rate
    }
    else if (station->m_lastSnr < 15.0)
    {
        station->m_rate = Min(1, maxRateIdx);
    }
    else if (station->m_lastSnr < 25.0)
    {
        station->m_rate = Min(2, maxRateIdx);
    }
    else
    {
        station->m_rate = maxRateIdx;
    }

    NS_LOG_DEBUG("station=" << station << " data ok success=" << station->m_success
                            << ", timer=" << station->m_timer << ", SNR=" << station->m_lastSnr << ", rate=" << (int)station->m_rate);
}

void
SmartWifiManagerV1::DoReportFinalRtsFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

void
SmartWifiManagerV1::DoReportFinalDataFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

WifiTxVector
SmartWifiManagerV1::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    auto station = static_cast<SmartWifiRemoteStationV1*>(st);
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
SmartWifiManagerV1::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<SmartWifiRemoteStationV1*>(st);
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