/*
 * Copyright (c) 2004,2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

 #include "smart-wifi-manager.h"

 #include "ns3/log.h"
 #include "ns3/wifi-tx-vector.h"
 
 #define Min(a, b) ((a < b) ? a : b)
 #define Max(a, b) ((a > b) ? a : b)
 
 namespace ns3
 {
 
 NS_LOG_COMPONENT_DEFINE("SmartWifiManager");
 
 /**
  * \brief hold per-remote-station state for Smart Wifi manager.
  *
  * This struct extends from WifiRemoteStation struct to hold additional
  * information required by the Smart Wifi manager
  */
 struct SmartWifiRemoteStation : public WifiRemoteStation
 {
     uint32_t m_timer;
     uint32_t m_success;
     uint32_t m_failed;
     bool m_recovery;
     uint32_t m_timerTimeout;
     uint32_t m_successThreshold;
     uint8_t m_rate;
 };
 
 NS_OBJECT_ENSURE_REGISTERED(SmartWifiManager);
 
 TypeId
 SmartWifiManager::GetTypeId()
 {
     static TypeId tid =
         TypeId("ns3::SmartWifiManager")
             .SetParent<WifiRemoteStationManager>()
             .SetGroupName("Wifi")
             .AddConstructor<SmartWifiManager>()
             .AddAttribute("SuccessK",
                           "Multiplication factor for the success threshold in the Smart algorithm.",
                           DoubleValue(2.0),
                           MakeDoubleAccessor(&SmartWifiManager::m_successK),
                           MakeDoubleChecker<double>())
             .AddAttribute("TimerK",
                           "Multiplication factor for the timer threshold in the Smart algorithm.",
                           DoubleValue(2.0),
                           MakeDoubleAccessor(&SmartWifiManager::m_timerK),
                           MakeDoubleChecker<double>())
             .AddAttribute("MaxSuccessThreshold",
                           "Maximum value of the success threshold in the Smart algorithm.",
                           UintegerValue(60),
                           MakeUintegerAccessor(&SmartWifiManager::m_maxSuccessThreshold),
                           MakeUintegerChecker<uint32_t>())
             .AddAttribute("MinTimerThreshold",
                           "The minimum value for the 'timer' threshold in the Smart algorithm.",
                           UintegerValue(15),
                           MakeUintegerAccessor(&SmartWifiManager::m_minTimerThreshold),
                           MakeUintegerChecker<uint32_t>())
             .AddAttribute("MinSuccessThreshold",
                           "The minimum value for the success threshold in the Smart algorithm.",
                           UintegerValue(10),
                           MakeUintegerAccessor(&SmartWifiManager::m_minSuccessThreshold),
                           MakeUintegerChecker<uint32_t>())
             .AddTraceSource("Rate",
                             "Traced value for rate changes (b/s)",
                             MakeTraceSourceAccessor(&SmartWifiManager::m_currentRate),
                             "ns3::TracedValueCallback::Uint64");
     return tid;
 }
 
 SmartWifiManager::SmartWifiManager()
     : WifiRemoteStationManager(),
       m_currentRate(0)
 {
     NS_LOG_FUNCTION(this);
 }
 
 SmartWifiManager::~SmartWifiManager()
 {
     NS_LOG_FUNCTION(this);
 }
 
 void
 SmartWifiManager::DoInitialize()
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
 SmartWifiManager::DoCreateStation() const
 {
     NS_LOG_FUNCTION(this);
     auto station = new SmartWifiRemoteStation();
 
     station->m_successThreshold = m_minSuccessThreshold;
     station->m_timerTimeout = m_minTimerThreshold;
     station->m_rate = 0;
     station->m_success = 0;
     station->m_failed = 0;
     station->m_recovery = false;
     station->m_timer = 0;
 
     return station;
 }
 
 void
 SmartWifiManager::DoReportRtsFailed(WifiRemoteStation* station)
 {
     NS_LOG_FUNCTION(this << station);
 }
 
 void
 SmartWifiManager::DoReportDataFailed(WifiRemoteStation* st)
 {
     NS_LOG_FUNCTION(this << st);
     auto station = static_cast<SmartWifiRemoteStation*>(st);
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
 SmartWifiManager::DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode)
 {
     NS_LOG_FUNCTION(this << station << rxSnr << txMode);
 }
 
 void
 SmartWifiManager::DoReportRtsOk(WifiRemoteStation* station,
                                double ctsSnr,
                                WifiMode ctsMode,
                                double rtsSnr)
 {
     NS_LOG_FUNCTION(this << station << ctsSnr << ctsMode << rtsSnr);
     NS_LOG_DEBUG("station=" << station << " rts ok");
 }
 
 void
 SmartWifiManager::DoReportDataOk(WifiRemoteStation* st,
                                 double ackSnr,
                                 WifiMode ackMode,
                                 double dataSnr,
                                 uint16_t dataChannelWidth,
                                 uint8_t dataNss)
 {
     NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
     auto station = static_cast<SmartWifiRemoteStation*>(st);
     station->m_timer++;
     station->m_success++;
     station->m_failed = 0;
     station->m_recovery = false;
     NS_LOG_DEBUG("station=" << station << " data ok success=" << station->m_success
                             << ", timer=" << station->m_timer);
     if ((station->m_success == station->m_successThreshold ||
          station->m_timer == station->m_timerTimeout) &&
         (station->m_rate < (GetNSupported(station) - 1)))
     {
         NS_LOG_DEBUG("station=" << station << " inc rate");
         station->m_rate++;
         station->m_timer = 0;
         station->m_success = 0;
         station->m_recovery = true;
     }
 }
 
 void
 SmartWifiManager::DoReportFinalRtsFailed(WifiRemoteStation* station)
 {
     NS_LOG_FUNCTION(this << station);
 }
 
 void
 SmartWifiManager::DoReportFinalDataFailed(WifiRemoteStation* station)
 {
     NS_LOG_FUNCTION(this << station);
 }
 
 WifiTxVector
 SmartWifiManager::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
 {
     NS_LOG_FUNCTION(this << st << allowedWidth);
     auto station = static_cast<SmartWifiRemoteStation*>(st);
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
 SmartWifiManager::DoGetRtsTxVector(WifiRemoteStation* st)
 {
     NS_LOG_FUNCTION(this << st);
     auto station = static_cast<SmartWifiRemoteStation*>(st);
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