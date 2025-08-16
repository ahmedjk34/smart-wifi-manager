/*
 * Copyright (c) 2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

 #ifndef SMART_WIFI_MANAGER_H
 #define SMART_WIFI_MANAGER_H
 
 #include "ns3/traced-value.h"
 #include "ns3/wifi-remote-station-manager.h"
 
 namespace ns3
 {
 
 /**
  * \brief Smart Rate control algorithm
  * \ingroup wifi
  *
  * This class implements the Smart rate control algorithm.
  * This RAA does not support HT modes and will error
  * exit if the user tries to configure this RAA with a Wi-Fi MAC
  * that supports 802.11n or higher.
  */
 class SmartWifiManager : public WifiRemoteStationManager
 {
   public:
     /**
      * \brief Get the type ID.
      * \return the object TypeId
      */
     static TypeId GetTypeId();
     SmartWifiManager();
     ~SmartWifiManager() override;
 
   private:
     void DoInitialize() override;
     WifiRemoteStation* DoCreateStation() const override;
     void DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode) override;
     void DoReportRtsFailed(WifiRemoteStation* station) override;
     void DoReportDataFailed(WifiRemoteStation* station) override;
     void DoReportRtsOk(WifiRemoteStation* station,
                        double ctsSnr,
                        WifiMode ctsMode,
                        double rtsSnr) override;
     void DoReportDataOk(WifiRemoteStation* station,
                         double ackSnr,
                         WifiMode ackMode,
                         double dataSnr,
                         uint16_t dataChannelWidth,
                         uint8_t dataNss) override;
     void DoReportFinalRtsFailed(WifiRemoteStation* station) override;
     void DoReportFinalDataFailed(WifiRemoteStation* station) override;
     WifiTxVector DoGetDataTxVector(WifiRemoteStation* station, uint16_t allowedWidth) override;
     WifiTxVector DoGetRtsTxVector(WifiRemoteStation* station) override;
 
     uint32_t m_minTimerThreshold;
     uint32_t m_minSuccessThreshold;
     double m_successK;
     uint32_t m_maxSuccessThreshold;
     double m_timerK;
 
     TracedValue<uint64_t> m_currentRate;
 };
 
 } // namespace ns3
 
 #endif /* SMART_WIFI_MANAGER_H */