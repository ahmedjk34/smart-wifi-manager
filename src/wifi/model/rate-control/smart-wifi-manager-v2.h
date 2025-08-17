/*
 * Copyright (c) 2004,2005,2006 INRIA
 * ... (license text unchanged)
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Modified for SmartWifiManagerV2: ahmedjk34
 */

#ifndef SMART_WIFI_MANAGER_V2_H
#define SMART_WIFI_MANAGER_V2_H

#include "ns3/wifi-remote-station-manager.h"
#include "ns3/traced-value.h"

namespace ns3
{

class SmartWifiManagerV2 : public WifiRemoteStationManager
{
public:
    static TypeId GetTypeId(void);
    SmartWifiManagerV2();
    virtual ~SmartWifiManagerV2();

protected:
    virtual void DoInitialize(void) override;
    virtual WifiRemoteStation* DoCreateStation(void) const override;
    virtual void DoReportRtsFailed(WifiRemoteStation* station) override;
    virtual void DoReportDataFailed(WifiRemoteStation* station) override;
    virtual void DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode) override;
    virtual void DoReportRtsOk(WifiRemoteStation* station, double ctsSnr, WifiMode ctsMode, double rtsSnr) override;
    virtual void DoReportDataOk(WifiRemoteStation* station, double ackSnr, WifiMode ackMode, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
    virtual void DoReportFinalRtsFailed(WifiRemoteStation* station) override;
    virtual void DoReportFinalDataFailed(WifiRemoteStation* station) override;
    virtual WifiTxVector DoGetDataTxVector(WifiRemoteStation* station, uint16_t allowedWidth) override;
    virtual WifiTxVector DoGetRtsTxVector(WifiRemoteStation* station) override;

    // V2 attributes (add anything new here)
    double m_successK;
    double m_timerK;
    uint32_t m_maxSuccessThreshold;
    uint32_t m_minTimerThreshold;
    uint32_t m_minSuccessThreshold;

    ns3::TracedValue<uint64_t> m_currentRate;
};

} // namespace ns3

#endif // SMART_WIFI_MANAGER_V2_H