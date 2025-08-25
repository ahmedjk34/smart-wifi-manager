#ifndef SMART_WIFI_MANAGER_ML_COMMON_H
#define SMART_WIFI_MANAGER_ML_COMMON_H

#include "ns3/wifi-mode.h"
#include <vector>

namespace ns3
{

inline std::vector<WifiMode> GetMlModeLookup()
{
    static std::vector<WifiMode> modes = {
        WifiMode("DsssRate1Mbps"),    // 0
        WifiMode("DsssRate2Mbps"),    // 1
        WifiMode("DsssRate5_5Mbps"),  // 2
        WifiMode("DsssRate11Mbps"),   // 3
        WifiMode("OfdmRate6Mbps"),    // 4
        WifiMode("OfdmRate9Mbps"),    // 5
        WifiMode("OfdmRate12Mbps"),   // 6
        WifiMode("OfdmRate18Mbps"),   // 7
        WifiMode("OfdmRate24Mbps"),   // 8
        WifiMode("OfdmRate36Mbps"),   // 9
        WifiMode("OfdmRate48Mbps"),   // 10
        WifiMode("OfdmRate54Mbps")    // 11
    };
    return modes;
}

} // namespace ns3

#endif