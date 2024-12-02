/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include <cstdint>
#include <string>

namespace alpaka::onHost
{
    struct DeviceProperties
    {
        auto getName() const
        {
            return m_name;
        }

        std::string m_name;
        uint32_t m_multiProcessorCount;
        uint32_t m_warpSize;
        uint32_t m_maxThreadsPerBlock;
    };

    inline std::ostream& operator<<(std::ostream& s, DeviceProperties const& p)
    {
        s << "name: " << p.m_name << "\n";
        s << "multiProcessorCount: " << p.m_multiProcessorCount << "\n";
        s << "warpSize: " << p.m_warpSize << "\n";
        s << "maxThreadsPerBlock: " << p.m_maxThreadsPerBlock << "\n";
        return s;
    };
} // namespace alpaka::onHost
