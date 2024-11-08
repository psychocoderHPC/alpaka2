/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/HostApiConcepts.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/Handle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/filter.hpp"

#include <type_traits>

namespace alpaka
{
    namespace trait
    {
        struct IsPlatformAvailable
        {
            template<concepts::Api T_Api>
            struct Op : std::false_type
            {
            };
        };

        struct IsMappingSupportedBy
        {
            template<typename T_Mapping, typename T_Device>
            struct Op : std::false_type
            {
            };
        };

        template<typename T_Mapping, concepts::DeviceHandle T_DeviceHandle>
        struct IsMappingSupportedBy::Op<T_Mapping, T_DeviceHandle>
            : IsMappingSupportedBy::Op<T_Mapping, typename T_DeviceHandle::element_type>
        {
        };

    } // namespace trait

    consteval bool isPlatformAvaiable(concepts::Api auto api)
    {
        return trait::IsPlatformAvailable::Op<std::decay_t<decltype(api)>>::value;
    }

    constexpr auto supportedMappings(concepts::DeviceHandle auto deviceHandle)
    {
        return meta::filter(
            [&](auto mapping) constexpr
            {
                return trait::IsMappingSupportedBy::
                    Op<std::decay_t<decltype(mapping)>, std::decay_t<decltype(deviceHandle)>>::value;
            },
            mapping::availableMappings);
    }
} // namespace alpaka
