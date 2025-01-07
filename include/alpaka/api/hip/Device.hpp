/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_HIP
#    include "alpaka/api/unifiedCudaHip/Device.hpp"
#    include "alpaka/onHost/trait.hpp"

#    include <type_traits>

namespace alpaka::onHost
{
    namespace trait
    {
        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<exec::GpuHip, unifiedCudaHip::Device<T_Platform>> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka::onHost

#endif
