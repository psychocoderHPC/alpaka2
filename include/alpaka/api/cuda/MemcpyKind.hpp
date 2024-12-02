/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA
#    include "alpaka/api/cpu/Api.hpp"
#    include "alpaka/api/cuda/Api.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"

#    include <cstdint>

namespace alpaka::onHost
{
    namespace cuda
    {
        template<typename T_Dest, typename T_Source>
        struct MemcpyKind
        {
            static_assert(sizeof(T_Dest) && false, "Not supported memcpy kind.");
        };

        template<>
        struct MemcpyKind<api::Cpu, api::Cuda>
        {
            static constexpr auto kind = ApiCudaRt::memcpyDeviceToHost;
        };

        template<>
        struct MemcpyKind<api::Cuda, api::Cuda>
        {
            static constexpr auto kind = ApiCudaRt::memcpyDeviceToDevice;
        };

        template<>
        struct MemcpyKind<api::Cuda, api::Cpu>
        {
            static constexpr auto kind = ApiCudaRt::memcpyHostToDevice;
        };

    } // namespace cuda
} // namespace alpaka::onHost
#endif
