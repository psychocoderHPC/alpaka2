/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA
#    include "alpaka/core/ApiCudaRt.hpp"
#elif ALPAKA_LANG_HIP
#    include "alpaka/core/ApiHipRt.hpp"
#endif

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cuda/Api.hpp"

#include <cstdint>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        template<typename T_Dest, typename T_Source>
        struct MemcpyKind
        {
            static_assert(sizeof(T_Dest) && false, "Not supported memcpy kind.");
        };

#if ALPAKA_LANG_CUDA
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
#endif
#if ALPAKA_LANG_HIP
        template<>
        struct MemcpyKind<api::Cpu, api::Hip>
        {
            static constexpr auto kind = ApiHipRt::memcpyDeviceToHost;
        };

        template<>
        struct MemcpyKind<api::Hip, api::Hip>
        {
            static constexpr auto kind = ApiHipRt::memcpyDeviceToDevice;
        };

        template<>
        struct MemcpyKind<api::Hip, api::Cpu>
        {
            static constexpr auto kind = ApiHipRt::memcpyHostToDevice;
        };
#endif
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost
