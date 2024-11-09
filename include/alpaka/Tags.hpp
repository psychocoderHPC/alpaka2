/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/core/PP.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{
    namespace object
    {
        struct Api
        {
        };

        constexpr Api api;

        ALPAKA_TAG(exec);
    } // namespace object

    namespace layer
    {
        ALPAKA_TAG(thread);
        ALPAKA_TAG(block);
        ALPAKA_TAG(shared);
    } // namespace layer

    namespace frame
    {
        ALPAKA_TAG(count);
        ALPAKA_TAG(extent);
    } // namespace frame

    namespace action
    {
        ALPAKA_TAG(sync);
    } // namespace action


    struct Empty
    {
    };

    namespace exec
    {
        struct CpuSerial
        {
        };

        constexpr CpuSerial cpuSerial;

        struct CpuOmpBlocks
        {
        };

        constexpr CpuOmpBlocks cpuOmpBlocks;

        struct CpuOmpBlocksAndThreads
        {
        };

        constexpr CpuOmpBlocksAndThreads cpuOmpBlocksAndThreads;

        struct GpuCuda
        {
        };

        constexpr GpuCuda gpuCuda;

        constexpr auto availableMappings = std::make_tuple(ALPAKA_PP_REMOVE_FIRST_COMMA(
#ifndef ALPAKA_DISABLE_EXEC_CpuSerial
            ,
            cpuSerial
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocks
            ,
            cpuOmpBlocks
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuBlockAndThreads
            ,
            cpuOmpBlocksAndThreads
#endif
#ifndef ALPAKA_DISABLE_EXEC_GpuCuda
            ,
            gpuCuda
#endif
            ));

        namespace traits
        {
            template<typename T_Mapping>
            struct IsSeqMapping : std::false_type
            {
            };

            template<>
            struct IsSeqMapping<CpuSerial> : std::true_type
            {
            };

            template<>
            struct IsSeqMapping<CpuOmpBlocks> : std::true_type
            {
            };

            template<typename T_Mapping>
            constexpr bool isSeqMapping_v = IsSeqMapping<T_Mapping>::value;
        } // namespace traits
    } // namespace mapping
} // namespace alpaka
