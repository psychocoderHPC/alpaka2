/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{

    namespace layer
    {
        ALPAKA_TAG(thread);
        ALPAKA_TAG(block);
    } // namespace layer

    namespace frame
    {
        ALPAKA_TAG(thread);
        ALPAKA_TAG(block);
    } // namespace frame

    namespace internal_layer
    {
        ALPAKA_TAG(threadCommand);
    } // namespace internal_layer

    namespace mapping
    {

        struct CpuBlockSerialThreadOne
        {
        };

        constexpr CpuBlockSerialThreadOne cpuBlockSerialThreadOne;

        struct CpuBlockOmpThreadOne
        {
        };

        constexpr CpuBlockOmpThreadOne cpuBlockOmpThreadOne;

        struct CpuBlockOmpThreadOmp
        {
        };

        constexpr CpuBlockOmpThreadOmp cpuBlockOmpThreadOmp;

        struct Cuda
        {
        };

        constexpr Cuda cuda;


        constexpr auto availableMappings
            = std::make_tuple(cpuBlockSerialThreadOne, cpuBlockOmpThreadOne, cpuBlockOmpThreadOmp, cuda);

        namespace traits
        {
            template<typename T_Mapping>
            struct IsSeqMapping : std::false_type
            {
            };

            template<>
            struct IsSeqMapping<CpuBlockSerialThreadOne> : std::true_type
            {
            };

            template<>
            struct IsSeqMapping<CpuBlockOmpThreadOne> : std::true_type
            {
            };

            template<typename T_Mapping>
            constexpr bool isSeqMapping_v = IsSeqMapping<T_Mapping>::value;
        } // namespace traits

    } // namespace mapping
} // namespace alpaka
