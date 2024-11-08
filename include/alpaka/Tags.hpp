/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"

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

    namespace mapping
    {
        struct Empty
        {
        };

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
