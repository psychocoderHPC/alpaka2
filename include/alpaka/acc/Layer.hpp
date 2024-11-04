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
        template<typename IndexVecType>
        struct OneLayer
        {
            constexpr OneLayer() = default;

            constexpr auto idx() const
            {
                return IndexVecType::create(0);
            }

            constexpr auto count() const
            {
                return IndexVecType::create(1);
            }
        };

        template<typename T_Idx, typename T_Count>
        struct GenericLayer
        {
            constexpr GenericLayer(T_Idx idx, T_Count count) : m_idx(idx), m_count(count)
            {
            }

            constexpr decltype(auto) idx() const
            {
                return unWrapp(m_idx);
            }

            constexpr decltype(auto) count() const
            {
                return unWrapp(m_count);
            }

            T_Idx m_idx;
            T_Count m_count;
        };

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
