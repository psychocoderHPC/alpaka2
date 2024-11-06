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

    namespace cpu
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

    } // namespace cpu
} // namespace alpaka
