/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/acc/Layer.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{
    template<typename T_IdxType, uint32_t T_dim>
    struct Serial
    {
        constexpr Serial(Vec<T_IdxType, T_dim> size) : m_extent(size)
        {
        }

        constexpr auto idx() const
        {
            return m_idx;
        }

        constexpr auto count() const
        {
            return m_extent;
        }

        template<uint32_t T_idx>
        void call(auto& acc, auto const& kernelBundle)
        {
            m_idx = Vec<T_IdxType, T_dim>::create(0);
            meta::ndLoopIncIdx(
                m_idx,
                m_extent,
                [&](auto const& currentIdx)
                { acc.template getLayer<T_idx>().template call<T_idx + 1>(acc, kernelBundle); });
        }

    private:
        Vec<T_IdxType, T_dim> m_idx = Vec<T_IdxType, T_dim>::create(0);
        Vec<T_IdxType, T_dim> m_extent = Vec<T_IdxType, T_dim>::create(1);
    };

} // namespace alpaka
