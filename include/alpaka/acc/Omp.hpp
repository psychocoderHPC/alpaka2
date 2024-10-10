/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/acc/Layer.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/meta/NdLoop.hpp"

#include <cassert>
#include <tuple>

#ifdef _OPENMP
namespace alpaka
{
    template<typename T_IdxType, uint32_t T_dim>
    struct Omp
    {
        constexpr Omp(Vec<T_IdxType, T_dim> size) : m_extent(size)
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
#    pragma omp for nowait
            for(T_IdxType i = 0; i < m_extent.product(); ++i)
            {
                m_idx = mapToND(m_extent, i);
                acc.template getLayer<T_idx>().template call<T_idx + 1>(acc, kernelBundle);
            }
        }

    private:
        Vec<T_IdxType, T_dim> m_idx = Vec<T_IdxType, T_dim>::create(0);
        Vec<T_IdxType, T_dim> m_extent = Vec<T_IdxType, T_dim>::create(1);
    };
} // namespace alpaka
#endif
