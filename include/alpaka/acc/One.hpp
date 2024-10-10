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

#include <stdexcept>
#include <tuple>

namespace alpaka
{
    template<typename T_IdxType, uint32_t T_dim>
    struct One
    {
        constexpr One(Vec<T_IdxType, T_dim> size)
        {
            if(size != count())
                throw std::invalid_argument("Extent must be one for the current layer.");
        }

        constexpr auto idx() const
        {
            return Vec<T_IdxType, T_dim>::create(0);
        }

        constexpr auto count() const
        {
            return Vec<T_IdxType, T_dim>::create(1);
        }

        template<uint32_t T_idx>
        void call(auto& acc, auto const& kernelBundle)
        {
            acc.template getLayer<T_idx>().template call<T_idx + 1>(acc, kernelBundle);
        }
    };
} // namespace alpaka
