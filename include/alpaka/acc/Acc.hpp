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
    struct ThreadCommand
    {
        constexpr ThreadCommand() = default;

        template<uint32_t T_idx>
        ALPAKA_FN_ACC void call(auto const& acc, auto const& kernelBundle) const
        {
            std::apply(kernelBundle.m_kernelFn, std::tuple_cat(std::tie(acc), kernelBundle.m_args));
        }
    };

    template<typename T_IdxLayers, typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_IdxLayers const&, T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc const&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;

        ALPAKA_FN_ACC void operator()(auto kernelBundle)
        {
            // constexpr auto l = layer::block;
            (*this)[layer::block].template call<1>(*this, kernelBundle);
        }

        template<uint32_t T_idx>
        constexpr decltype(auto) getLayer() const
        {
            return (*this)[std::get<T_idx>(IdxLayer{})];
        }

        template<uint32_t T_idx>
        constexpr decltype(auto) getLayer()
        {
            return (*this)[std::get<T_idx>(IdxLayer{})];
        }

        using IdxLayer = T_IdxLayers;
        using IdxType = decltype(std::declval<Acc>()[layer::block].count());
    };
} // namespace alpaka
