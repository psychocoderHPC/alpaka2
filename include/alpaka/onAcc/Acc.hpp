/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    template<typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc const&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;

        template<typename T, size_t T_uniqueId>
        constexpr decltype(auto) declareSharedVar() const
        {
            return alpaka::onAcc::declareSharedVar<T, T_uniqueId>(*this);
        }

        template<typename T, size_t T_uniqueId>
        constexpr decltype(auto) declareSharedMdArray(alpaka::concepts::CVector auto const& extent) const
        {
            return alpaka::onAcc::declareSharedMdArray<T, T_uniqueId>(*this, extent);
        }

        template<typename T>
        constexpr auto getDynSharedMem(auto const& acc) -> T*
        {
            return alpaka::onAcc::getDynSharedMem<T>(acc);
        }

        constexpr void syncBlockThreads() const
        {
            (*this)[action::sync]();
        }

        static constexpr bool hasKey(auto key)
        {
            constexpr auto idx = Idx<ALPAKA_TYPEOF(key), std::decay_t<T_Storage>>::value;
            return idx != -1;
        }
    };

} // namespace alpaka::onAcc
