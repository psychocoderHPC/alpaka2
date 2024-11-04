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
    template<typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc const&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;

        template<typename T>
        constexpr decltype(auto) allocVar() const
        {
            return (*this)[layer::shared].template allocVar<T>();
        }

        constexpr void sync() const
        {
            (*this)[action::sync]();
        }
    };

} // namespace alpaka
