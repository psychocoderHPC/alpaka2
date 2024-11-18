/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "../Vec.hpp"
#include "../core/Dict.hpp"
#include "../core/Tag.hpp"
#include "../core/common.hpp"
#include "../meta/NdLoop.hpp"
#include "../tag.hpp"

#include "../../../../../../../usr/include/c++/11/cassert"
#include "../../../../../../../usr/include/c++/11/tuple"

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

        template<typename T>
        constexpr decltype(auto) declareSharedVar() const
        {
            return (*this)[layer::shared].template allocVar<T>();
        }

        constexpr void syncBlockThreads() const
        {
            (*this)[action::sync]();
        }

        consteval bool hasKey(auto key) const
        {
            return hasTag(static_cast<T_Storage>(*this));
        }
    };

} // namespace alpaka::onAcc
