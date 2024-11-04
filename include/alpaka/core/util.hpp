/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cstdio>
#include <tuple>
#include <utility>

namespace alpaka
{
    template<typename T>
    constexpr decltype(auto) unWrapp(T && value)
    {
        using WrappedType = std::unwrap_reference_t<std::decay_t<decltype(value)>>;
        return std::unwrap_reference_t<WrappedType>(std::forward<T>(value));
    }
} // namespace alpaka
