/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onAcc/internal.hpp"

namespace alpaka::onAcc
{
    constexpr void syncBlockThreads(auto const& acc)
    {
        internalCompute::syncBlockThreads(acc);
    }

    template<typename T>
    constexpr decltype(auto) declareSharedVar(auto const& acc)
    {
        return internalCompute::declareSharedVar<T>(acc);
    }
} // namespace alpaka::onAcc
