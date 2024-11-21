/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/MdSpan.hpp"
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

    template<typename T_Array>
    constexpr decltype(auto) declareSharedArray(auto const& acc)
    {
        return MdSpanArray<T_Array>{internalCompute::declareSharedVar<T_Array>(acc)};
    }

} // namespace alpaka::onAcc
