/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/math/math.hpp"

#include <cmath>

namespace alpaka::math
{
    constexpr auto sin(auto const& arg)
    {
        return internal::Sin::Op<ALPAKA_TYPEOF(apiCtx), ALPAKA_TYPEOF(arg)>{}(apiCtx, arg);
    }

    constexpr auto exp(auto const& arg)
    {
        return internal::Exp::Op<ALPAKA_TYPEOF(apiCtx), ALPAKA_TYPEOF(arg)>{}(apiCtx, arg);
    }
} // namespace alpaka::math
