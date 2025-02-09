/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/math/math.hpp"

#include <cmath>

namespace alpaka::math
{
    constexpr auto sin(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Sin::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }

    constexpr auto exp(auto const& arg)
    {
        auto const mathImpl = trait::getMathImpl(thisApi());
        return internal::Exp::Op<ALPAKA_TYPEOF(mathImpl), ALPAKA_TYPEOF(arg)>{}(mathImpl, arg);
    }
} // namespace alpaka::math
