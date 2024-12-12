/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/math/math.hpp"

#include <algorithm>

namespace alpaka::trait
{
    /** Map's all API's by default to stl math functions. */
    struct GetMathImpl
    {
        template<typename T_Api>
        struct Op
        {
            constexpr decltype(auto) operator()(alpaka::api::Cpu const) const
            {
                return alpaka::math::internal::stlMath;
            }
        };
    };

    template<typename T_Api>
    constexpr decltype(auto) getMathImpl(T_Api const api)
    {
        return GetMathImpl::Op<T_Api>{}(api);
    }
} // namespace alpaka::trait
