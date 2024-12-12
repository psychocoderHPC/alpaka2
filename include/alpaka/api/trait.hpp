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
            constexpr decltype(auto) operator()(T_Api const) const
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

namespace alpaka::onAcc::trait
{
    /** Defines the implementation used for atomic operations toghether with the used executor */
    struct GetAtomicImpl
    {
        template<typename T_Executor>
        struct Op
        {
            constexpr decltype(auto) operator()(T_Executor const) const
            {
                static_assert(
                    sizeof(T_Executor) && false,
                    "Atomic implementation for the current used executor is not defined.");
                return 0;
            }
        };
    };

    template<typename T_Executor>
    constexpr decltype(auto) getAtomicImpl(T_Executor const executor)
    {
        return GetAtomicImpl::Op<T_Executor>{}(executor);
    }
} // namespace alpaka::onAcc::trait
