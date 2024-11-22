/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/core/Unreachable.hpp"

#include <cmath>

namespace alpaka::math
{
    namespace internal
    {
        struct Sin
        {
            template<typename T_Api, typename T_Arg>
            struct Op
            {
                constexpr auto operator()(T_Api, T_Arg const& arg) const
                {
                    using std::sin;
                    return sin(arg);
                }
            };
        };

        struct Exp
        {
            template<typename T_Api, typename T_Arg>
            struct Op
            {
                constexpr auto operator()(T_Api, T_Arg const& arg) const
                {
                    using std::exp;
                    return exp(arg);
                }
            };
        };
#if ALPAKA_LANG_CUDA
        //! The CUDA sin trait specialization for real types.
        template<typename T_Arg>
        requires(std::is_floating_point_v<T_Arg>)
        struct Sin::Op<api::Cuda, T_Arg>
        {
            constexpr auto operator()(api::Cuda, T_Arg const& arg)
            {
                if constexpr(is_decayed_v<T_Arg, float>)
                    return ::sinf(arg);
                else if constexpr(is_decayed_v<T_Arg, double>)
                    return ::sin(arg);
                else
                    static_assert(!sizeof(T_Arg), "Unsupported data type");

                ALPAKA_UNREACHABLE(T_Arg{});
            }
        };

        template<typename T_Arg>
        requires(std::is_floating_point_v<T_Arg>)
        struct Exp::Op<api::Cuda, T_Arg>
        {
            constexpr auto operator()(api::Cuda, T_Arg const& arg)
            {
                if constexpr(is_decayed_v<T_Arg, float>)
                    return ::expf(arg);
                else if constexpr(is_decayed_v<T_Arg, double>)
                    return ::exp(arg);
                else
                    static_assert(!sizeof(T_Arg), "Unsupported data type");

                ALPAKA_UNREACHABLE(T_Arg{});
            }
        };
#endif
    } // namespace internal

    constexpr auto sin(auto const& arg)
    {
        return internal::Sin::Op<ALPAKA_TYPE(apiCtx),ALPAKA_TYPE(arg)>{}(apiCtx, arg);
    }

    constexpr auto exp(auto const& arg)
    {
        return internal::Exp::Op<ALPAKA_TYPE(apiCtx),ALPAKA_TYPE(arg)>{}(apiCtx, arg);
    }
} // namespace alpaka::math
