/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/decay.hpp"

#include <cmath>

namespace alpaka::math::internal
{

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

} // namespace alpaka::math::internal
