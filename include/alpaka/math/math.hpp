/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>

namespace alpaka::math::internal
{
    struct StlMath
    {
    };

    constexpr auto stlMath = StlMath{};

    struct Sin
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::sin;
                return sin(arg);
            }
        };
    };

    struct Exp
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Op
        {
            constexpr auto operator()(T_MathImpl, T_Arg const& arg) const
            {
                using std::exp;
                return exp(arg);
            }
        };
    };
} // namespace alpaka::math::internal
