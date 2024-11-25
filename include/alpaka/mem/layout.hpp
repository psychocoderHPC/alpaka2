/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc::iter
{
    namespace layout
    {
        struct Strided
        {
        };

        struct Contigious
        {
        };

        struct Optimized
        {
        };

        constexpr auto strided = Strided{};
        constexpr auto contigious = Contigious{};
        constexpr auto optimized = Optimized{};

    } // namespace layout
} // namespace alpaka::onAcc::iter
