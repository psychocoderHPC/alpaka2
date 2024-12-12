/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc
{
    //! Defines the parallelism hierarchy levels of alpaka
    namespace hierarchy
    {
        struct Grids
        {
        };

        constexpr auto grids = Grids{};

        struct Blocks
        {
        };

        constexpr auto blocks = Blocks{};

        struct Threads
        {
        };

        constexpr auto threads = Threads{};
    } // namespace hierarchy
} // namespace alpaka::onAcc
