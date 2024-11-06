/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    namespace cpu
    {
        struct NoOp
        {
            constexpr void operator()() const
            {
            }
        };
    } // namespace cpu
} // namespace alpaka
