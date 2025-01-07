/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    namespace onAcc::internal
    {
        struct CudaHipAtomic
        {
        };

        constexpr auto cudaHipAtomic = CudaHipAtomic{};

    } // namespace onAcc::internal

    namespace math::internal
    {
        struct CudaHipMath
        {
        };

        constexpr auto cudaHipMath = CudaHipMath{};
    } // namespace math::internal
} // namespace alpaka
