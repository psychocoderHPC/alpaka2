/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc::internal
{
    struct CudaHipAtomic
    {
    };

    constexpr auto cudaHipAtomic = CudaHipAtomic{};
} // namespace alpaka::onAcc::internal
