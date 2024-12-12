/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/tag.hpp"
#include "alpaka/api/trait.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::exec
{
    struct GpuCuda
    {
    };

    constexpr GpuCuda gpuCuda;
} // namespace alpaka::exec

namespace alpaka::onAcc::trait
{
    template<>
    struct GetAtomicImpl::Op<alpaka::exec::GpuCuda>
    {
        constexpr decltype(auto) operator()(alpaka::exec::GpuCuda const) const
        {
            return internal::cudaHipAtomic;
        }
    };
} // namespace alpaka::onAcc::trait
