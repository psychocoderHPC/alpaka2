/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/api/unifiedCudaHip/trait.hpp"

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

namespace alpaka::unifiedCudaHip::trait
{
    template<>
    struct IsUnifiedExecutor<alpaka::exec::GpuCuda> : std::true_type
    {
    };
} // namespace alpaka::unifiedCudaHip::trait
