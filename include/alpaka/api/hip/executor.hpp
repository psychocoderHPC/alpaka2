/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/api/unifiedCudaHip/trait.hpp"

namespace alpaka::exec
{
    struct GpuHip
    {
    };

    constexpr GpuHip gpuHip;
} // namespace alpaka::exec

namespace alpaka::onAcc::trait
{
    template<>
    struct GetAtomicImpl::Op<alpaka::exec::GpuHip>
    {
        constexpr decltype(auto) operator()(alpaka::exec::GpuHip const) const
        {
            return internal::cudaHipAtomic;
        }
    };
} // namespace alpaka::onAcc::trait

namespace alpaka::unifiedCudaHip::trait
{
    template<>
    struct IsUnifiedExecutor<alpaka::exec::GpuHip> : std::true_type
    {
    };
} // namespace alpaka::unifiedCudaHip::trait
