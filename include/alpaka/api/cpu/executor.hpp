/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/tag.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::exec
{
    struct CpuSerial
    {
    };

    constexpr CpuSerial cpuSerial;

    struct CpuOmpBlocks
    {
    };

    constexpr CpuOmpBlocks cpuOmpBlocks;

    struct CpuOmpBlocksAndThreads
    {
    };

    constexpr CpuOmpBlocksAndThreads cpuOmpBlocksAndThreads;

    namespace traits
    {
        template<>
        struct IsSeqExecutor<CpuSerial> : std::true_type
        {
        };

        template<>
        struct IsSeqExecutor<CpuOmpBlocks> : std::true_type
        {
        };
    } // namespace traits
} // namespace alpaka::exec

namespace alpaka::onAcc::trait
{
    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuSerial>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuSerial const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuOmpBlocks>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuOmpBlocks const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuOmpBlocksAndThreads>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuOmpBlocksAndThreads const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };
} // namespace alpaka::onAcc::trait
