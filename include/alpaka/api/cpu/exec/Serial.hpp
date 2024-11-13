/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Acc.hpp"
#include "alpaka/Blocking.hpp"
#include "alpaka/Tags.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/cpu/IdxLayer.hpp"
#include "alpaka/api/cpu/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/cpu/block/sync/NoOp.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/NdLoop.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{
    namespace cpu
    {
        template<typename T_NumBlocks, typename T_NumThreads>
        struct Serial
        {
            using IndexVecType = typename ThreadBlocking<T_NumBlocks, T_NumThreads>::vecType;

            constexpr Serial(ThreadBlocking<T_NumBlocks, T_NumThreads> threadBlocking)
                : m_threadBlocking{std::move(threadBlocking)}
            {
            }

            void operator()(auto const& kernelBundle) const
            {
                this->operator()(kernelBundle, Dict{DictEntry{alpaka::Empty{}, alpaka::Empty{}}});
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                // copy from num blocks to derive correct index type
                auto blockIdx = m_threadBlocking.m_numBlocks;
                auto blockSharedMem = cpu::SingleThreadStaticShared{};

                auto const blockLayerEntry = DictEntry{
                    layer::block,
                    alpaka::cpu::GenericLayer{std::cref(blockIdx), std::cref(m_threadBlocking.m_numBlocks)}};
                auto const threadLayerEntry = DictEntry{layer::thread, cpu::OneLayer<IndexVecType>{}};
                auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                auto const blockSyncEntry = DictEntry{action::sync, cpu::NoOp{}};

                auto acc = Acc(
                    joinDict(Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry}, dict));
                meta::ndLoopIncIdx(
                    blockIdx,
                    m_threadBlocking.m_numBlocks,
                    [&](auto const&)
                    {
                        kernelBundle(acc);
                        acc[layer::shared].reset();
                    });
            }

            ThreadBlocking<T_NumBlocks, T_NumThreads> m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(exec::CpuSerial, auto const& threadBlocking)
    {
        return cpu::Serial(threadBlocking);
    }
} // namespace alpaka
