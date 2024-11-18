/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#if ALPAKA_OMP

#    include "alpaka/Blocking.hpp"
#    include "alpaka/Vec.hpp"
#    include "alpaka/api/cpu/IdxLayer.hpp"
#    include "alpaka/api/cpu/block/mem/SingleThreadStaticShared.hpp"
#    include "alpaka/api/cpu/block/sync/NoOp.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/meta/NdLoop.hpp"
#    include "alpaka/onAcc/Acc.hpp"
#    include "alpaka/tag.hpp"

#    include <cassert>
#    include <stdexcept>
#    include <tuple>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_NumBlocks, typename T_NumThreads>
        struct OmpBlocks
        {
            using IndexVecType = typename ThreadBlocking<T_NumBlocks, T_NumThreads>::vecType;
            using IndexType = typename IndexVecType::type;

            constexpr OmpBlocks(ThreadBlocking<T_NumBlocks, T_NumThreads> threadBlocking)
                : m_threadBlocking{std::move(threadBlocking)}
            {
            }

            void operator()(auto const& kernelBundle) const
            {
                this->operator()(kernelBundle, Dict{DictEntry{alpaka::Empty{}, alpaka::Empty{}}});
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                if(m_threadBlocking.m_numThreads.product() != 1u)
                    throw std::runtime_error("Thread block extent must be 1.");
#    pragma omp parallel
                {
                    // copy from num blocks to derive correct index type
                    auto blockIdx = m_threadBlocking.m_numBlocks;
                    auto blockSharedMem = onAcc::cpu::SingleThreadStaticShared{};
                    auto blockCount = m_threadBlocking.m_numBlocks;

                    auto const blockLayerEntry = DictEntry{
                        layer::block,
                        onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCount)}};
                    auto const threadLayerEntry = DictEntry{layer::thread, onAcc::cpu::OneLayer<IndexVecType>{}};
                    auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                    auto const blockSyncEntry = DictEntry{action::sync, onAcc::cpu::NoOp{}};

                    auto acc = onAcc::Acc(
                        joinDict(Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry}, dict));

#    pragma omp for nowait
                    for(IndexType i = 0; i < blockCount.product(); ++i)
                    {
                        blockIdx = mapToND(blockCount, i);
                        kernelBundle(acc);
                        blockSharedMem.reset();
                    }
                }
            }

            ThreadBlocking<T_NumBlocks, T_NumThreads> m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(exec::CpuOmpBlocks, auto const& threadBlocking)
    {
        return cpu::OmpBlocks(threadBlocking);
    }
} // namespace alpaka::onHost

#endif
