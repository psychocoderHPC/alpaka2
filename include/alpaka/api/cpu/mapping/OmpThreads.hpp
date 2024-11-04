/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#if ALPAKA_OMP

#    include "alpaka/Blocking.hpp"
#    include "alpaka/Vec.hpp"
#    include "alpaka/acc/Acc.hpp"
#    include "alpaka/acc/Layer.hpp"
#    include "alpaka/api/cpu/blockSync/Omp.hpp"
#    include "alpaka/api/cpu/mem/OmpStaticShared.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/meta/NdLoop.hpp"

#    include <cassert>
#    include <stdexcept>
#    include <tuple>

namespace alpaka
{
    namespace cpu
    {
        template<typename T_NumBlocks, typename T_NumThreads>
        struct OmpThreads
        {
            using IndexVecType = typename ThreadBlocking<T_NumBlocks, T_NumThreads>::vecType;
            using IndexType = typename IndexVecType::type;

            constexpr OmpThreads(ThreadBlocking<T_NumBlocks, T_NumThreads> threadBlocking)
                : m_threadBlocking{std::move(threadBlocking)}
            {
            }

            void operator()(auto const& kernelBundle) const
            {
                this->operator()(kernelBundle, Dict{DictEntry{alpaka::mapping::Empty{}, alpaka::mapping::Empty{}}});
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                // open scope to threadsafe set omp nested and dynamic
#    pragma omp parallel num_threads(1)
                {
#    pragma omp single
                    {
                        // affects all new parallel regions started by this thread
                        ::omp_set_dynamic(0);
                        ::omp_set_nested(2);
                    }
#    pragma omp parallel
                    {
                        // copy from num blocks to derive correct index type
                        auto blockIdx = m_threadBlocking.m_numBlocks;
                        auto blockSharedMem = cpu::OmpStaticShared{};
                        auto blockCountND = m_threadBlocking.m_numBlocks;
                        auto threadCountND = m_threadBlocking.m_numThreads;
                        auto const threadCount = threadCountND.product();

                        auto const blockLayerEntry = DictEntry{
                            layer::block,
                            alpaka::mapping::GenericLayer{std::cref(blockIdx), std::cref(blockCountND)}};
                        auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                        auto const blockSyncEntry = DictEntry{action::sync, cpu::OmpSync{}};

#        pragma omp for
                        for(IndexType i = 0; i < blockCountND.product(); ++i)
                        {
                            blockIdx = mapToND(blockCountND, i);
#        pragma omp parallel num_threads(threadCount)
                            {
                                int x = omp_get_thread_num();
                                IndexVecType threadIdx = mapToND(threadCountND, static_cast<IndexType>(x));


                                //usleep(100 * x * i);
                                //std::cout << "ompt threadIdx = " << threadIdx << std::endl;

                                auto const threadLayerEntry = DictEntry{
                                    layer::thread,
                                    alpaka::mapping::GenericLayer{std::cref(threadIdx), std::cref(threadCountND)}};
                                auto acc = Acc(joinDict(
                                    Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry},
                                    dict));
                                kernelBundle(acc);
                            }
                            blockSharedMem.reset();
                        }
#    endif
                    }
                }
            }

            ThreadBlocking<T_NumBlocks, T_NumThreads> m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(mapping::CpuBlockOmpThreadOmp, auto const& threadBlocking)
    {
        return cpu::OmpThreads(threadBlocking);
    }
} // namespace alpaka
