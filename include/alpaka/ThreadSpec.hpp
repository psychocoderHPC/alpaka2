/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka
{
    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
    struct ThreadSpec
    {
        using type = typename T_NumBlocks::type;
        using NumBlocksVecType = typename T_NumBlocks::UniVec;
        using NumThreadsVecType = typename T_NumThreads::UniVec;

        consteval uint32_t dim() const
        {
            return T_NumThreads::dim();
        }

        NumBlocksVecType m_numBlocks;
        NumThreadsVecType m_numThreads;

        ThreadSpec(T_NumBlocks const& numBlocks, T_NumThreads const& numThreadsPerBlock)
            : m_numBlocks(numBlocks)
            , m_numThreads(numThreadsPerBlock)
        {
        }
    };

    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
    ThreadSpec(T_NumBlocks const&, T_NumThreads const&) -> ThreadSpec<T_NumBlocks, T_NumThreads>;
} // namespace alpaka
