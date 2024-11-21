/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka
{
    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
    struct ThreadBlocking
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

        ThreadBlocking(T_NumBlocks const& numBlocks, T_NumThreads const& numThreads)
            : m_numBlocks(numBlocks)
            , m_numThreads(numThreads)
        {
        }
    };

    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
    ThreadBlocking(T_NumBlocks const&, T_NumThreads const&) -> ThreadBlocking<T_NumBlocks, T_NumThreads>;

    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_BlockSize>
    struct DataBlocking
    {
        using type = typename T_NumBlocks::type;

        consteval uint32_t dim() const
        {
            return T_BlockSize::dim();
        }

        T_NumBlocks m_numBlocks;
        T_BlockSize m_blockSize;
        T_BlockSize m_numThreads;

        DataBlocking(T_NumBlocks const& numBlocks, T_BlockSize const& blockSize)
            : m_numBlocks(numBlocks)
            , m_blockSize(blockSize)
            , m_numThreads(blockSize)
        {
        }

        DataBlocking(T_NumBlocks const& numBlocks, T_BlockSize const& blockSize, T_BlockSize const& numThreads)
            : m_numBlocks(numBlocks)
            , m_blockSize(blockSize)
            , m_numThreads(numThreads)
        {
        }
    };

    template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_BlockSize>
    std::ostream& operator<<(std::ostream& s, DataBlocking<T_NumBlocks, T_BlockSize> const& d)
    {
        return s << "blocks=" << d.m_numBlocks << " blockSize=" << d.m_blockSize;
    }
} // namespace alpaka
