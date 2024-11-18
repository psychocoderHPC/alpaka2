/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka
{
    template<typename T_NumBlocks, typename T_NumThreads>
    struct ThreadBlocking
    {
        using type = typename T_NumBlocks::type;
        using vecType = T_NumBlocks;

        consteval uint32_t dim() const
        {
            return T_NumThreads::dim();
        }

        T_NumBlocks m_numBlocks;
        T_NumThreads m_numThreads;

        ThreadBlocking(T_NumBlocks const& numBlocks, T_NumThreads const& numThreads)
            : m_numBlocks(numBlocks)
            , m_numThreads(numThreads)
        {
        }
    };

    template<typename T_NumBlocks, typename T_BlockSize>
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

    template<typename T_NumBlocks, typename T_BlockSize>
    std::ostream& operator<<(std::ostream& s, DataBlocking<T_NumBlocks, T_BlockSize> const& d)
    {
        return s << "blocks=" << d.m_numBlocks << " blockSize=" << d.m_blockSize;
    }
} // namespace alpaka
