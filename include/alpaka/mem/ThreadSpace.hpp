/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka
{
    template<typename T_ThreadIdx, typename T_ThreadCount>
    struct ThreadSpace
    {
        constexpr ThreadSpace(T_ThreadIdx const& threadIdx, T_ThreadCount const& threadCount)
            : m_threadIdx(threadIdx)
            , m_threadCount(threadCount)
        {
        }

        std::string toString(std::string const separator = ",", std::string const enclosings = "{}") const
        {
            std::string locale_enclosing_begin;
            std::string locale_enclosing_end;
            size_t enclosing_dim = enclosings.size();

            if(enclosing_dim > 0)
            {
                /* % avoid out of memory access */
                locale_enclosing_begin = enclosings[0 % enclosing_dim];
                locale_enclosing_end = enclosings[1 % enclosing_dim];
            }

            std::stringstream stream;
            stream << locale_enclosing_begin;
            stream << m_threadIdx << separator << m_threadCount;
            stream << locale_enclosing_end;
            return stream.str();
        }

        T_ThreadIdx m_threadIdx;
        T_ThreadCount m_threadCount;
    };
} // namespace alpaka
