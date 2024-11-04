/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <memory>

namespace alpaka
{
    namespace cpu
    {
        struct SingleThreadStaticShared
        {
            static constexpr uint32_t maxMemBytes = 64u * 1024u;
             std::array<uint8_t, maxMemBytes> m_data;
             uint32_t m_counter = 0u;

            template<typename T>
            T& allocVar()
            {
                T* ptr = reinterpret_cast<T*>(m_data.data() + m_counter);
                m_counter += sizeof(T);
                return *ptr;
            }

            void reset()
            {
                m_counter = 0u;
            }
        };
    } // namespace cpu
} // namespace alpaka
