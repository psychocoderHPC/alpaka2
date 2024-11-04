/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/core/common.hpp"
#if ALPAKA_OMP

#    include "alpaka/api/cpu/mem/SingleThreadStaticShared.hpp"

namespace alpaka
{
    namespace cpu
    {
        struct OmpStaticShared : SingleThreadStaticShared
        {
            template<typename T>
            T& allocVar()
            {
#    pragma omp barrier
                T* ptr = reinterpret_cast<T*>(m_data.data() + m_counter);
#    pragma omp barrier
#    pragma omp single nowait
                {
                    this->m_counter += sizeof(T);
                }
                return *ptr;
            }
        };
    } // namespace cpu
} // namespace alpaka

#endif
