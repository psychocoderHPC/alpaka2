/* Copyright 2024 Jeffrey Kelling, Rene Widera, Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA

#    include "alpaka/core/common.hpp"

namespace alpaka
{
    namespace cuda
    {

        struct Sync
        {
            __device__ void operator()() const
            {
                __syncthreads();
            }
        };

        struct StaticShared
        {
            template<typename T>
            __device__ T& allocVar() const
            {
                __shared__ uint8_t shMem alignas(alignof(T))[sizeof(T)];
                return *(reinterpret_cast<T*>(shMem));
            }
        };
    } // namespace cuda
} // namespace alpaka

#endif
