/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_HIP

namespace alpaka::onAcc
{
    namespace unifiedCudaHip
    {
        template<typename T_IdxType, uint32_t T_dim>
        struct BlockLayer
        {
            constexpr auto idx() const
            {
                return Vec<T_IdxType, 3u>{hipBlockIdx_z, hipBlockIdx_y, hipBlockIdx_x}.template rshrink<T_dim>();
            }

            constexpr auto count() const
            {
                return Vec<T_IdxType, 3u>{hipGridDim_z, hipGridDim_y, hipGridDim_x}.template rshrink<T_dim>();
            }
        };

        template<typename T_IdxType, uint32_t T_dim>
        struct ThreadLayer
        {
            constexpr auto idx() const
            {
                return Vec<T_IdxType, 3u>{hipThreadIdx_z, hipThreadIdx_y, hipThreadIdx_x}.template rshrink<T_dim>();
            }

            constexpr auto count() const
            {
                return Vec<T_IdxType, 3u>{hipBlockDim_z, hipBlockDim_y, hipBlockDim_x}.template rshrink<T_dim>();
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onAcc

#endif
