/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA

namespace alpaka::onAcc
{
    namespace unifiedCudaHip
    {
        template<typename T_NumBlocksType>
        struct BlockLayer
        {
            constexpr auto idx() const
            {
                return Vec<typename T_NumBlocksType::type, 3u>{::blockIdx.z, ::blockIdx.y, ::blockIdx.x}
                    .template rshrink<T_NumBlocksType::dim()>();
            }

            constexpr auto count() const
            {
                return Vec<typename T_NumBlocksType::type, 3u>{::gridDim.z, ::gridDim.y, ::gridDim.x}
                    .template rshrink<T_NumBlocksType::dim()>();
            }
        };

        template<typename T_NumThreadsType>
        struct ThreadLayer
        {
            constexpr auto idx() const
            {
                return Vec<typename T_NumThreadsType::type, 3u>{::threadIdx.z, ::threadIdx.y, ::threadIdx.x}
                    .template rshrink<T_NumThreadsType::dim()>();
            }

            constexpr auto count() const
            {
                return Vec<typename T_NumThreadsType::type, 3u>{::threadIdx.z, ::threadIdx.y, ::threadIdx.x}
                    .template rshrink<T_NumThreadsType::dim()>();
            }

            constexpr auto count() const requires alpaka::concepts::CVector<T_NumThreadsType>
            {
                return T_NumThreadsType{};
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onAcc

#endif
