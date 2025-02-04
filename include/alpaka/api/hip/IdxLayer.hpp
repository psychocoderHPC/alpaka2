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
        template<typename T_NumBlocksType>
        struct BlockLayer
        {
            constexpr auto idx() const
            {
                return Vec<typename T_NumBlocksType::type, 3u>{hipBlockIdx_z, hipBlockIdx_y, hipBlockIdx_x}
                    .template rshrink<T_NumBlocksType::dim()>();
            }

            constexpr auto count() const
            {
                return Vec<typename T_NumBlocksType::type, 3u>{hipGridDim_z, hipGridDim_y, hipGridDim_x}
                    .template rshrink<T_NumBlocksType::dim()>();
            }
        };

        template<typename T_NumThreadsType>
        struct ThreadLayer
        {
            constexpr auto idx() const
            {
                return Vec<typename T_NumThreadsType::type, 3u>{hipThreadIdx_z, hipThreadIdx_y, hipThreadIdx_x}
                    .template rshrink<T_NumThreadsType::dim()>();
            }

            constexpr auto count() const
            {
                return Vec<typename T_NumThreadsType::type, 3u>{hipBlockDim_z, hipBlockDim_y, hipBlockDim_x}
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
