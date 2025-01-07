/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"

namespace alpaka::trait
{
    template<>
    struct GetMathImpl::Op<alpaka::api::Cuda>
    {
        constexpr decltype(auto) operator()(alpaka::api::Cuda const) const
        {
            return alpaka::math::internal::cudaHipMath;
        }
    };
} // namespace alpaka::trait
