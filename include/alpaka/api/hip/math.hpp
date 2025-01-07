/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/hip/Api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"

namespace alpaka::trait
{
    template<>
    struct GetMathImpl::Op<alpaka::api::Hip>
    {
        constexpr decltype(auto) operator()(alpaka::api::Hip const) const
        {
            return alpaka::math::internal::cudaHipMath;
        }
    };
} // namespace alpaka::trait
