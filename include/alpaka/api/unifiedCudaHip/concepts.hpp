/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/trait.hpp"

#include <concepts>

namespace alpaka
{
    namespace concepts
    {
        template<typename T>
        concept UnifiedCudaHipExecutor = alpaka::unifiedCudaHip::trait::IsUnifiedExecutor<T>::value;

        template<typename T>
        concept UnifiedCudaHipApi = alpaka::unifiedCudaHip::trait::IsUnifiedApi<T>::value;
    } // namespace concepts
} // namespace alpaka
