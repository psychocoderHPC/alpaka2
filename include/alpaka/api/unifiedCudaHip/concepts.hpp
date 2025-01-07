/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/executor.hpp"

#include <concepts>

namespace alpaka
{
    namespace concepts
    {
        template<typename T>
        concept UnifiedCudaHipExecutor
            = std::same_as<T, alpaka::exec::GpuCuda> || std::same_as<T, alpaka::exec::GpuHip>;
    } // namespace concepts
} // namespace alpaka
