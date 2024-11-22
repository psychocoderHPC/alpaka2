/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/trait.hpp"

#include <algorithm>

namespace alpaka
{
#if ALPAKA_LANG_CUDA && (ALPAKA_COMP_CLANG_CUDA || ALPAKA_COMP_NVCC) && __CUDA_ARCH__
    constexpr auto apiCtx = api::cuda;
#else
    constexpr auto apiCtx = api::cpu;
#endif
    namespace onHost
    {
        constexpr auto apis = std::make_tuple(api::cpu, api::cuda);

        constexpr auto enabledApis = meta::filter([](auto api) { return isPlatformAvaiable(api); }, apis);
    } // namespace onHost
} // namespace alpaka
