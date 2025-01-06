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
    /** provides the API used during the execution of the current code path
     *
     * @attention if api::cpu os returned it can also mena that this method was called within the host controlling
     * workflow and not within a kernel running on a CPU device.
     */
    constexpr auto thisApi()
    {
#if ALPAKA_LANG_CUDA && (ALPAKA_COMP_CLANG_CUDA || ALPAKA_COMP_NVCC) && __CUDA_ARCH__
        return api::cuda;
#elif ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
        return api::hip;
#else
        return api::cpu;
#endif
    }

    namespace onHost
    {
        constexpr auto apis = std::make_tuple(api::cpu, api::cuda, api::hip);

        constexpr auto enabledApis = meta::filter([](auto api) { return isPlatformAvaiable(api); }, apis);
    } // namespace onHost
} // namespace alpaka
