/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/trait.hpp"

#include <algorithm>

namespace alpaka::onHost
{
    constexpr auto apis = std::make_tuple(api::cpu, api::cuda);

    constexpr auto enabledApis = meta::filter([](auto api) { return isPlatformAvaiable(api); }, apis);
} // namespace alpaka::onHost
