/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Tags.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/filter.hpp"

#include <algorithm>

namespace alpaka
{
    constexpr auto getExecutors(auto const& api)
    {
        using PlatformType = decltype(makePlatform(api));
        using DeviceType = decltype(makeDevice(std::declval<PlatformType>(), 0));
        using autoDeviceMappings = decltype(supportedMappings(std::declval<DeviceType>()));
        return autoDeviceMappings{};
    }

    constexpr auto createApiAndExecTuple(auto const& api, auto const& executorTuple)
    {
        return std::apply(
            [api](auto... executor) {
                return std::make_tuple(Dict{DictEntry{object::api, api}, DictEntry{object::exec, executor}}...);
            },
            executorTuple);
    }

    constexpr auto allExecutorsAndApis(auto const& usedApis)
    {
        return std::apply(
            [](auto... api) { return std::tuple_cat(createApiAndExecTuple(api, getExecutors(api))...); },
            usedApis);
    }
} // namespace alpaka
