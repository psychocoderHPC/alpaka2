/* Copyright 2026 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Dict.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::internal
{
    template<typename... T_Entries>
    requires(Dict<T_Entries...>::hasKey(object::api))
    struct GetApi::Op<Dict<T_Entries...>>
    {
        inline constexpr auto operator()(auto&& any) const
        {
            return any[object::api];
        }
    };

    template<typename... T_Entries>
    requires(Dict<T_Entries...>::hasKey(object::exec))
    struct GetExecutor::Op<Dict<T_Entries...>>
    {
        inline constexpr auto operator()(auto&& any) const
        {
            return any[object::exec];
        }
    };

    template<typename... T_Entries>
    requires(Dict<T_Entries...>::hasKey(object::deviceKind))
    struct GetDeviceType::Op<Dict<T_Entries...>>
    {
        inline constexpr auto operator()(auto&& any) const
        {
            return any[object::deviceKind];
        }
    };
} // namespace alpaka::internal
