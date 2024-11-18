/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Blocking.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/Handle.hpp"

namespace alpaka
{
    namespace internal
    {
        struct GetStaticName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const&) const
                {
                    return T_Any::getName();
                }
            };
        };

        struct GetName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getName();
                }
            };
        };

        struct GetApi
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.getApi();
                }
            };
        };

        inline auto getApi(auto&& any)
        {
            return GetApi::Op<std::decay_t<decltype(any)>>{}(any);
        }
    } // namespace internal
} // namespace alpaka
