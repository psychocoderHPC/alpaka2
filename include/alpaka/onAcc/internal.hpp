/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onAcc
{
    namespace internalCompute
    {
        struct SyncBlockThreads
        {
            template<typename T_Acc>
            struct Op
            {
                constexpr auto operator()(T_Acc const& acc) const
                {
                    acc[action::sync]();
                }
            };
        };

        constexpr void syncBlockThreads(auto const& acc)
        {
            SyncBlockThreads::Op<std::decay_t<decltype(acc)>>{}(acc);
        }

        struct DeclareSharedVar
        {
            template<typename T, typename T_Acc>
            struct Op
            {
                constexpr decltype(auto) operator()(T_Acc const& acc) const
                {
                    return acc[layer::shared].template allocVar<T>();
                }
            };
        };

        template<typename T>
        constexpr decltype(auto) declareSharedVar(auto const& acc)
        {
            return DeclareSharedVar::Op<T, std::decay_t<decltype(acc)>>{}(acc);
        }

    } // namespace internalCompute
} // namespace alpaka::onAcc
