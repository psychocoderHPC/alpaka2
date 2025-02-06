/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/vecConcepts.hpp"

#include <concepts>
#include <cstdint>

namespace alpaka
{
    namespace trait
    {
        template<typename T>
        struct GetDim
        {
            static constexpr uint32_t value = T::dim();
        };

        template<std::integral T>
        struct GetDim<T>
        {
            static constexpr uint32_t value = 1u;
        };

        template<typename T>
        constexpr uint32_t getDim_v = GetDim<T>::value;
    } // namespace trait

    template<typename T>
    consteval uint32_t getDim(T const& any)
    {
        return trait::getDim_v<T>;
    }

    template<typename T_From, typename T_To>
    constexpr bool isLosslessConvertible_v = concepts::IsLosslessConvertible<T_From, T_To>;

    template<typename T_From, typename T_To>
    constexpr bool isConvertible_v = concepts::IsConvertible<T_From, T_To>;

} // namespace alpaka
