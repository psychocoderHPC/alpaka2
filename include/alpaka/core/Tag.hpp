/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/UniqueId.hpp"

#include <type_traits>

namespace alpaka
{
    template<typename T_Id = decltype([]() -> void {})>
    struct Tag
    {
    };

#define ALPAKA_TAG(name)                                                                                              \
    constexpr Tag<std::integral_constant<size_t, __COUNTER__>> name                                                   \
    {                                                                                                                 \
    }

    namespace trait
    {
        template<typename T_Object, typename T_Sfinae = void>
        struct IsTag : std::false_type
        {
        };

        template<typename T_Id>
        struct IsTag<Tag<T_Id>> : std::true_type
        {
        };

        template<typename T_Id>
        constexpr bool isTag_v = IsTag<T_Id>::value;

    } // namespace trait

} // namespace alpaka
