/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

namespace alpaka
{
    template<typename = decltype([]() -> void {})>
    struct UniqueId
    {
        static constexpr auto singleton = [] {};
        static constexpr decltype(singleton) const* address = std::addressof(singleton);
        static constexpr decltype(singleton) const* base = nullptr;
        static constexpr size_t id = std::distance(base, address);
    };
} // namespace alpaka
