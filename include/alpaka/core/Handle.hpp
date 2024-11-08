/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>
#include <memory>
#include <type_traits>

namespace alpaka
{
    template<typename T_Object, typename... T_Args>
    inline auto make_sharedSingleton(T_Args&&... args)
    {
        static std::mutex mutex;
        static std::weak_ptr<T_Object> platform;

        std::lock_guard<std::mutex> lk(mutex);
        if(auto sharedPtr = platform.lock())
        {
            return sharedPtr;
        }
        auto new_platform = std::make_shared<T_Object>(std::forward<T_Args>(args)...);
        platform = new_platform;
        return new_platform;
    }

    template<typename T>
    using Handle = std::shared_ptr<T>;
} // namespace alpaka
