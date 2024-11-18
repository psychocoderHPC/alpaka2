/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost.hpp"

namespace alpaka::onHost
{
    template<typename T>
    struct Platform : std::shared_ptr<T>
    {
    private:
        using Parent = std::shared_ptr<T>;

        friend struct internal::MakePlatform;
        friend struct alpaka::internal::GetName;
        friend struct internal::GetDeviceCount;
        friend struct internal::GetNativeHandle;
        friend struct internal::MakeDevice;

    public:
        using element_type = typename Parent::element_type;

        Platform(std::shared_ptr<T>&& ptr) : std::shared_ptr<T>{std::forward<std::shared_ptr<T>>(ptr)}
        {
        }

        void _()
        {
            static_assert(concepts::PlatformHandle<Parent>);
            static_assert(concepts::Platform<Platform>);
        }

        std::string getName() const
        {
            return onHost::getName(static_cast<Parent>(*this));
        }

        uint32_t getDeviceCount() const
        {
            return onHost::getDeviceCount(static_cast<Parent>(*this));
        }

        auto makeDevice(uint32_t idx)
        {
            return onHost::makeDevice(static_cast<Parent>(*this), idx);
        }
    };

    template<typename T>
    ALPAKA_FN_HOST Platform(std::shared_ptr<T>) -> Platform<T>;
} // namespace alpaka::onHost
