/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/HostApiTraits.hpp"

#include <concepts>
#include <string>

namespace alpaka
{
    namespace concepts
    {
        template<typename T>
        concept HasStaticName = requires(T t) {
            {
                internal::GetStaticName::Op<std::decay_t<T>>{}(t)
            } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasName = requires(T t) {
            {
                internal::GetName::Op<T>{}(t)
            } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept Platform = requires(T platform) {
            {
                internal::GetName::Op<T>{}(platform)
            };
            {
                internal::GetDeviceCount::Op<T>{}(platform)
            } -> std::same_as<uint32_t>;
            // api.makeDevice(uint32_t{0});
        };

        template<typename T>
        concept Api = requires(T t) { requires HasStaticName<T>; };

        template<typename T>
        concept Device = requires(T device) {
            {
                internal::GetName::Op<T>{}(device)
            } -> std::convertible_to<std::string>;

            {
                internal::MakeQueue::Op<T>{}(device)
            };
            {
                internal::GetNativeHandle::Op<T>{}(device)
            };
        };

        template<typename T>
        concept Queue = requires(T device) {
            {
                internal::GetName::Op<T>{}(device)
            } -> std::convertible_to<std::string>;
            {
                internal::GetNativeHandle::Op<T>{}(device)
            };
        };
    } // namespace concepts

    namespace concepts
    {
        template<typename T>
        concept QueueHandle = requires(T t) {
            typename T::element_type;
            requires Queue<typename T::element_type>;
        };


        template<typename T>
        concept DeviceHandle = requires(T t) {
            typename T::element_type;
            requires Device<typename T::element_type>;
        };

        template<typename T>
        concept PlatformHandle = requires(T t) {
            typename T::element_type;
            requires Platform<typename T::element_type>;
        };

        template<typename T>
        concept NameHandle = requires(T t) {
            typename T::element_type;
            requires HasName<typename T::element_type>;
        };

        template<typename T>
        concept StaticNameHandle = requires(T t) {
            typename T::element_type;
            requires HasStaticName<typename T::element_type>;
        };

        template<typename T>
        concept HasGet = requires(T t) { t.get(); };
    } // namespace concepts
} // namespace alpaka
