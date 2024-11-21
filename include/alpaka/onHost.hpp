/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/concepts.hpp"

namespace alpaka::onHost
{
    inline concepts::PlatformHandle auto makePlatform(alpaka::concepts::Api auto&& api)
    {
        return internal::makePlatform(ALPAKA_FORWARD(api));
    }

    inline std::convertible_to<std::string> auto getStaticName(concepts::StaticNameHandle auto const& any)
    {
        return alpaka::internal::GetStaticName::Op<std::decay_t<decltype(*any.get())>>{}(*any.get());
    }

    inline std::convertible_to<std::string> auto getName(concepts::NameHandle auto const& any)
    {
        return alpaka::internal::GetName::Op<std::decay_t<decltype(*any.get())>>{}(*any.get());
    }

    inline uint32_t getDeviceCount(concepts::PlatformHandle auto const& platform)
    {
        return internal::GetDeviceCount::Op<std::decay_t<decltype(*platform.get())>>{}(*platform.get());
    }

    inline concepts::DeviceHandle auto makeDevice(concepts::PlatformHandle auto const& platform, uint32_t idx)
    {
        return internal::MakeDevice::Op<std::decay_t<decltype(*platform.get())>>{}(*platform.get(), idx);
    }

    inline auto getNativeHandle(auto const& any)
    {
        return internal::getNativeHandle(*any.get());
    }

    inline auto makeQueue(concepts::DeviceHandle auto const& device)
    {
        return internal::MakeQueue::Op<std::decay_t<decltype(*device.get())>>{}(*device.get());
    }

    inline auto wait(alpaka::concepts::HasGet auto const& any)
    {
        return internal::Wait::wait(*any.get());
    }

    template<typename TKernelFn, typename... TArgs>
    inline void enqueue(
        concepts::QueueHandle auto const& queue,
        auto const executor,
        auto const& blockCfg,
        KernelBundle<TKernelFn, TArgs...> kernelBundle)
    {
        internal::enqueue(*queue.get(), executor, blockCfg, std::move(kernelBundle));
    }

    inline void enqueue(concepts::QueueHandle auto const& queue, auto task)
    {
        return internal::Enqueue::Task<std::decay_t<decltype(*queue.get())>, std::decay_t<decltype(task)>>{}(
            *queue.get(),
            std::move(task));
    }

    inline decltype(auto) data(auto&& any)
    {
        return internal::Data::data(ALPAKA_FORWARD(any));
    }

    inline decltype(auto) data(alpaka::concepts::HasGet auto&& any)
    {
        return internal::Data::data(*any.get());
    }

    inline decltype(auto) getApi(auto&& any)
    {
        return alpaka::internal::getApi(ALPAKA_FORWARD(any));
    }

    inline decltype(auto) getApi(alpaka::concepts::HasGet auto&& any)
    {
        return alpaka::internal::getApi(*any.get());
    }

    template<typename T_Type>
    inline auto alloc(auto const& any, auto const& extents)
    {
        return internal::Alloc::Op<T_Type, std::decay_t<decltype(*any.get())>, ALPAKA_TYPE(extents)>{}(
            *any.get(),
            extents);
    }

    inline auto memcpy(concepts::QueueHandle auto& queue, auto& dest, auto const& source, auto const& extents)
    {
        return internal::Memcpy::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            std::decay_t<decltype(source)>,
            std::decay_t<decltype(extents)>>{}(*queue.get(), dest, source, extents);
    }

    inline auto memcpy(concepts::QueueHandle auto& queue, auto& dest, auto const& source)
    {
        return memcpy(queue, dest, source, dest.getExtents());
    }
} // namespace alpaka::onHost
