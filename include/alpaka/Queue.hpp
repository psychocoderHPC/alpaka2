/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Handle.hpp"
#include "alpaka/hostApi.hpp"

#include <memory>

namespace alpaka
{
    template<typename T_Queue>
    struct Queue : std::shared_ptr<T_Queue>
    {
    private:
        using Parent = std::shared_ptr<T_Queue>;

    public:
        friend struct alpaka::internal::Enqueue;
        friend struct alpaka::internal::Wait;
        using element_type = typename Parent::element_type;

        Queue(std::shared_ptr<T_Queue>&& ptr) : std::shared_ptr<T_Queue>{std::forward<std::shared_ptr<T_Queue>>(ptr)}
        {
        }

        void _()
        {
            static_assert(concepts::QueueHandle<Parent>);
            static_assert(concepts::Queue<Queue>);
        }

        std::string getName() const
        {
            return alpaka::getName(static_cast<Parent>(*this));
        }

        [[nodiscard]] uint32_t getNativeHandle() const
        {
            return alpaka::getNativeHandle(static_cast<Parent>(*this));
        }

        bool operator==(Queue const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Queue const& other) const
        {
            return this->get() != other.get();
        }

        void wait() const
        {
            return alpaka::wait(static_cast<Parent>(*this));
        }

        void enqueue(auto const mapping, auto const& blockCfg, auto&& f, auto&&... args)
        {
            return alpaka::enqueue(
                static_cast<Parent>(*this),
                std::move(mapping),
                blockCfg,
                std::move(KernelBundle{f, args...}));
        }

        template<typename TKernelFn, typename... TArgs>
        void enqueue(auto const mapping, auto const& blockCfg, KernelBundle<TKernelFn, TArgs...> kernelBundle)
        {
            return alpaka::enqueue(static_cast<Parent>(*this), std::move(mapping), blockCfg, std::move(kernelBundle));
        }

        void enqueue(auto const mapping, concepts::KernelBundleWithSize auto const& kernelBundleWithSize)
        {
            return alpaka::enqueue(
                static_cast<Parent>(*this),
                std::move(mapping),
                kernelBundleWithSize.m_numBlocks,
                kernelBundleWithSize.m_numThreads,
                kernelBundleWithSize.m_kernelBundle);
        }
    };
} // namespace alpaka
