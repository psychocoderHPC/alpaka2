/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/HostApiTraits.hpp"
#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cpu/exec/OmpBlocks.hpp"
#include "alpaka/api/cpu/exec/OmpThreads.hpp"
#include "alpaka/api/cpu/exec/Serial.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/Handle.hpp"
#include "alpaka/hostApi.hpp"
#include "alpaka/meta/NdLoop.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <sstream>

namespace alpaka
{
    namespace cpu
    {
        template<typename T_Device>
        struct Queue
        {
        public:
            Queue(concepts::DeviceHandle auto device, uint32_t const idx) : m_device(std::move(device)), m_idx(idx)
            {
            }

            ~Queue()
            {
                alpaka::internal::Wait::wait(*this);
            }

            Queue(Queue const&) = delete;
            Queue(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(concepts::Queue<Queue>);
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            core::CallbackThread m_workerThread;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("cpu::Queue id=") + std::to_string(m_idx);
            }

            friend struct alpaka::internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return 0;
            }

            friend struct alpaka::internal::Enqueue;

            template<typename T_NumBlocks, typename T_NumThreads>
            void enqueue(
                auto const executor,
                ThreadBlocking<T_NumBlocks, T_NumThreads> const& threadBlocking,
                auto kernelBundle)
            {
                m_workerThread.submit(
                    [=, kernel = std::move(kernelBundle)]()
                    {
                        Acc acc = makeAcc(executor, threadBlocking);
                        acc(std::move(kernel));
                    });
            }

            template<typename T_Mapping, typename T_NumBlocks, typename T_BlockSize>
            void enqueue(
                T_Mapping const executor,
                DataBlocking<T_NumBlocks, T_BlockSize> dataBlocking,
                auto kernelBundle)
            {
                auto threadBlocking
                    = internal::adjustThreadBlocking(*m_device.get(), executor, dataBlocking, kernelBundle);
                m_workerThread.submit(
                    [=, kernel = std::move(kernelBundle)]()
                    {
                        auto moreLayer = Dict{
                            DictEntry(frame::count, dataBlocking.m_numBlocks),
                            DictEntry(frame::extent, dataBlocking.m_blockSize),
                            DictEntry(object::api, api::cpu),
                            DictEntry(object::exec, executor)};
                        Acc acc = makeAcc(executor, threadBlocking);
                        acc(std::move(kernel), moreLayer);
                    });
            }

            void enqueue(auto task)
            {
                m_workerThread.submit([task]() { task(); });
            }

            friend struct alpaka::internal::Wait;
            friend struct alpaka::internal::Memcpy;
            friend struct alpaka::internal::GetApi;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Device>
        struct Wait::Op<cpu::Queue<T_Device>>
        {
            void operator()(cpu::Queue<T_Device>& queue) const
            {
                std::promise<void> p;
                auto f = p.get_future();
                internal::enqueue(queue, [&p]() { p.set_value(); });

                f.wait();
            }
        };
#if 0
        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Dest dest, T_Source const source, T_Extents const& extents)
                const
            {
                static_assert(std::is_same_v<ALPAKA_TYPE(dest), ALPAKA_TYPE(source)>);
                constexpr auto dim = dest.dim();
                internal::Enqueue::enqueue(
                    queue,
                    [extents, l_dest = std::move(dest), l_source = std::move(source)]()
                    {
                        if constexpr(dim == 1u)
                        {
                            std::memcpy(
                                alpaka::data(l_dest),
                                alpaka::data(l_source),
                                extents.x() * sizeof(typename T_Dest::type));
                        }
                        else
                        {
                            auto const dstExtentWithoutRow = extents.template rshrink<dim - 1u>(1u);
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                auto const destPitchBytesWithoutRow = l_dest.getPitches().template rshrink<dim - 1u>(1u);
                                auto* destPtr = alpaka::data(l_dest);
                                auto const sourcePitchBytesWithoutRow
                                    = l_source.getPitches().template rshrink<dim - 1u>(1u);
                                auto* sourcePtr = alpaka::data(l_source);

                                std::cout << "row" << dstExtentWithoutRow << std::endl;
                                meta::ndLoopIncIdx(
                                    dstExtentWithoutRow,
                                    [&](auto const& idx)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutRow).sum(),
                                            reinterpret_cast<std::uint8_t*>(sourcePtr)
                                                + (idx * sourcePitchBytesWithoutRow).sum(),
                                            static_cast<size_t>(extents.back()) * sizeof(typename T_Dest::type));
                                    });
                            }
                        }
                    });
            }
        };
#endif

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Dest dest, T_Source const source, T_Extents const& extents)
                const
            {
                static_assert(std::is_same_v<ALPAKA_TYPE(dest), ALPAKA_TYPE(source)>);
                constexpr auto dim = dest.dim();
                if constexpr(dim == 1u)
                {
                    internal::enqueue(
                        queue,
                        [extents, l_dest = std::move(dest), l_source = std::move(source)]() {
                            std::memcpy(
                                alpaka::data(l_dest),
                                alpaka::data(l_source),
                                extents.x() * sizeof(typename T_Dest::type));
                        });
                }

                else
                {
                    internal::enqueue(
                        queue,
                        [extents, l_dest = std::move(dest), l_source = std::move(source)]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                auto const destPitchBytesWithoutColumn = l_dest.getPitches().eraseBack();
                                auto* destPtr = alpaka::data(l_dest);
                                auto const sourcePitchBytesWithoutColumn = l_source.getPitches().eraseBack();
                                auto* sourcePtr = alpaka::data(l_source);

                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            reinterpret_cast<std::uint8_t*>(sourcePtr)
                                                + (idx * sourcePitchBytesWithoutColumn).sum(),
                                            static_cast<size_t>(extents.back()) * sizeof(typename T_Dest::type));
                                    });
                            }
                        });
                }
            }
        };

        template<typename T_Device>
        struct GetApi::Op<cpu::Queue<T_Device>>
        {
            decltype(auto) operator()(auto&& queue) const
            {
                return alpaka::getApi(queue.m_device);
            }
        };
#if 0
        template<typename T_Type, typename T_Device, typename T_Extents>
        struct Alloc::Op<T_Type, cpu::Queue<T_Device>, T_Extents>
        {
            auto operator()(cpu::Queue<T_Device>& queue, T_Extents const& extents) const
            {
                return alpaka::alloc<T_Type>(queue.m_device,extents);
            }
        };
#endif
    } // namespace internal

#if 0
    namespace internal
    {
        template<
            concepts::Queue T_Device,
            typename T_Mapping,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelBundle>
        struct Enqueue::Kernel<Handle<cpu::Queue<T_Device>>, T_Mapping, T_NumBlocks, T_NumThreads, T_KernelBundle>
        {
            void operator()(
                Handle<cpu::Queue<T_Device>> const queue,
                T_Mapping const executor,
                T_NumBlocks const numBlocks,
                T_NumThreads const numThreads,
                T_KernelBundle kernelBundle) const
            {
                std::cout << "enqueu overload" << std::endl;
                return queue->enqueue(executor, numBlocks, numThreads, std::move(kernelBundle));
            }
        };
    } // namespace internal
#endif
} // namespace alpaka
