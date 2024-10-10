/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/HostApiTraits.hpp"
#include "alpaka/acc/makeAcc.hpp"
#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/Handle.hpp"
#include "alpaka/hostApi.hpp"

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
                auto const mapping,
                ThreadBlocking<T_NumBlocks, T_NumThreads> const& threadBlocking,
                auto kernelBundle)
            {
                m_workerThread.submit(
                    [=, kernel = std::move(kernelBundle)]()
                    {
                        Acc acc = makeAcc(mapping, threadBlocking);
                        acc(std::move(kernel));
                    });
            }

            template<typename T_Mapping, typename T_NumBlocks, typename T_BlockSize>
            void enqueue(
                T_Mapping const mapping,
                DataBlocking<T_NumBlocks, T_BlockSize> dataBlocking,
                auto kernelBundle)
            {
                auto threadBlocking
                    = internal::adjustThreadBlocking(*m_device.get(), mapping, dataBlocking, kernelBundle);
                m_workerThread.submit(
                    [=, kernel = std::move(kernelBundle)]()
                    {
                        auto moreLayer = Dict{
                            DictEntry(frame::block, dataBlocking.m_numBlocks),
                            DictEntry(frame::thread, dataBlocking.m_blockSize)};
                        Acc acc = makeAcc(mapping, threadBlocking, moreLayer);
                        acc(std::move(kernel));
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
                Enqueue::enqueue(queue, [&p]() { p.set_value(); });

                f.wait();
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Dest dest, T_Source const source) const
            {
                static_assert(std::is_same_v<ALPAKA_TYPE(dest), ALPAKA_TYPE(source)>);
                constexpr auto dim = dest.dim();
                internal::Enqueue::enqueue(
                    queue,
                    [l_dest = std::move(dest), l_source = std::move(source)]()
                    {
                        if constexpr(dim == 1u)
                        {
                            std::memcpy(
                                alpaka::data(l_dest),
                                alpaka::data(l_source),
                                l_dest.getExtent().x() * sizeof(typename T_Dest::type));
                        }
                        else
                        {
                            static_assert(dim != 1u);
                        }
                    });
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
                T_Mapping const mapping,
                T_NumBlocks const numBlocks,
                T_NumThreads const numThreads,
                T_KernelBundle kernelBundle) const
            {
                std::cout << "enqueu overload" << std::endl;
                return queue->enqueue(mapping, numBlocks, numThreads, std::move(kernelBundle));
            }
        };
    } // namespace internal
#endif
} // namespace alpaka
