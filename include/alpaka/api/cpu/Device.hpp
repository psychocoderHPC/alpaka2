/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Device.hpp"
#include "alpaka/Queue.hpp"
#include "alpaka/Trait.hpp"
#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cpu/Queue.hpp"
#include "alpaka/core/Handle.hpp"
#include "alpaka/hostApi.hpp"
#include "alpaka/mem/View.hpp"
#include "alpaka/mem/Data.hpp"

#include <cstdint>
#include <memory>
#include <sstream>

namespace alpaka
{
    namespace cpu
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
        public:
            Device(concepts::PlatformHandle auto platform, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
            {
            }

            Device(Device const&) = delete;
            Device(Device&&) = delete;

            bool operator==(Device const& other) const
            {
                return m_idx == other.m_idx;
            }

            bool operator!=(Device const& other) const
            {
                return m_idx != other.m_idx;
            }

        private:
            void _()
            {
                static_assert(concepts::Device<Device>);
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            std::weak_ptr<cpu::Queue<Device>> queue;

            std::shared_ptr<Device> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("cpu::Device id=") + std::to_string(m_idx);
            }

            friend struct alpaka::internal::GetNativeHandle;

            [[nodiscard]] uint32_t getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct alpaka::internal::MakeQueue;

            Handle<cpu::Queue<Device>> makeQueue()
            {
                if(auto sharedPtr = queue.lock())
                {
                    return sharedPtr;
                }
                auto thisHandle = this->getSharedPtr();
                auto newQueue = std::make_shared<cpu::Queue<Device>>(std::move(thisHandle), 0u);
                queue = newQueue;
                return newQueue;
            }

            friend struct alpaka::internal::Alloc;
            friend struct alpaka::internal::GetApi;
        };
    } // namespace cpu

    namespace trait
    {
#if 1
        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<mapping::CpuBlockSerialThreadOne, cpu::Device<T_Platform>> : std::true_type
        {
        };
#endif
#if 1


        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<mapping::CpuBlockOmpThreadOmp, cpu::Device<T_Platform>> : std::true_type
        {
        };
#endif
#if 1
        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<mapping::CpuBlockOmpThreadOne, cpu::Device<T_Platform>> : std::true_type
        {
        };
#endif
    } // namespace trait

    namespace internal
    {
        template<typename T_Type, typename T_Platform, typename T_Extents>
        struct Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents) const
            {
                auto* ptr = new T_Type[extents.x()];
                auto deleter = [](T_Type* ptr) { delete[](ptr); };
                auto data = std::make_shared<alpaka::Data<Handle<std::decay_t<decltype(device)>>, T_Type, T_Extents>>(
                    device.getSharedPtr(),
                    ptr,
                    extents,
                    T_Extents{sizeof(T_Type)},
                    std::move(deleter));
                // return std::make_shared<alpaka::View<std::decay_t<decltype(data)>,T_Extents>>(data);
                return alpaka::View<std::decay_t<decltype(data)>, T_Extents>(data);
            }
        };

        template<typename T_Platform>
        struct GetApi::Op<cpu::Device<T_Platform>>
        {
            decltype(auto) operator()(auto&& device) const
            {
                return alpaka::getApi(device.m_platform);
            }
        };

        template<
            typename T_Platform,
            typename T_Mapping,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelBundle>
        requires mapping::traits::isSeqMapping_v<T_Mapping>
        struct AdjustThreadBlocking::
            Op<cpu::Device<T_Platform>, T_Mapping, DataBlocking<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            auto operator()(
                cpu::Device<T_Platform> const& queue,
                T_Mapping const& mapping,
                DataBlocking<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                return ThreadBlocking<T_NumBlocks, T_NumThreads>{dataBlocking.m_numBlocks, numThreads};
            }
        };
    } // namespace internal
} // namespace alpaka
