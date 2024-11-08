/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA
#    include "alpaka/Device.hpp"
#    include "alpaka/Queue.hpp"
#    include "alpaka/api/cuda/Api.hpp"
#    include "alpaka/api/cuda/Queue.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/Handle.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/hostApi.hpp"
#    include "alpaka/mem/Data.hpp"

#    include <cstdint>
#    include <memory>
#    include <mutex>
#    include <sstream>
#    include <vector>

namespace alpaka
{
    namespace cuda
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
            using TApi = ApiCudaRt;

        public:
            Device(concepts::PlatformHandle auto platform, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(idx));
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
            std::vector<std::weak_ptr<cuda::Queue<Device>>> queues;
            std::mutex queuesGuard;

            std::shared_ptr<Device> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("cuda::Device id=") + std::to_string(m_idx);
            }

            friend struct alpaka::internal::GetNativeHandle;

            [[nodiscard]] uint32_t getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct alpaka::internal::MakeQueue;

            Handle<cuda::Queue<Device>> makeQueue()
            {
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{queuesGuard};
                auto newQueue = std::make_shared<cuda::Queue<Device>>(std::move(thisHandle), queues.size());

                queues.emplace_back(newQueue);
                return newQueue;
            }

            friend struct alpaka::internal::Alloc;
            friend struct alpaka::internal::GetApi;
        };
    } // namespace cuda

    namespace internal
    {
        template<typename T_Type, typename T_Platform, typename T_Extents>
        struct Alloc::Op<T_Type, cuda::Device<T_Platform>, T_Extents>
        {
            auto operator()(cuda::Device<T_Platform>& device, T_Extents const& extents) const
            {
                using TApi = typename cuda::Device<T_Platform>::TApi;
                T_Type* ptr = nullptr;
                auto pitches = T_Extents::all(sizeof(T_Type));

                using Idx = typename T_Extents::type;

                constexpr auto dim = extents.dim();
                if constexpr(dim == 1u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        TApi::malloc((void**) &ptr, static_cast<std::size_t>(extents.x()) * sizeof(T_Type)));
                }
                else if constexpr(dim == 2u)
                {
                    size_t rowPitchInBytes = 0u;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::mallocPitch(
                        (void**) &ptr,
                        &rowPitchInBytes,
                        static_cast<std::size_t>(extents.x()) * sizeof(T_Type),
                        static_cast<std::size_t>(extents.y())));

                    pitches = mem::calculatePitches<T_Type>(extents, static_cast<Idx>(rowPitchInBytes));
                }
                else if constexpr(dim == 3u)
                {
                    typename TApi::Extent_t const extentVal = TApi::makeExtent(
                        static_cast<std::size_t>(extents.x()) * sizeof(T_Type),
                        static_cast<std::size_t>(extents.y()),
                        static_cast<std::size_t>(extents.z()));
                    typename TApi::PitchedPtr_t pitchedPtrVal;
                    pitchedPtrVal.ptr = nullptr;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::malloc3D(&pitchedPtrVal, extentVal));

                    ptr = reinterpret_cast<T_Type*>(pitchedPtrVal.ptr);
                    Idx rowPitchInBytes = pitchedPtrVal.pitch;
                    pitches = mem::calculatePitches<T_Type>(extents, static_cast<Idx>(pitchedPtrVal.pitch));
                }

                auto deleter = [](T_Type* ptr) { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::free(ptr)); };
                auto data = std::make_shared<alpaka::Data<Handle<std::decay_t<decltype(device)>>, T_Type, T_Extents>>(
                    device.getSharedPtr(),
                    ptr,
                    extents,
                    pitches,
                    deleter);
                return alpaka::View<std::decay_t<decltype(data)>, T_Extents>(data);
            }
        };

        template<typename T_Platform>
        struct GetApi::Op<cuda::Device<T_Platform>>
        {
            decltype(auto) operator()(auto&& device) const
            {
                return alpaka::getApi(device.m_platform);
            }
        };
    } // namespace internal

    namespace trait
    {
        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<mapping::Cuda, cuda::Device<T_Platform>> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

#endif
