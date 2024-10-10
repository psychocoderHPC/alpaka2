/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA

#    include "alpaka/Trait.hpp"
#    include "alpaka/api/cuda/Api.hpp"
#    include "alpaka/api/cuda/Device.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/Handle.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/hostApi.hpp"

#    include <memory>
#    include <mutex>
#    include <sstream>
#    include <vector>

namespace alpaka
{
    namespace cuda
    {
        struct Platform : std::enable_shared_from_this<Platform>
        {
            using TApi = ApiCudaRt;

        public:
            Platform() = default;

            Platform(Platform const&) = delete;
            Platform(Platform&&) = delete;

        private:
            void _()
            {
                static_assert(concepts::Platform<Platform>);
            }

            std::vector<std::weak_ptr<cuda::Device<Platform>>> devices;
            std::mutex deviceGuard;

            std::shared_ptr<Platform> getSharedPtr()
            {
                return shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return "cuda::Platform";
            }

            friend struct alpaka::internal::GetDeviceCount;

            uint32_t getDeviceCount()
            {
                int numDevices{0};
                typename TApi::Error_t error = TApi::getDeviceCount(&numDevices);
                if(error != TApi::success)
                    numDevices = 0;

                if(devices.size() < numDevices)
                {
                    std::lock_guard<std::mutex> lk{deviceGuard};
                    devices.resize(numDevices);
                }
                return static_cast<uint32_t>(numDevices);
            }

            friend struct alpaka::internal::MakeDevice;

            Handle<cuda::Device<Platform>> makeDevice(uint32_t const& idx)
            {
                uint32_t const numDevices = getDeviceCount();
                if(idx >= numDevices)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for CPU device with index " << idx
                          << " because there are only " << numDevices << " devices!";
                    throw std::runtime_error(ssErr.str());
                }
                std::lock_guard<std::mutex> lk{deviceGuard};

                if(auto sharedPtr = devices[idx].lock())
                {
                    return sharedPtr;
                }
                auto thisHandle = getSharedPtr();
                auto newDevice = std::make_shared<cuda::Device<Platform>>(std::move(thisHandle), idx);
                devices[idx] = newDevice;
                return newDevice;
            }
        };
    } // namespace cuda

    namespace internal
    {
        template<>
        struct MakePlatform::Op<api::Cuda>
        {
            auto operator()(api::Cuda const&) const
            {
                return alpaka::make_sharedSingleton<cuda::Platform>();
            }
        };

        template<>
        struct GetApi::Op<cuda::Platform>
        {
            decltype(auto) operator()(auto&& platform) const
            {
                return api::Cuda{};
            }
        };
    } // namespace internal
} // namespace alpaka
#endif
