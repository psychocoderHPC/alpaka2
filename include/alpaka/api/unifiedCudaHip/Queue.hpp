/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    include "alpaka/api/cuda/IdxLayer.hpp"
#    include "alpaka/api/hip/IdxLayer.hpp"
#    include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#    include "alpaka/api/unifiedCudaHip/MemcpyKind.hpp"
#    include "alpaka/api/unifiedCudaHip/concepts.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onHost.hpp"
#    include "alpaka/onHost/FrameSpec.hpp"
#    include "alpaka/onHost/Handle.hpp"
#    include "alpaka/onHost/internal.hpp"

#    include <cstdint>
#    include <sstream>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        struct CallKernel;

        template<typename T_Device>
        struct Queue
        {
            using ApiInterface = typename T_Device::ApiInterface;

        public:
            Queue(concepts::DeviceHandle auto device, uint32_t const idx) : m_device(std::move(device)), m_idx(idx)
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(m_device)));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::streamCreateWithFlags(&m_UniformCudaHipQueue, ApiInterface::streamNonBlocking));
            }

            ~Queue()
            {
                onHost::internal::Wait::wait(*this);
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
            typename ApiInterface::Stream_t m_UniformCudaHipQueue;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("unifiedCudaHip::Queue id=") + std::to_string(m_idx);
            }

            friend struct onHost::internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipQueue;
            }

            friend struct onHost::internal::Enqueue;

            friend struct onHost::internal::Wait;

            void wait() const
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface,ApiInterface::streamSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetApi;
            friend struct onHost::internal::Memcpy;
            friend struct onHost::internal::Memset;
            friend struct CallKernel;
        };

        template<
            typename T_Api,
            typename T_Executor,
            typename T_IdxType,
            uint32_t T_dim,
            typename TKernelBundle,
            typename T_NumFrames,
            typename T_FrameSize>
        __global__ void gpuKernel(
            TKernelBundle const kernelBundle,
            T_NumFrames const numFrames,
            T_FrameSize const frameExtent)
        {
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::unifiedCudaHip::BlockLayer<T_IdxType, T_dim>{}),
                    DictEntry(layer::thread, onAcc::unifiedCudaHip::ThreadLayer<T_IdxType, T_dim>{}),
                    DictEntry(frame::count, numFrames),
                    DictEntry(frame::extent, frameExtent),
                    DictEntry(action::sync, onAcc::unifiedCudaHip::Sync{}),
                    DictEntry(object::api, T_Api{}),
                    DictEntry(object::exec, T_Executor{})},
            };
            kernelBundle(acc);
        }

        template<typename T_Api, typename T_Executor, typename T_IdxType, uint32_t T_dim, typename TKernelBundle>
        __global__ void gpuKernel(TKernelBundle const kernelBundle)
        {
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::unifiedCudaHip::BlockLayer<T_IdxType, T_dim>{}),
                    DictEntry(layer::thread, onAcc::unifiedCudaHip::ThreadLayer<T_IdxType, T_dim>{}),
                    DictEntry(action::sync, onAcc::unifiedCudaHip::Sync{}),
                    DictEntry(object::api, T_Api{}),
                    DictEntry(object::exec, T_Executor{})},
            };
            kernelBundle(acc);
        }

        template<typename TIdx, uint32_t T_dim>
        ALPAKA_FN_HOST auto convertVecToUniformCudaHipDim(Vec<TIdx, T_dim> const& vec) -> dim3
        {
            dim3 dim(1, 1, 1);
            if constexpr(T_dim >= 1u)
                dim.x = static_cast<unsigned>(vec[T_dim - 1u]);
            if constexpr(T_dim >= 2u)
                dim.y = static_cast<unsigned>(vec[T_dim - 2u]);
            if constexpr(T_dim >= 3u)
                dim.z = static_cast<unsigned>(vec[T_dim - 3u]);

            return dim;
        }

        struct CallKernel
        {
            template<
                typename T_Executor,
                typename T_Device,
                typename T_NumBlocks,
                typename T_NumThreads,
                typename T_KernelBundle,
                typename... T_Args>
            void operator()(
                T_Executor const executor,
                unifiedCudaHip::Queue<T_Device>& queue,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                T_KernelBundle kernelBundle,
                T_Args const&... args) const
            {
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto kernelName = gpuKernel<
                    ALPAKA_TYPEOF(onHost::getApi(queue)),
                    T_Executor,
                    typename T_NumBlocks::type,
                    T_NumBlocks::dim(),
                    T_KernelBundle,
                    T_Args...>;

                uint32_t blockDynSharedMemBytes
                    = onHost::getDynSharedMemBytes(exec::gpuCuda, threadBlocking, kernelBundle);

                kernelName<<<
                    convertVecToUniformCudaHipDim(threadBlocking.m_numBlocks),
                    convertVecToUniformCudaHipDim(threadBlocking.m_numThreads),
                    static_cast<std::size_t>(blockDynSharedMemBytes),
                    queue.getNativeHandle()>>>(kernelBundle, args...);
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::unifiedCudaHip::Queue<T_Device>>
    {
        decltype(auto) operator()(auto&& queue) const
        {
            return onHost::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

namespace alpaka::onHost
{
    namespace internal
    {

        template<
            typename T_Device,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelBundle>
        struct Enqueue::
            Kernel<unifiedCudaHip::Queue<T_Device>, T_Executor, ThreadSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                T_Executor const executor,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                T_KernelBundle kernelBundle) const
            {
                unifiedCudaHip::CallKernel{}(executor, queue, threadBlocking, std::move(kernelBundle));
            }
        };

        template<
            typename T_Device,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            typename T_NumFrames,
            typename T_FrameExtent,
            typename T_KernelBundle>
        struct Enqueue::
            Kernel<unifiedCudaHip::Queue<T_Device>, T_Executor, FrameSpec<T_NumFrames, T_FrameExtent>, T_KernelBundle>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                T_Executor const executor,
                FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
                T_KernelBundle kernelBundle) const
            {
                auto threadBlocking
                    = internal::adjustThreadSpec(*queue.m_device.get(), executor, frameSpec, kernelBundle);
                unifiedCudaHip::CallKernel{}(
                    executor,
                    queue,
                    threadBlocking,
                    std::move(kernelBundle),
                    frameSpec.m_numFrames,
                    frameSpec.m_frameExtent);
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                T_Dest dest,
                T_Source const source,
                T_Extents const& extents) const
            {
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto* destPtr = (void*) onHost::data(dest);
                auto* const srcPtr = (void*) onHost::data(source);

                auto copyKind = unifiedCudaHip::MemcpyKind<
                    ALPAKA_TYPEOF(alpaka::internal::getApi(dest)),
                    ALPAKA_TYPEOF(alpaka::internal::getApi(source))>::kind;

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpyAsync(
                            destPtr,
                            srcPtr,
                            extents.x() * sizeof(typename T_Dest::type),
                            copyKind,
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpy2DAsync(
                            destPtr,
                            dest.getPitches().y(),
                            srcPtr,
                            source.getPitches().y(),
                            extents.x() * sizeof(typename T_Dest::type),
                            extents.y(),
                            copyKind,
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 3u)
                {
                    // zero-init required per CUDA documentation
                    typename ApiInterface::Memcpy3DParms_t memCpy3DParms{};

                    memCpy3DParms.srcPtr = ApiInterface::makePitchedPtr(
                        srcPtr,
                        source.getPitches().y(),
                        source.getExtents().x(),
                        source.getExtents().y());
                    memCpy3DParms.dstPtr = ApiInterface::makePitchedPtr(
                        destPtr,
                        dest.getPitches().y(),
                        dest.getExtents().x(),
                        dest.getExtents().y());
                    memCpy3DParms.extent = ApiInterface::makeExtent(
                        extents.x() * sizeof(typename T_Dest::type),
                        extents.y(),
                        extents.z());
                    memCpy3DParms.kind = copyKind;

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpy3DAsync(&memCpy3DParms, internal::getNativeHandle(queue)));
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Extents>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                T_Dest dest,
                uint8_t byteValue,
                T_Extents const& extents) const
            {
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto* destPtr = (void*) onHost::data(dest);

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memsetAsync(
                            destPtr,
                            static_cast<int>(byteValue),
                            extents.x() * sizeof(typename T_Dest::type),
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memset2DAsync(
                            destPtr,
                            dest.getPitches().y(),
                            static_cast<int>(byteValue),
                            extents.x() * sizeof(typename T_Dest::type),
                            extents.y(),
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 3u)
                {
                    typename ApiInterface::PitchedPtr_t const pitchedPtrVal = ApiInterface::makePitchedPtr(
                        destPtr,
                        dest.getPitches().y(),
                        dest.getExtents().x(),
                        dest.getExtents().y());

                    typename ApiInterface::Extent_t const extentVal = ApiInterface::makeExtent(
                        extents.x() * sizeof(typename T_Dest::type),
                        extents.y(),
                        extents.z());

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memset3DAsync(
                            pitchedPtrVal,
                            static_cast<int>(byteValue),
                            extentVal,
                            internal::getNativeHandle(queue)));
                }
            }
        };
    } // namespace internal
} // namespace alpaka::onHost
#endif
