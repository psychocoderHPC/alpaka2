/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA
#    include "alpaka/api/cuda/Api.hpp"
#    include "alpaka/api/cuda/ComputeApi.hpp"
#    include "alpaka/api/cuda/IdxLayer.hpp"
#    include "alpaka/api/cuda/MemcpyKind.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onHost.hpp"
#    include "alpaka/onHost/Handle.hpp"
#    include "alpaka/onHost/internal.hpp"

#    include <cstdint>
#    include <sstream>

namespace alpaka::onHost
{
    namespace cuda
    {
        struct CallKernel;

        template<typename T_Device>
        struct Queue
        {
            using TApi = ApiCudaRt;

        public:
            Queue(concepts::DeviceHandle auto device, uint32_t const idx) : m_device(std::move(device)), m_idx(idx)
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(onHost::getNativeHandle(m_device)));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::streamCreateWithFlags(&m_UniformCudaHipQueue, TApi::streamNonBlocking));
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
            typename TApi::Stream_t m_UniformCudaHipQueue;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("cuda::Queue id=") + std::to_string(m_idx);
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
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetApi;
            friend struct onHost::internal::Memcpy;
            friend struct CallKernel;
        };

        template<
            typename T_IdxType,
            uint32_t T_dim,
            typename TKernelBundle,
            typename T_NumFrames,
            typename T_FrameSize>
        __global__ void gpuKernel(
            TKernelBundle const kernelBundle,
            T_NumFrames const numFrames,
            T_FrameSize const framesSize)
        {
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::cuda::CudaBlock<T_IdxType, T_dim>{}),
                    DictEntry(layer::thread, onAcc::cuda::CudaThread<T_IdxType, T_dim>{}),
                    DictEntry(layer::shared, onAcc::cuda::StaticShared{}),
                    DictEntry(frame::count, numFrames),
                    DictEntry(frame::extent, framesSize),
                    DictEntry(action::sync, onAcc::cuda::Sync{}),
                    DictEntry(object::api, api::cuda),
                    DictEntry(object::exec, exec::gpuCuda)},
            };
            kernelBundle(acc);
        }

        template<typename T_IdxType, uint32_t T_dim, typename TKernelBundle>
        __global__ void gpuKernel(TKernelBundle const kernelBundle)
        {
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::cuda::CudaBlock<T_IdxType, T_dim>{}),
                    DictEntry(layer::shared, onAcc::cuda::StaticShared{}),
                    DictEntry(layer::thread, onAcc::cuda::CudaThread<T_IdxType, T_dim>{}),
                    DictEntry(action::sync, onAcc::cuda::Sync{}),
                    DictEntry(object::api, api::cuda),
                    DictEntry(object::exec, exec::gpuCuda)},
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
                typename T_Device,
                typename T_NumBlocks,
                typename T_NumThreads,
                typename T_KernelBundle,
                typename... T_Args>
            void operator()(
                cuda::Queue<T_Device>& queue,
                ThreadBlocking<T_NumBlocks, T_NumThreads> const& threadBlocking,
                T_KernelBundle kernelBundle,
                T_Args const&... args) const
            {
                using TApi = typename cuda::Queue<T_Device>::TApi;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto kernelName = gpuKernel<typename T_NumBlocks::type, T_NumBlocks::dim(), T_KernelBundle, T_Args...>;

                kernelName<<<
                    convertVecToUniformCudaHipDim(threadBlocking.m_numBlocks),
                    convertVecToUniformCudaHipDim(threadBlocking.m_numThreads),
                    static_cast<std::size_t>(0),
                    queue.getNativeHandle()>>>(kernelBundle, args...);
            }
        };
    } // namespace cuda
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cuda::Queue<T_Device>>
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

        template<typename T_Device, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
        struct Enqueue::
            Kernel<cuda::Queue<T_Device>, exec::GpuCuda, ThreadBlocking<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            void operator()(
                cuda::Queue<T_Device>& queue,
                exec::GpuCuda const,
                ThreadBlocking<T_NumBlocks, T_NumThreads> const& threadBlocking,
                T_KernelBundle kernelBundle) const
            {
                cuda::CallKernel{}(queue, threadBlocking, std::move(kernelBundle));
            }
        };

        template<typename T_Device, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
        struct Enqueue::
            Kernel<cuda::Queue<T_Device>, exec::GpuCuda, DataBlocking<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            void operator()(
                cuda::Queue<T_Device>& queue,
                exec::GpuCuda const executor,
                DataBlocking<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle kernelBundle) const
            {
                auto threadBlocking
                    = internal::adjustThreadBlocking(*queue.m_device.get(), executor, dataBlocking, kernelBundle);
                cuda::CallKernel{}(
                    queue,
                    threadBlocking,
                    std::move(kernelBundle),
                    dataBlocking.m_numBlocks,
                    dataBlocking.m_blockSize);
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cuda::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(cuda::Queue<T_Device>& queue, T_Dest dest, T_Source const source, T_Extents const& extents)
                const
            {
                using TApi = typename cuda::Queue<T_Device>::TApi;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto* destPtr = (void*) onHost::data(dest);
                auto* const srcPtr = (void*) onHost::data(source);

                auto copyKind = cuda::MemcpyKind<
                    ALPAKA_TYPE(alpaka::internal::getApi(dest)),
                    ALPAKA_TYPE(alpaka::internal::getApi(source))>::kind;

                constexpr auto dim = extents.dim();
                if constexpr(dim == 1u)
                {
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memcpyAsync(
                        destPtr,
                        srcPtr,
                        extents.x() * sizeof(typename T_Dest::type),
                        copyKind,
                        internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memcpy2DAsync(
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
                    typename TApi::Memcpy3DParms_t memCpy3DParms{};

                    memCpy3DParms.srcPtr = TApi::makePitchedPtr(
                        srcPtr,
                        source.getPitches().y(),
                        source.getExtents().x(),
                        source.getExtents().y());
                    memCpy3DParms.dstPtr = TApi::makePitchedPtr(
                        destPtr,
                        dest.getPitches().y(),
                        dest.getExtents().x(),
                        dest.getExtents().y());
                    memCpy3DParms.extent
                        = TApi::makeExtent(extents.x() * sizeof(typename T_Dest::type), extents.y(), extents.z());
                    memCpy3DParms.kind = copyKind;

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        TApi::memcpy3DAsync(&memCpy3DParms, internal::getNativeHandle(queue)));
                }
            }
        };
    } // namespace internal
} // namespace alpaka::onHost
#endif
