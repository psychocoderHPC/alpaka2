/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA
#    include "alpaka/acc/Cuda.hpp"
#    include "alpaka/api/cuda/Api.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/CallbackThread.hpp"
#    include "alpaka/core/DemangleTypeNames.hpp"
#    include "alpaka/core/Handle.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/hostApi.hpp"

#    include <cstdint>
#    include <sstream>

namespace alpaka
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
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(alpaka::getNativeHandle(m_device)));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::streamCreateWithFlags(&m_UniformCudaHipQueue, TApi::streamNonBlocking));
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
            typename TApi::Stream_t m_UniformCudaHipQueue;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("cuda::Queue id=") + std::to_string(m_idx);
            }

            friend struct alpaka::internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipQueue;
            }

            friend struct alpaka::internal::Enqueue;

            friend struct alpaka::internal::Wait;

            void wait() const
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetApi;
            friend struct alpaka::internal::Memcpy;
            friend struct CallKernel;
        };

        template<typename T_Dest, typename T_Source>
        struct Memcpy;

        template<>
        struct Memcpy<api::Cpu, api::Cuda>
        {
            static constexpr auto kind = ApiCudaRt::memcpyDeviceToHost;
        };

        template<>
        struct Memcpy<api::Cuda, api::Cuda>
        {
            static constexpr auto kind = ApiCudaRt::memcpyDeviceToDevice;
        };

        template<>
        struct Memcpy<api::Cuda, api::Cpu>
        {
            static constexpr auto kind = ApiCudaRt::memcpyHostToDevice;
        };

        template<
            typename T_IdxType,
            uint32_t T_dim,
            typename TKernelBundle,
            typename T_NumFrames,
            typename T_FrameSize>
        __global__ void gpuKernel(
            TKernelBundle const kernelBundle,
            T_NumFrames const& numFrames,
            T_FrameSize const& framesSize)
        {
            auto acc = Acc{
                std::make_tuple(layer::block, layer::thread, internal_layer::threadCommand),
                Dict{
                    DictEntry(layer::block, CudaBlock<T_IdxType, T_dim>{}),
                    DictEntry(layer::thread, CudaThread<T_IdxType, T_dim>{}),
                    DictEntry(internal_layer::threadCommand, ThreadCommand{}),
                    DictEntry(frame::block, numFrames),
                    DictEntry(frame::thread, framesSize)},
            };
            acc(std::move(kernelBundle));
        }

        template<typename T_IdxType, uint32_t T_dim, typename TKernelBundle>
        __global__ void gpuKernel(TKernelBundle const kernelBundle)
        {
            auto acc = Acc{
                std::make_tuple(layer::block, layer::thread, internal_layer::threadCommand),
                Dict{
                    DictEntry(layer::block, CudaBlock<T_IdxType, T_dim>{}),
                    DictEntry(layer::thread, CudaThread<T_IdxType, T_dim>{}),
                    DictEntry(internal_layer::threadCommand, ThreadCommand{})},
            };
            acc(std::move(kernelBundle));
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
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(alpaka::getNativeHandle(queue.m_device)));

                auto kernelName = gpuKernel<typename T_NumBlocks::type, T_NumBlocks::dim(), T_KernelBundle, T_Args...>;

                kernelName<<<
                    convertVecToUniformCudaHipDim(threadBlocking.m_numBlocks),
                    convertVecToUniformCudaHipDim(threadBlocking.m_numThreads),
                    static_cast<std::size_t>(0),
                    queue.getNativeHandle()>>>(kernelBundle, args...);

                auto const msg
                    = std::string{"execution of kernel '" + core::demangledName<T_KernelBundle>() + "' failed with"};
                ::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(msg.c_str(), __FILE__, __LINE__);
            }
        };
    } // namespace cuda

    namespace internal
    {

        template<typename T_Device, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
        struct Enqueue::
            Kernel<cuda::Queue<T_Device>, mapping::Cuda, ThreadBlocking<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            void operator()(
                cuda::Queue<T_Device>& queue,
                mapping::Cuda const,
                ThreadBlocking<T_NumBlocks, T_NumThreads> const& threadBlocking,
                T_KernelBundle kernelBundle) const
            {
                cuda::CallKernel{}(queue, threadBlocking, std::move(kernelBundle));
            }
        };

        template<typename T_Device, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
        struct Enqueue::
            Kernel<cuda::Queue<T_Device>, mapping::Cuda, DataBlocking<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            void operator()(
                cuda::Queue<T_Device>& queue,
                mapping::Cuda const mapping,
                DataBlocking<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle kernelBundle) const
            {
                std::cout << "enqueu cuda overload data blocking" << std::endl;
                auto threadBlocking
                    = internal::adjustThreadBlocking(*queue.m_device.get(), mapping, dataBlocking, kernelBundle);
                cuda::CallKernel{}(
                    queue,
                    threadBlocking,
                    std::move(kernelBundle),
                    dataBlocking.m_numBlocks,
                    dataBlocking.m_blockSize);
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source>
        struct Memcpy::Op<cuda::Queue<T_Device>, T_Dest, T_Source>
        {
            void operator()(cuda::Queue<T_Device>& queue, T_Dest dest, T_Source const source) const
            {
                internal::Wait::wait(queue);
                std::cout << "cuda memcpy" << std::endl;
                using TApi = typename cuda::Queue<T_Device>::TApi;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(alpaka::getNativeHandle(queue.m_device)));
                // Initiate the memory copy.
                auto copyKind
                    = cuda::Memcpy<ALPAKA_TYPE(internal::getApi(dest)), ALPAKA_TYPE(internal::getApi(source))>::kind;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memcpyAsync(
                    (void*) alpaka::data(dest),
                    (void*) alpaka::data(source),
                    dest.getExtent().x() * sizeof(typename T_Dest::type),
                    copyKind,
                    internal::getNativeHandle(queue)));
            }
        };
    } // namespace internal
} // namespace alpaka

#endif
