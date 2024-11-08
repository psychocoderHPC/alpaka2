/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Blocking.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/Handle.hpp"
#include "alpaka/core/common.hpp"

namespace alpaka
{
    namespace internal
    {
        struct MakePlatform
        {
            template<typename T_Api>
            struct Op
            {
                auto operator()(T_Api&& api) const;
            };
        };

        static auto makePlatform(auto&& api)
        {
            return MakePlatform::Op<std::decay_t<decltype(api)>>{}(api);
        }

        struct GetStaticName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const&) const
                {
                    return T_Any::getName();
                }
            };
        };

        struct GetName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getName();
                }
            };
        };

        struct GetDeviceCount
        {
            template<typename T_Platform>
            struct Op
            {
                uint32_t operator()(T_Platform& platform) const
                {
                    return platform.getDeviceCount();
                }
            };
        };

        struct MakeDevice
        {
            template<typename T_Platform>
            struct Op
            {
                auto operator()(auto& platform, uint32_t idx) const
                {
                    return platform.makeDevice(idx);
                }
            };
        };

        struct GetNativeHandle
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getNativeHandle();
                }
            };
        };

        static auto getNativeHandle(auto&& any)
        {
            return GetNativeHandle::Op<std::decay_t<decltype(any)>>{}(any);
        }

        struct MakeQueue
        {
            template<typename T_Device>
            struct Op
            {
                auto operator()(T_Device& device) const
                {
                    return device.makeQueue();
                }
            };
        };

        struct Wait
        {
            template<typename T_Any>
            struct Op
            {
                void operator()(T_Any const& any) const
                {
                    any.wait();
                }
            };

            static auto wait(auto&& any)
            {
                return Op<std::decay_t<decltype(any)>>{}(any);
            }
        };

        struct Enqueue
        {
            template<typename T_Queue, typename T_Mapping, typename T_BlockCfg, typename T_KernelBundle>
            struct Kernel
            {
                void operator()(
                    T_Queue& queue,
                    T_Mapping const mapping,
                    T_BlockCfg const& blockCfg,
                    T_KernelBundle kernelBundle) const
                {
                    queue.enqueue(mapping, blockCfg, std::move(kernelBundle));
                }
            };

            template<typename T_Queue, typename T_Task>
            struct Task
            {
                void operator()(T_Queue& queue, T_Task task) const
                {
                    queue.enqueue(std::move(task));
                }
            };
        };

        inline void enqueue(auto& queue, auto task)
        {
            Enqueue::Task<std::decay_t<decltype(queue)>, std::decay_t<decltype(task)>>{}(queue, std::move(task));
        }

        template<typename TKernelFn, typename... TArgs>
        inline void enqueue(
            auto& queue,
            auto const mapping,
            auto const& blockCfg,
            KernelBundle<TKernelFn, TArgs...> kernelBundle)
        {
            Enqueue::Kernel<
                std::decay_t<decltype(queue)>,
                std::decay_t<decltype(mapping)>,
                std::decay_t<decltype(blockCfg)>,
                KernelBundle<TKernelFn, TArgs...>>{}(queue, mapping, blockCfg, std::move(kernelBundle));
        }

        struct AdjustThreadBlocking
        {
            template<typename T_Device, typename T_Mapping, typename T_DataBlocking, typename T_KernelBundle>
            struct Op
            {
                auto operator()(
                    T_Device const&,
                    T_Mapping const& mapping,
                    T_DataBlocking const& blockCfg,
                    T_KernelBundle const& kernelBundle) const
                {
                    return ThreadBlocking{blockCfg.m_numBlocks, blockCfg.m_numThreads};
                }
            };
        };

        template<typename T_NumBlocks, typename T_NumThreads, typename TKernelFn, typename... TArgs>
        static auto adjustThreadBlocking(
            auto const& device,
            auto const& mapping,
            DataBlocking<T_NumBlocks, T_NumThreads> const& dataBlocking,
            KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
        {
            return AdjustThreadBlocking::Op<
                std::decay_t<decltype(device)>,
                std::decay_t<decltype(mapping)>,
                DataBlocking<T_NumBlocks, T_NumThreads>,
                KernelBundle<TKernelFn, TArgs...>>{}(device, mapping, dataBlocking, kernelBundle);
        }

        struct Data
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.data();
                }
            };

            static auto data(auto&& any)
            {
                return Op<std::decay_t<decltype(any)>>{}(any);
            }
        };

        struct GetApi
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.getApi();
                }
            };
        };

        inline auto getApi(auto&& any)
        {
            return GetApi::Op<std::decay_t<decltype(any)>>{}(any);
        }

        struct Alloc
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct Memcpy
        {
            template<typename T_Queue, typename T_Dest, typename T_Source, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, T_Dest&, T_Source const&, T_Extents const&) const;
            };
        };
    } // namespace internal
} // namespace alpaka
