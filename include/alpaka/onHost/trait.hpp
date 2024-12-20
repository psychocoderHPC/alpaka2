/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/api/executor.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/tag.hpp"

#include <type_traits>

namespace alpaka::onHost
{
    namespace trait
    {
        struct IsPlatformAvailable
        {
            template<alpaka::concepts::Api T_Api>
            struct Op : std::false_type
            {
            };
        };

        struct IsMappingSupportedBy
        {
            template<typename T_Mapping, typename T_Device>
            struct Op : std::false_type
            {
            };
        };

        template<typename T_Mapping, concepts::DeviceHandle T_DeviceHandle>
        struct IsMappingSupportedBy::Op<T_Mapping, T_DeviceHandle>
            : IsMappingSupportedBy::Op<T_Mapping, typename T_DeviceHandle::element_type>
        {
        };

        template<typename T_Kernel, typename T_Spec>
        struct BlockDynSharedMemBytes
        {
            BlockDynSharedMemBytes(T_Kernel kernel, T_Spec spec)
            {
            }

            uint32_t operator()(auto const executor, auto const&... args) const
            {
                return 0;
            }
        };

        template<typename T_Executor, typename T_Spec, typename T_KernelBundle>
        struct GetDynSharedMemBytes
        {
            static constexpr bool zeroSharedMemory = true;

            uint32_t operator()(
                T_Executor const executor,
                T_Spec const spec,
                [[maybe_unused]] T_KernelBundle const& kernelBundle) const
            {
                return 0u;
            }
        };

        template<typename T_Executor, typename T_Spec, typename T_KernelFn, typename... T_Args>
        requires requires() { std::declval<T_KernelFn>().dynSharedMemBytes; } || requires() {
            BlockDynSharedMemBytes<T_KernelFn, T_Spec>{std::declval<T_KernelFn>(), std::declval<T_Spec>()}(
                std::declval<T_Executor>(),
                std::declval<remove_restrict_t<std::decay_t<T_Args>>>()...);
        }
        struct GetDynSharedMemBytes<T_Executor, T_Spec, KernelBundle<T_KernelFn, T_Args...>>
        {
            uint32_t operator()(
                T_Executor const executor,
                T_Spec const spec,
                [[maybe_unused]] KernelBundle<T_KernelFn, T_Args...> const& kernelBundle) const
            {
                if constexpr(requires {
                                 BlockDynSharedMemBytes{kernelBundle.m_kernelFn, spec}(
                                     executor,
                                     std::declval<remove_restrict_t<std::decay_t<T_Args>>>()...);
                             })
                {
                    return std::apply(
                        [&](auto const&... args)
                        {
                            return BlockDynSharedMemBytes{kernelBundle.m_kernelFn, spec}(
                                executor,
                                remove_restrict_t<std::decay_t<T_Args>>()...);
                        },
                        kernelBundle.m_args);
                }
                else
                    return kernelBundle.m_kernelFn.dynSharedMemBytes;
            }
        };

        template<typename T_Executor, typename T_Spec, typename T_KernelBundle>
        struct HasUserDefinedDynSharedMemBytes : std::true_type
        {
        };

        template<typename T_Executor, typename T_Spec, typename T_KernelBundle>
        requires(trait::GetDynSharedMemBytes<T_Executor, T_Spec, T_KernelBundle>::zeroSharedMemory == true)
        struct HasUserDefinedDynSharedMemBytes<T_Executor, T_Spec, T_KernelBundle> : std::false_type
        {
        };
    } // namespace trait

    consteval bool isPlatformAvaiable(alpaka::concepts::Api auto api)
    {
        return trait::IsPlatformAvailable::Op<std::decay_t<decltype(api)>>::value;
    }

    consteval bool isExecutorSupportedBy(auto executor, concepts::DeviceHandle auto const& deviceHandle)
    {
        return trait::IsMappingSupportedBy::Op<ALPAKA_TYPEOF(executor), ALPAKA_TYPEOF(deviceHandle)>::value;
    }

    constexpr auto supportedMappings(concepts::DeviceHandle auto deviceHandle)
    {
        return meta::filter(
            [&](auto executor) constexpr { return isExecutorSupportedBy(executor, deviceHandle); },
            exec::availableMappings);
    }

    template<typename T_Executor, typename T_Spec, typename T_KernelBundle>
    constexpr uint32_t getDynSharedMemBytes(T_Executor const executor, T_Spec spec, T_KernelBundle const& kernelBundle)
    {
        return trait::GetDynSharedMemBytes<T_Executor, T_Spec, T_KernelBundle>{}(executor, spec, kernelBundle);
    }

    template<typename T_Executor, typename T_Spec, typename T_KernelBundle>
    consteval bool hasUserDefinedDynSharedMemBytes(
        T_Executor const executor,
        T_Spec spec,
        T_KernelBundle const& kernelBundle)
    {
        return trait::HasUserDefinedDynSharedMemBytes<T_Executor, T_Spec, T_KernelBundle>::value;
    }

} // namespace alpaka::onHost
