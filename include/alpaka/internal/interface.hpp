/* Copyright 2024 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/demangledName.hpp"

namespace alpaka
{
    /** alpaka internal implementations.
     *
     * @attention do not use any functions from this namespace in our user applications.
     *          The interface can change at any time without further notice and is for internal use only.
     */
    namespace internal
    {
        struct GetStaticName
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()([[maybe_unused]] T_Any const& any) const
                {
                    if constexpr(requires { T_Any::getName(); })
                        return T_Any::getName();
                    else
                        return onHost::demangledName(any);
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

        struct GetApi
        {
            template<typename T_Any>
            struct Op
            {
                inline constexpr auto operator()(auto&& any) const -> decltype(any.getApi())
                {
                    return any.getApi();
                }
            };
        };

        inline constexpr auto getApi(auto&& any) -> decltype(GetApi::Op<ALPAKA_TYPEOF(any)>{}(any))
        {
            return GetApi::Op<std::decay_t<decltype(any)>>{}(any);
        }

        template<typename T_Any>
        inline constexpr auto getApi(onHost::Handle<T_Any>&& anyHandle)
        {
            return GetApi::Op<ALPAKA_TYPEOF(*anyHandle.get())>{}(*anyHandle.get());
        }

        struct GetExecutor
        {
            template<typename T_Any>
            struct Op
            {
                inline constexpr auto operator()(auto&& any) const -> decltype(any.getExecutor())
                {
                    return any.getExecutor();
                }
            };
        };

        template<typename T_Any>
        inline constexpr auto getExecutor(T_Any&& any) -> decltype(GetExecutor::Op<ALPAKA_TYPEOF(any)>{}(any))
        {
            return GetExecutor::Op<std::decay_t<T_Any>>{}(ALPAKA_FORWARD(any));
        }

        template<typename T_Any>
        inline constexpr auto getExecutor(onHost::Handle<T_Any>&& anyHandle)
        {
            return GetExecutor::Op<ALPAKA_TYPEOF(*anyHandle.get())>{}(*anyHandle.get());
        }

        struct GetDeviceType
        {
            template<typename T_Any>
            struct Op
            {
                inline constexpr auto operator()(auto&& any) const -> decltype(any.getDeviceKind())
                {
                    return any.getDeviceKind();
                }
            };
        };

        inline constexpr auto getDeviceKind(auto&& any) -> decltype(GetDeviceType::Op<ALPAKA_TYPEOF(any)>{}(any))
        {
            return GetDeviceType::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct GetAlignment
        {
            template<typename T_Any>
            struct Op
            {
                constexpr auto operator()(auto&& any) const requires requires { any.getAlignment(); }
                {
                    return any.getAlignment();
                }

                constexpr auto operator()(auto&& any) const
                {
                    alpaka::unused(any);
                    return Alignment<>{};
                }
            };
        };

        constexpr auto getAlignment(auto&& any)
        {
            return GetAlignment::Op<std::decay_t<decltype(any)>>{}(any);
        }

        /** Load data from a data source as SIMD vector
         *
         * A data source is not required to have physical stored data, it can also be a generator, therefore only the
         * data source knows how load create aSIMD vector.
         */
        struct LoadAsSimd
        {
            template<typename T_AnyDataSource, alpaka::concepts::Alignment T_Alignment, alpaka::concepts::Vector T_Idx>
            struct Op
            {
                /** Get data as SIMD vector
                 *
                 * @see loadAsSimd for more details.
                 */
                template<uint32_t T_simdWidth>
                constexpr auto load(auto&& anyDataSource, T_Alignment dataAlignment, T_Idx const& index) const;
            };
        };

        /** Get data as SIMD vector
         *
         * Load T_simdWidth contiguous data staring from index. The data is contiguous in the fast moving dimension of
         * index.
         *
         * @tparam T_simdWidth number of elements in the SIMD vector
         * @param anyDataSource data source to load data from
         * @param dataAlignment Alignment of the data source resulting SIMD vector. This can be smaller or equal
         * compared to the data source alignment due to possible offsets applied before.
         * @param index Offset index relative to the first element of data source.
         * @return SIMD vector with data loaded from the data source, aligned to dataAlignment
         */
        template<uint32_t T_simdWidth>
        constexpr auto loadAsSimd(auto&& anyDataSource, auto dataAlignment, auto const& index)
        {
            return LoadAsSimd::Op<ALPAKA_TYPEOF(anyDataSource), ALPAKA_TYPEOF(dataAlignment), ALPAKA_TYPEOF(index)>{}
                .template load<T_simdWidth>(ALPAKA_FORWARD(anyDataSource), dataAlignment, index);
        }
    } // namespace internal
} // namespace alpaka
