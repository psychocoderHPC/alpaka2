/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/MdSpan.hpp"

#include <cassert>
#include <tuple>
#include <type_traits>

namespace alpaka
{

    template<typename... Args>
    consteval size_t count_arguments(Args&&...)
    {
        return sizeof...(Args);
    }

} // namespace alpaka

#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__))
#    define ALPAKA_DEVICE_COMPILE 1
#else
#    define ALPAKA_DEVICE_COMPILE 0
#endif

/* select namespace depending on __CUDA_ARCH__ compiler flag*/
#if(ALPAKA_DEVICE_COMPILE == 1)
#    define ALPAKA_DEVICE_GLOBAL_NAMESPACE(id) using namespace ALPAKA_PP_CAT(alpaka_onAcc, id)
#else
#    define ALPAKA_DEVICE_GLOBAL_NAMESPACE(id) using namespace ALPAKA_PP_CAT(alpaka_onHost, id)
#endif

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    define ALPAKA_DEVICE_GLOBAL_DATA(id, dataType, name, ...)                                                        \
        namespace ALPAKA_PP_CAT(alpaka_onAcc, id)                                                                     \
        {                                                                                                             \
            __device__ std::type_identity_t<ALPAKA_PP_REMOVE_BRACKETS dataType> ALPAKA_PP_CAT(name, id)               \
                = __VA_OPT__({__VA_ARGS__});                                                                          \
        }
#else
#    define ALPAKA_DEVICE_GLOBAL_DATA(id, dataType, name, ...)

#endif

#if ALPAKA_DEVICE_COMPILE
#    define ALPAKA_DEVICE_GLOBAL_ACCESS(name, id)                                                                     \
        [[maybe_unused]] __device__ constexpr auto name = alpaka::onAcc::GlobalDeviceMemoryWrapper<                   \
            ALPAKA_PP_CAT(globalVariables, id)::ALPAKA_PP_CAT(GlobalStorage, id)>                                     \
        {                                                                                                             \
        }
#else
#    define ALPAKA_DEVICE_GLOBAL_ACCESS(name, id)                                                                     \
        [[maybe_unused]] constexpr auto name = alpaka::onAcc::GlobalDeviceMemoryWrapper<                              \
            ALPAKA_PP_CAT(globalVariables, id)::ALPAKA_PP_CAT(GlobalStorage, id)>                                     \
        {                                                                                                             \
        }
#endif

namespace alpaka::onAcc
{
    template<typename T_Storage>
    struct GlobalDeviceMemoryWrapper : private T_Storage
    {
        constexpr decltype(auto) get() const
        {
            return alpaka::unWrapp(T_Storage::get());
        }

        constexpr operator std::reference_wrapper<typename T_Storage::type>()
        {
            return T_Storage::get();
        }
    };

#define ALPAKA_DEVICE_GLOBAL_CREATE(location, dataType, id, name, ...)                                                \
    ALPAKA_DEVICE_GLOBAL_DATA(id, dataType, name, __VA_ARGS__)                                                        \
    namespace ALPAKA_PP_CAT(alpaka_onHost, id)                                                                        \
    {                                                                                                                 \
        [[maybe_unused]] std::type_identity_t<ALPAKA_PP_REMOVE_BRACKETS dataType> ALPAKA_PP_CAT(name, id)             \
            = __VA_OPT__({__VA_ARGS__});                                                                              \
    }                                                                                                                 \
    namespace ALPAKA_PP_CAT(globalVariables, id)                                                                      \
    {                                                                                                                 \
        ALPAKA_DEVICE_GLOBAL_NAMESPACE(id);                                                                           \
        struct ALPAKA_PP_CAT(GlobalStorage, id)                                                                       \
        {                                                                                                             \
            using type = ALPAKA_PP_REMOVE_BRACKETS dataType;                                                          \
            ALPAKA_FN_ACC auto get() const                                                                            \
            {                                                                                                         \
                return std::conditional_t<                                                                            \
                    std::is_array_v<ALPAKA_PP_REMOVE_BRACKETS dataType>,                                              \
                    alpaka::MdSpanArray<ALPAKA_PP_REMOVE_BRACKETS dataType>,                                          \
                    std::reference_wrapper<ALPAKA_PP_REMOVE_BRACKETS dataType>>{ALPAKA_PP_CAT(name, id)};             \
            }                                                                                                         \
        };                                                                                                            \
    }                                                                                                                 \
    ALPAKA_DEVICE_GLOBAL_ACCESS(name, id)

#define ALPAKA_DEVICE_GLOBAL(type, name, ...)                                                                         \
    ALPAKA_DEVICE_GLOBAL_CREATE(__device__, type, __COUNTER__, name, __VA_ARGS__)

} // namespace alpaka::onAcc
