/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc.hpp"

#include <cstdint>
#include <new>

namespace alpaka::onAcc
{
    // Concept to check if a type is a container
    template<typename T>
    concept Container = requires(T t) {
        typename T::value_type;
        t.size();
        t[0];
    };

    template<uint32_t T_width>
    inline constexpr auto* reinterpretPtr(auto&& data)
    {
        using DataTypeType = std::remove_reference_t<decltype(data)>;
        using DstType = std::conditional_t<
            std::is_const_v<DataTypeType>,
            Simd<ALPAKA_TYPEOF(data), T_width> const*,
            Simd<ALPAKA_TYPEOF(data), T_width>*>;

        return reinterpret_cast<DstType>(&data);
    }

    template<uint32_t T_maxWidthInByte, typename T_Type>
    inline constexpr auto simdWidth()
    {
        constexpr uint32_t maxNumElements = T_maxWidthInByte / sizeof(T_Type);
        constexpr uint32_t width = std::max(maxNumElements, 1u);
        return width;
    }

    template<uint32_t T_width>
    constexpr auto executeDo(auto const& acc, auto& iter, auto&& func, auto&&... data)
    {
        auto const idx = *iter;
        func(acc, *reinterpretPtr<T_width>(data[idx])...);
        ++iter;
        return true;
    }

    template<uint32_t T_width, uint32_t... T_repeat>
    constexpr void execute(
        auto const& acc,
        auto& iter,
        std::integer_sequence<uint32_t, T_repeat...>,
        auto&& func,
        auto&&... data)
    {
        ((T_repeat + 1 != 0u && executeDo<T_width>(acc, iter, ALPAKA_FORWARD(func), ALPAKA_FORWARD(data)...)), ...);
    }

    template<
        typename T_Type,
        uint32_t T_MaxSimdWitdthInByte, // 16 for CUDA
        uint32_t T_memAlignInByte, // 128 for CUDA
        uint32_t T_maxConcurrencyInByte,
        uint32_t T_cacheInByte> // 16 for CUDA
    inline constexpr auto simdWidthContiguous()
    {
        constexpr uint32_t maxSimdBytes
            = std::min(std::min(T_MaxSimdWitdthInByte, T_memAlignInByte), T_maxConcurrencyInByte);
        // T_maxConcurrencyInByte);

        constexpr uint32_t numElemPerSimd = maxSimdBytes / sizeof(T_Type);
        constexpr uint32_t simdWidth = std::max(numElemPerSimd, 1u);

        return simdWidth;
    }

#define ALPAKA_SIMD_V2 1

    // forEach function interface
    template<uint32_t T_maxWidthInByte, uint32_t T_maxConcurrencyInByte>
    ALPAKA_FN_ACC constexpr void forEach(
        auto const& acc,
        auto const workGroup,
        alpaka::concepts::Vector auto ex,
        auto&& func,
        auto&& data0,
        auto&&... dataN)
    {
        auto extents = typename ALPAKA_TYPEOF(ex)::UniVec{ex};
        using ValueType = ALPAKA_TYPEOF(data0[ALPAKA_TYPEOF(extents)::all(0)]);
#if ALPAKA_SIMD_V2
        constexpr uint32_t maxSimdWidth = 16;

        // 16 for CUDA
        constexpr uint32_t cachlineBytes =
#    ifdef __cpp_lib_hardware_interference_size
#        if (ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1) || (ALPAKA_LANG_CUDA && __CUDA_ARCH__)
            16;
#        else
            std::hardware_constructive_interference_size;
#        endif
#    else
            64; // Fallback value, typically 64 bytes
#    endif
#    if __CUDA_ARCH__
        static_assert(cachlineBytes == 16 && maxSimdWidth == 16);
#    endif
        constexpr uint32_t width
            = simdWidthContiguous<ValueType, maxSimdWidth, T_maxWidthInByte, T_maxConcurrencyInByte, cachlineBytes>();
        // = simdWidth<maxSimdWidth*sizeof(ValueType),ValueType>();
#else
        constexpr auto width = simdWidth<T_maxWidthInByte, ValueType>();
#endif
        if constexpr(width != 1u)
        {
            auto const wSize = workGroup.size(acc).x();
#if ALPAKA_SIMD_V2
            constexpr uint32_t simdWidthInByte = width * sizeof(ValueType);
            constexpr uint32_t maxNumRepetitions = std::max(T_maxConcurrencyInByte / simdWidthInByte, 1u);
            constexpr uint32_t numRepetitionsPerCacheline = std::max(cachlineBytes / simdWidthInByte, 1u);
            constexpr uint32_t numRepetitions
                = std::max((maxNumRepetitions / numRepetitionsPerCacheline) * numRepetitionsPerCacheline, 1u);

            auto const numElemOneRound = width * numRepetitions * wSize;
            auto const numSimdElem = extents.x() / numElemOneRound;
            auto const remainderNumElemOffset = numSimdElem * numElemOneRound;

#    if ALPAKA_SIMD_PRINT
            std::cout << "typesize" << sizeof(ValueType) << "simd = " << width
                      << " numRepetitions = " << numRepetitions << " numElemOneRound = " << numElemOneRound
                      << " numSimdElem = " << numSimdElem << " remainderNumElemOffset = " << remainderNumElemOffset
                      << " maxNumRepetitions = " << maxNumRepetitions << std::endl;
#    endif
#else
            constexpr auto const numRepetitions = std::max(T_maxConcurrencyInByte / width, 1u);
            auto const numElemOneRound = width * numRepetitions * wSize;
            auto const numSimdElem = extents.x() / numElemOneRound;
            auto const remainderNumElemOffset = numSimdElem * numElemOneRound;
#    if ALPAKA_SIMD_PRINT
            std::cout << "typesize" << sizeof(ValueType) << "simd = " << width
                      << " numRepetitions = " << numRepetitions << " numElemOneRound = " << numElemOneRound
                      << " numSimdElem = " << numSimdElem << " remainderNumElemOffset = " << remainderNumElemOffset
                      << std::endl;
#    endif
#endif
            auto domainSize = extents;
            domainSize.x() = remainderNumElemOffset;
            auto stride = ALPAKA_TYPEOF(extents)::all(1);
            stride.x() = width;

            using IdxType = ALPAKA_TYPEOF(extents);
            auto simdIdxContainer = onAcc::makeIdxMap(acc, workGroup, IdxRange{IdxType::all(0), domainSize, stride});

            for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
            {
                execute<width>(
                    acc,
                    iter,
                    std::make_integer_sequence<uint32_t, numRepetitions>{},
                    ALPAKA_FORWARD(func),
                    ALPAKA_FORWARD(data0),
                    ALPAKA_FORWARD(dataN)...);
            }

            auto remainderDomainSize = extents;
            remainderDomainSize.x() = remainderNumElemOffset;
            for(auto idx : onAcc::makeIdxMap(acc, workGroup, IdxRange{remainderDomainSize, extents}))
            {
                func(acc, *reinterpretPtr<1u>(data0[idx]), *reinterpretPtr<1u>(dataN[idx])...);
            }
        }
        else
            for(auto idx : onAcc::makeIdxMap(acc, workGroup, IdxRange{extents}))
            {
                func(acc, *reinterpretPtr<1u>(data0[idx]), *reinterpretPtr<1u>(dataN[idx])...);
            }
    }

} // namespace alpaka::onAcc
