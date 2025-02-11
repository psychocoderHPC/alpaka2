/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/trait.hpp"
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

    template<uint32_t T_memAlignmentInByte, uint32_t T_width>
    inline constexpr auto* reinterpretPtr(auto&& data)
    {
        using DataTypeType = std::remove_reference_t<decltype(data)>;
        using DstType = std::conditional_t<
            std::is_const_v<DataTypeType>,
            Simd<T_memAlignmentInByte, ALPAKA_TYPEOF(data), T_width> const*,
            Simd<T_memAlignmentInByte, ALPAKA_TYPEOF(data), T_width>*>;

        return reinterpret_cast<DstType>(&data);
    }

    template<uint32_t T_memAlignmentInByte, uint32_t T_width>
    constexpr auto executeDo(auto const& acc, auto& iter, auto&& func, auto&&... data)
    {
        auto const idx = *iter;
        func(acc, *reinterpretPtr<T_memAlignmentInByte, T_width>(data[idx])...);
        ++iter;
        return true;
    }

    template<uint32_t T_memAlignmentInByte, uint32_t T_width, uint32_t... T_repeat>
    constexpr void execute(
        auto const& acc,
        auto& iter,
        std::integer_sequence<uint32_t, T_repeat...>,
        auto&& func,
        auto&&... data)
    {
        ((T_repeat + 1 != 0u
          && executeDo<T_memAlignmentInByte, T_width>(acc, iter, ALPAKA_FORWARD(func), ALPAKA_FORWARD(data)...)),
         ...);
    }

    template<
        typename T_Type,
        uint32_t T_maxConcurrencyInByte,
        uint32_t T_cacheInByte> // 16 for CUDA
    inline constexpr auto calcSimdWidth()
    {
        constexpr uint32_t maxSimdBytes = std::min(T_cacheInByte, T_maxConcurrencyInByte);
        // T_maxConcurrencyInByte);

        constexpr uint32_t numElemPerSimd = maxSimdBytes / sizeof(T_Type);
        constexpr uint32_t simdWidth = std::max(numElemPerSimd, 1u);

        return simdWidth;
    }

    // forEach function interface
    template<uint32_t T_maxConcurrencyInByte, size_t T_memAlignmentInByte = 0u>
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

        // @attention ALPAKA_TYPEOF() must be used because HIP 6.0 compile error that data0.getAlignment() is not a constant expression
        constexpr uint32_t minAlignmentFromDataInByte
            = static_cast<uint32_t>(std::min({ALPAKA_TYPEOF(data0)::getAlignment().x(), ALPAKA_TYPEOF(dataN)::getAlignment().x()...}));

        constexpr uint32_t dataAlignmentInByte
            = T_memAlignmentInByte != 0u ? static_cast<uint32_t>(alignof(ValueType)) : minAlignmentFromDataInByte;

        /** @todo `getApi` must be moved to alpaka namespace */
        constexpr uint32_t maxArchSimdWidth = getArchSimdWidth<ValueType>(thisApi());
        constexpr uint32_t cachlineBytes = getCachelineSize(thisApi());

        constexpr uint32_t width
            = std::min(maxArchSimdWidth, calcSimdWidth<ValueType, T_maxConcurrencyInByte, cachlineBytes>());

        if constexpr(width != 1u)
        {
            auto const wSize = workGroup.size(acc).x();

            constexpr uint32_t simdWidthInByte = width * sizeof(ValueType);
            constexpr uint32_t maxNumRepetitions = std::max(T_maxConcurrencyInByte / simdWidthInByte, 1u);
            constexpr uint32_t numRepetitionsPerCacheline = std::max(cachlineBytes / simdWidthInByte, 1u);
            constexpr uint32_t numRepetitions
                = std::max((maxNumRepetitions / numRepetitionsPerCacheline) * numRepetitionsPerCacheline, 1u);

            auto const numElemOneRound = width * numRepetitions * wSize;
            auto const numSimdElem = extents.x() / numElemOneRound;
            auto const remainderNumElemOffset = numSimdElem * numElemOneRound;

#if ALPAKA_SIMD_PRINT
            std::cout << "typesize" << sizeof(ValueType) << "simd = " << width
                      << " numRepetitions = " << numRepetitions << " numElemOneRound = " << numElemOneRound
                      << " numSimdElem = " << numSimdElem << " remainderNumElemOffset = " << remainderNumElemOffset
                      << " maxNumRepetitions = " << maxNumRepetitions << "align =" << minAlignmentFromData
                      << std::endl;
#endif

            auto domainSize = extents;
            domainSize.x() = remainderNumElemOffset;
            auto stride = ALPAKA_TYPEOF(extents)::all(1);
            stride.x() = width;

            using IdxType = ALPAKA_TYPEOF(extents);
            auto simdIdxContainer = onAcc::makeIdxMap(acc, workGroup, IdxRange{IdxType::all(0), domainSize, stride});

            for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
            {
                execute<dataAlignmentInByte, width>(
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
                func(
                    acc,
                    *reinterpretPtr<dataAlignmentInByte, 1u>(data0[idx]),
                    *reinterpretPtr<dataAlignmentInByte, 1u>(dataN[idx])...);
            }
        }
        else
            for(auto idx : onAcc::makeIdxMap(acc, workGroup, IdxRange{extents}))
            {
                func(
                    acc,
                    *reinterpretPtr<dataAlignmentInByte, 1u>(data0[idx]),
                    *reinterpretPtr<dataAlignmentInByte, 1u>(dataN[idx])...);
            }
    }

} // namespace alpaka::onAcc
