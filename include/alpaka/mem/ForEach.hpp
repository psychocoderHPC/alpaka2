/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc.hpp"

#include <cstdint>

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

    // forEach function interface
    template<uint32_t T_maxWidthInByte, uint32_t T_maxElementConcurrency>
    ALPAKA_FN_ACC constexpr void forEach(
        auto const& acc,
        auto const workGroup,
        auto numElements,
        auto&& func,
        auto&& data0,
        auto&&... dataN)
    {
        constexpr auto width = simdWidth<T_maxWidthInByte, ALPAKA_TYPEOF(data0[0u])>();

        auto const wSize = workGroup.size(acc).x();
        constexpr auto const maxNumRepetitions = std::max(T_maxElementConcurrency / width, 1u);
        auto const numElemOneRound = width * maxNumRepetitions * wSize;
        auto const numSimdElem = numElements / numElemOneRound;
        auto const remainderNumElemOffset = numSimdElem * numElemOneRound;
        using IdxType = ALPAKA_TYPEOF(numElements);
        auto simdIdxContainer = onAcc::makeIdxMap(
            acc,
            workGroup,
            IdxRange{Vec{static_cast<IdxType>(0)}, Vec{remainderNumElemOffset}, Vec{static_cast<IdxType>(width)}});

        for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
        {
#if 1
            // for(uint32_t i = 0; i < maxNumRepetitions; ++i)
            {
                // if(iter != simdIdxContainer.end())
                {

                    execute<width>(
                        acc,
                        iter,
                        std::make_integer_sequence<uint32_t, maxNumRepetitions>{},
                        ALPAKA_FORWARD(func),
                        ALPAKA_FORWARD(data0),
                        ALPAKA_FORWARD(dataN)...);
                }
            }
#else
            for(uint32_t i = 0; i < maxNumRepetitions; ++i)
            {
                // if(iter != simdIdxContainer.end())
                {
                    auto const idx = *iter;
                    func(acc, *reinterpretPtr<width>(data0[idx]), *reinterpretPtr<width>(dataN[idx])...);
                    ++iter;
                }
            }
#endif
        }

        for(auto idx : onAcc::makeIdxMap(
                acc,
                workGroup,
                IdxRange{Vec{static_cast<IdxType>(remainderNumElemOffset)}, Vec{numElements}}))
        {
            func(acc, *reinterpretPtr<1u>(data0[idx]), *reinterpretPtr<1u>(dataN[idx])...);
        }
    }

} // namespace alpaka::onAcc
