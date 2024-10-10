/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Acc.hpp"
#include "alpaka/acc/Layer.hpp"
#include "alpaka/acc/Omp.hpp"
#include "alpaka/acc/One.hpp"
#include "alpaka/acc/Serial.hpp"
#include "alpaka/core/config.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{
    struct NoEntry
    {
    };

    template<typename T_AdditionalLayer = decltype(Dict{DictEntry(NoEntry{}, NoEntry{})})>
    inline auto makeAcc(
        mapping::CpuBlockSerialThreadOne,
        auto const& threadBlocking,
        T_AdditionalLayer const& additionalLayer = T_AdditionalLayer{DictEntry(NoEntry{}, NoEntry{})})
    {
        return Acc{
            std::make_tuple(layer::block, layer::thread, internal_layer::threadCommand),
            joinDict(
                Dict{
                    DictEntry(layer::block, Serial{threadBlocking.m_numBlocks}),
                    DictEntry(layer::thread, One{threadBlocking.m_numThreads}),
                    DictEntry(internal_layer::threadCommand, ThreadCommand{})},
                additionalLayer)};
    }

#if ALPAKA_OMP
    template<typename T_AdditionalLayer = decltype(Dict{DictEntry(NoEntry{}, NoEntry{})})>
    inline auto makeAcc(
        mapping::CpuBlockOmpThreadOne,
        auto const& threadBlocking,
        T_AdditionalLayer const& additionalLayer = T_AdditionalLayer{DictEntry(NoEntry{}, NoEntry{})})
    {
        return Acc{
            std::make_tuple(layer::block, layer::thread, internal_layer::threadCommand),
            joinDict(
                Dict{
                    DictEntry(layer::block, Omp{threadBlocking.m_numBlocks}),
                    DictEntry(layer::thread, One{threadBlocking.m_numThreads}),
                    DictEntry(internal_layer::threadCommand, ThreadCommand{})},
                additionalLayer)};
    }

    template<typename T_AdditionalLayer = decltype(Dict{DictEntry(NoEntry{}, NoEntry{})})>
    inline auto makeAcc(
        decltype(mapping::cpuBlockOmpThreadOmp),
        auto const& threadBlocking,
        T_AdditionalLayer const& additionalLayer = T_AdditionalLayer{DictEntry(NoEntry{}, NoEntry{})})
    {
        return Acc{
            std::make_tuple(layer::block, layer::thread, internal_layer::threadCommand),
            joinDict(
                Dict{
                    DictEntry(layer::block, Omp{threadBlocking.m_numBlocks}),
                    DictEntry(layer::thread, Omp{threadBlocking.m_numThreads}),
                    DictEntry(internal_layer::threadCommand, ThreadCommand{})},
                additionalLayer)};
    }
#endif

} // namespace alpaka
