/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/atomicOp.hpp"

#include <type_traits>

namespace alpaka::onAcc
{
    //! Defines the parallelism hierarchy levels of alpaka
    namespace hierarchy
    {
        struct Grids
        {
        };

        struct Blocks
        {
        };

        struct Threads
        {
        };
    } // namespace hierarchy

    //! The atomic operation trait.
    namespace trait
    {
        //! The atomic operation trait.
        template<typename TOp, typename TAtomic, typename T, typename THierarchy, typename TSfinae = void>
        struct AtomicOp;
    } // namespace trait

    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename TOp, typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicOp(T* const addr, T const& value, THierarchy const& = THierarchy()) -> T
    {
        return trait::AtomicOp<TOp, ALPAKA_TYPE(apiCtx), T, THierarchy>::atomicOp(apiCtx, addr, value);
    }

    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    template<typename TOp, typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicOp(T* const addr, T const& compare, T const& value, THierarchy const& = THierarchy()) -> T
    {
        return trait::AtomicOp<TOp, ALPAKA_TYPE(apiCtx), T, THierarchy>::atomicOp(apiCtx, addr, compare, value);
    }

    //! Executes an atomic add operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicAdd(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAdd>(addr, value, hier);
    }

    //! Executes an atomic sub operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicSub(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicSub>(addr, value, hier);
    }

    //! Executes an atomic min operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicMin(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMin>(addr, value, hier);
    }

    //! Executes an atomic max operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicMax(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMax>(addr, value, hier);
    }

    //! Executes an atomic exchange operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicExch(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicExch>(addr, value, hier);
    }

    //! Executes an atomic increment operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicInc(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicInc>(addr, value, hier);
    }

    //! Executes an atomic decrement operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicDec(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicDec>(addr, value, hier);
    }

    //! Executes an atomic and operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicAnd(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAnd>(addr, value, hier);
    }

    //! Executes an atomic or operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicOr(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicOr>(addr, value, hier);
    }

    //! Executes an atomic xor operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicXor(T* const addr, T const& value, THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicXor>(addr, value, hier);
    }

    //! Executes an atomic compare-and-swap operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    template<typename T, typename THierarchy = hierarchy::Grids>
    constexpr auto atomicCas(T* const addr, T const& compare, T const& value, THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<AtomicCas>(addr, compare, value, hier);
    }
} // namespace alpaka::onAcc
