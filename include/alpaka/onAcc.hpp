/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file
 *
 * On some constexpr function signatures ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from
 * __host__ __device__ warning is popping up and generated code is wrong.
 */

#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/mem/Iter.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/internal.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/onAcc/traverse.hpp"

namespace alpaka::onAcc
{
    constexpr void syncBlockThreads(auto const& acc)
    {
        internalCompute::syncBlockThreads(acc);
    }

    template<typename T>
    constexpr decltype(auto) declareSharedVar(auto const& acc)
    {
        return internalCompute::declareSharedVar<T>(acc);
    }

    template<typename T, alpaka::concepts::CVector T_Extent>
    constexpr decltype(auto) declareSharedMdArray(auto const& acc, T_Extent const& extent)
    {
        using CArrayType = typename CArrayType<T, T_Extent>::type;
        return MdSpanArray<CArrayType>{internalCompute::declareSharedVar<CArrayType>(acc)};
    }

    /** Creates an index container that can be traversed with a range based for loop.
     *
     * @param workGroup participating thread description. More than one thread can have the same index within the
     * group. All worker with the same id will get the same index as result.
     * @param range Index range description.
     * @param traverse Policy to configure the method used to find the next valid index for a worker. @see namespace
     * traverse
     * @param idxLayout Policy to define how indecision will be mapped to worker threads. @see namsepsace layout
     * @return
     *
     * @{
     */
    template<concepts::IdxTraversing T_Traverse = traverse::Flat, concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{})
    {
        return internal::MakeIter::
            Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, range}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxLayout);
    }

    template<
        concepts::IdxTraversing T_Traverse = traverse::Tiled,
        concepts::IdxMapping T_IdxLayout = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        alpaka::concepts::HasStaticDim auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxLayout idxLayout = T_IdxLayout{}) requires(ALPAKA_TYPEOF(range)::dim() == 1u)
    {
        return internal::MakeIter::
            Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(DomainSpec{workGroup, range}), T_Traverse, T_IdxLayout>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxLayout);
    }

    /**
     * @}
     */

} // namespace alpaka::onAcc
