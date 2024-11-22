/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/Iter.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/internal.hpp"

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

    template<typename T, concepts::CVector T_Extent>
    constexpr decltype(auto) declareSharedMdArray(auto const& acc, T_Extent const& extent)
    {
        using CArrayType = typename CArrayType<T, T_Extent>::type;
        return MdSpanArray<CArrayType>{internalCompute::declareSharedVar<CArrayType>(acc)};
    }

    /**
     * ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from __host__ __device__ warning
     * is popping up and generated code is wrong.
     */
    template<
        iter::concepts::IdxTraversing T_Traverse = iter::traverse::Flat,
        iter::concepts::IdxMapping T_IdxMapping = iter::idxLayout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxMapping idxMapping = T_IdxMapping{})
    {
        return iter::internal::MakeIter::
            Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(iter::DomainSpec{workGroup, range}), T_Traverse, T_IdxMapping>{}(
                acc,
                iter::DomainSpec{workGroup, range},
                traverse,
                idxMapping);
    }

} // namespace alpaka::onAcc
