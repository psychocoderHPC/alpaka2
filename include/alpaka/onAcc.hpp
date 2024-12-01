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

    template<typename T, alpaka::concepts::CVector T_Extent>
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
        concepts::IdxTraversing T_Traverse = traverse::Flat,
        concepts::IdxMapping T_IdxMapping = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxMapping idxMapping = T_IdxMapping{})
    {
        return internal::MakeIter::
            Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(DomainSpec{workGroup, range}), T_Traverse, T_IdxMapping>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxMapping);
    }

    template<
        concepts::IdxTraversing T_Traverse = traverse::Tiled,
        concepts::IdxMapping T_IdxMapping = layout::Optimized>
    ALPAKA_FN_HOST_ACC constexpr auto makeIdxMap(
        auto const& acc,
        auto const workGroup,
        alpaka::concepts::HasStaticDim auto const range,
        T_Traverse traverse = T_Traverse{},
        T_IdxMapping idxMapping = T_IdxMapping{}) requires(ALPAKA_TYPE(range)::dim() == 1u)
    {
        return internal::MakeIter::
            Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(DomainSpec{workGroup, range}), T_Traverse, T_IdxMapping>{}(
                acc,
                DomainSpec{workGroup, range},
                traverse,
                idxMapping);
    }

} // namespace alpaka::onAcc
