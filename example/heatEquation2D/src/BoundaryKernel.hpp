/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "analyticalSolution.hpp"
#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Applies boundary conditions
//! forward difference in t and second-order central difference in x
//!
//! \param uBuf grid values of u for each x, y and the current value of t:
//!                 u(x, y, t)  | t = t_current
//! \param chunkSize
//! \param pitch
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct BoundaryKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double* const uBuf,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::Vector auto const pitch,
        uint32_t step,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        constexpr uint32_t dim = 2u;
        using Idx = uint32_t;
        using IdxVec = alpaka::Vec<Idx, dim>;

        // Get extents(dimensions)
        auto const gridBlockExtent = acc[alpaka::layer::block].count();
        auto const blockThreadExtent = acc[alpaka::layer::thread].count();
        auto const numThreadsPerBlock = blockThreadExtent.product();

        // Get indexes
        auto const gridBlockIdx = acc[alpaka::layer::block].idx();
        auto const blockThreadIdx = acc[alpaka::layer::thread].idx();
        auto const threadIdx1D = alpaka::linearize(blockThreadExtent, blockThreadIdx);
        auto const blockStartIdx = gridBlockIdx * chunkSize;

        // Lambda function to apply boundary conditions
        auto applyBoundary = [&](auto const& globalIdxStart, auto const length, bool isRow)
        {
            for(auto i = threadIdx1D; i < length; i += numThreadsPerBlock)
            {
                auto idx2D = globalIdxStart + (isRow ? IdxVec{0, i} : IdxVec{i, 0});
                auto elem = getElementPtr(uBuf, idx2D, pitch);
                *elem = exactSolution(idx2D[1] * dx, idx2D[0] * dy, step * dt);
            }
        };

        // Apply boundary conditions for the top row
        if(gridBlockIdx[0] == 0)
        {
            applyBoundary(blockStartIdx + IdxVec{0, 1}, chunkSize[1], true);
        }

        // Apply boundary conditions for the bottom row
        if(gridBlockIdx[0] == gridBlockExtent[0] - 1)
        {
            applyBoundary(blockStartIdx + IdxVec{chunkSize[0] + 1, 1}, chunkSize[1], true);
        }

        // Apply boundary conditions for the left column
        if(gridBlockIdx[1] == 0)
        {
            applyBoundary(blockStartIdx + IdxVec{1, 0}, chunkSize[0], false);
        }

        // Apply boundary conditions for the right column
        if(gridBlockIdx[1] == gridBlockExtent[1] - 1)
        {
            applyBoundary(blockStartIdx + IdxVec{1, chunkSize[1] + 1}, chunkSize[0], false);
        }
    }
};
