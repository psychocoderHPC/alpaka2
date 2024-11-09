/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! \tparam T_SharedMemSize1D size of the shared memory box
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param uCurrBuf Current buffer with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNextBuf resulting grid values of u for each x, y pair and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param chunkSize The size of the chunk or tile that the user divides the problem into. This defines the size of the
//!                  workload handled by each thread block.
//! \param pitchCurr The pitch (or stride) in memory corresponding to the TDim grid in the accelerator's memory.
//!              This is used to calculate memory offsets when accessing elements in the current buffer.
//! \param pitchNext The pitch used to calculate memory offsets when accessing elements in the next buffer.
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
template<size_t T_SharedMemSize1D>
struct StencilKernel
{
    template<typename TAcc, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const uCurrBuf,
        double* const uNextBuf,
        alpaka::Vec<TIdx, 2u> const chunkSize,
        alpaka::Vec<TIdx, 2u> const pitchCurr,
        alpaka::Vec<TIdx, 2u> const pitchNext,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        using IdxVec = alpaka::Vec<TIdx, 2u>;

        auto& sdata = alpaka::declareSharedVar<double[T_SharedMemSize1D]>(acc);

        // Get extents(dimensions)
        auto const blockThreadExtent = acc[alpaka::layer::thread].count();
        auto const numThreadsPerBlock = blockThreadExtent.product();

        // Get indexes
        auto const gridBlockIdx = acc[alpaka::layer::block].idx();
        auto const blockThreadIdx = acc[alpaka::layer::thread].idx();
        auto const threadIdx1D = alpaka::linearize(blockThreadExtent, blockThreadIdx);
        auto const blockStartIdx = gridBlockIdx * chunkSize;

        constexpr IdxVec halo{2, 2};

        for(auto i = threadIdx1D; i < T_SharedMemSize1D; i += numThreadsPerBlock)
        {
            auto idx2d = alpaka::mapToND(chunkSize + halo, i);
            idx2d = idx2d + blockStartIdx;
            auto elem = getElementPtr(uCurrBuf, idx2d, pitchCurr);
            sdata[i] = *elem;
        }

        alpaka::syncBlockThreads(acc);

        // Each kernel executes one element
        double const rX = dt / (dx * dx);
        double const rY = dt / (dy * dy);

        // go over only core cells
        for(auto idx2D : alpaka::DataFrameIter{acc})
        {
            idx2D = idx2D + IdxVec{1, 1}; // offset for halo above and to the left
            auto localIdx1D = alpaka::linearize(chunkSize + halo, idx2D);


            auto bufIdx = idx2D + blockStartIdx;
            auto elem = getElementPtr(uNextBuf, bufIdx, pitchNext);

            *elem = sdata[localIdx1D] * (1.0 - 2.0 * rX - 2.0 * rY) + sdata[localIdx1D - 1] * rX
                    + sdata[localIdx1D + 1] * rX + sdata[localIdx1D - chunkSize[1] - halo[1]] * rY
                    + sdata[localIdx1D + chunkSize[1] + halo[1]] * rY;
        }
    }
};
