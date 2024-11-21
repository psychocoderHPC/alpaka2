/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver
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
//! \param sharedMemExtents size of the shared memory box
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct StencilKernel
{
    template<typename TAcc, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto const uCurrBuf,
        auto uNextBuf,
        alpaka::Vec<TIdx, 2u> const chunkSize,
        alpaka::concepts::CVector auto sharedMemExtents,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        auto sdata = alpaka::onAcc::declareSharedMdArray<double>(acc, sharedMemExtents);

        // Get indexes
        auto const gridBlockIdx = acc[alpaka::layer::block].idx();
        auto const blockStartIdx = gridBlockIdx * chunkSize;

        for(auto idx2d : alpaka::makeIter(acc, alpaka::iter::withinDataFrame, sharedMemExtents + 0u))
        {
            auto bufIdx = idx2d + blockStartIdx;
            sdata[idx2d] = uCurrBuf[bufIdx];
        }

        alpaka::onAcc::syncBlockThreads(acc);

        // Each kernel executes one element
        double const rX = dt / (dx * dx);
        double const rY = dt / (dy * dy);

        constexpr auto top = alpaka::CVec<uint32_t, -1u, 0u>{};
        constexpr auto bottom = alpaka::CVec<uint32_t, 1u, 0u>{};
        constexpr auto left = alpaka::CVec<uint32_t, 0u, -1u>{};
        constexpr auto right = alpaka::CVec<uint32_t, 0u, 1u>{};

        // go over only core cells
        // alpaka::Vec{1, 1}; offset for halo above and to the left
        for(auto idx2D :
            alpaka::makeIter(acc, alpaka::iter::withinDataFrame, alpaka::Vec{1u, 1u}, alpaka::Vec{16u, 16u}))
        {
            auto bufIdx = idx2D + blockStartIdx;

            uNextBuf[bufIdx] = sdata[idx2D] * (1.0 - 2.0 * rX - 2.0 * rY) + sdata[idx2D + left] * rX
                               + sdata[idx2D + right] * rX + sdata[idx2D + top] * rY + sdata[idx2D + bottom] * rY;
        }
    }
};
