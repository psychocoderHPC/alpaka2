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
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto const uCurrBuf,
        auto uNextBuf,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::CVector auto sharedMemExtents,
        alpaka::concepts::Vector auto numNodes,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        using namespace alpaka;

        for(alpaka::concepts::Dim<2u> auto blockStartIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, numNodes, chunkSize}))
        {
            auto sdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);

            // avoid data race with the stencil calculation at the end
            onAcc::syncBlockThreads(acc);

            for(alpaka::concepts::Dim<2u> auto idx2d :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sharedMemExtents}))
            {
                auto bufIdx = idx2d + blockStartIdx;
                sdata[idx2d] = uCurrBuf[bufIdx];
            }

            onAcc::syncBlockThreads(acc);

            // Each kernel executes one element
            double const rX = dt / (dx * dx);
            double const rY = dt / (dy * dy);

            constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
            constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};

            // go over only core cells
            // Vec{1, 1}; offset for halo above and to the left
#if 0
            for(alpaka::concepts::Dim<2u> auto idx2D : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{chunkSize} >> 1u,
                    onAcc::traverse::tiled))
            {
                auto bufIdx = idx2D + blockStartIdx;

                uNextBuf[bufIdx] = sdata[idx2D] * (1.0 - 2.0 * rX - 2.0 * rY) + sdata[idx2D - xDir] * rX
                                   + sdata[idx2D + xDir] * rX + sdata[idx2D - yDir] * rY + sdata[idx2D + yDir] * rY;
            }
#else
            auto guardOffset = CVec<uint32_t, 1u, 1u>{};
            onAcc::forEach<8, 8>(
                acc,
                onAcc::worker::threadsInBlock,
                chunkSize,
                [&](auto const&, auto& next, auto const& self, auto const& mx,auto const& px, auto const& my,auto const& py) constexpr {
                    next = self.load() * (1.0 - 2.0 * rX - 2.0 * rY) + mx.load() * rX + px.load() * rX + my.load() * rY
                           + py.load() * rY;
                },
                uNextBuf.shift(blockStartIdx + guardOffset),
                sdata.shift(guardOffset),
                sdata.shift(guardOffset - xDir),
                sdata.shift(xDir + guardOffset),
                sdata.shift(guardOffset - yDir),
                sdata.shift(yDir + guardOffset));
#endif
        }
    }
};
