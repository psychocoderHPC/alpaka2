/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/tag.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace alpaka::onHost
{
    namespace internal::detail
    {
        /** Half open byte range [begin, end) a single thread is responsible for */
        struct ByteRange
        {
            size_t begin;
            size_t end;

            constexpr bool isEmpty() const
            {
                return begin >= end;
            }

            constexpr size_t numBytes() const
            {
                return end - begin;
            }
        };

        /** Split a copy of numBytes bytes into one contiguous byte range per thread of the grid
         *
         * The work is distributed over the bytes of the copy and not over its elements or rows. Distributing
         * elements or rows leaves threads idle as soon as there are fewer elements/rows than threads, e.g. copying
         * 10 elements of 100MB with a grid of 100 threads would only keep 10 threads busy.
         *
         * @param acc Accelerator interface providing thread/block information
         * @param numBytes Total number of bytes to copy
         * @return The byte range of the calling thread, empty if the thread has nothing to copy
         */
        constexpr ByteRange makeThreadByteRange(auto const& acc, size_t numBytes)
        {
            auto const threadIdxInGrid = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
            auto const gridSize = acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

            auto const linearIdx = static_cast<size_t>(alpaka::linearize(gridSize, threadIdxInGrid));
            auto const numThreads = static_cast<size_t>(gridSize.product());

            // the chunk size below divides by the number of threads
            ALPAKA_ASSERT_ACC(numThreads > size_t{0u});
            ALPAKA_ASSERT_ACC(linearIdx < numThreads);

            // ceiling division, the last threads get less work or none at all
            size_t const chunkBytes = (numBytes + numThreads - size_t{1u}) / numThreads;
            size_t const begin = linearIdx * chunkBytes;

            // this thread is beyond the end of the copy
            if(begin >= numBytes)
                return ByteRange{numBytes, numBytes};

            return ByteRange{begin, std::min(begin + chunkBytes, numBytes)};
        }

        /** Kernel for a contiguous parallel memcpy
         *
         * Each thread copies one contiguous byte range with a single std::memcpy.
         */
        struct ParallelMemcpyKernel
        {
            /** Execute a contiguous parallel memcpy
             *
             * @tparam T_AccInterface Accelerator interface type (provides idx and blockDim/gridDim)
             * @param acc Accelerator interface providing thread/block information
             * @param srcPtr Source buffer pointer
             * @param destPtr Destination buffer pointer
             * @param numBytes Total number of bytes to copy
             */
            template<alpaka::onAcc::concepts::Acc<ALPAKA_TYPEOF(api::host)> T_AccInterface>
            void operator()(T_AccInterface const& acc, void const* srcPtr, void* destPtr, size_t numBytes) const
            {
                // the grid size is the number of chunks, a frame specification would decouple both
                static_assert(
                    !T_AccInterface::launchedWithFrameSpec(),
                    "The parallel memcpy kernels must be enqueued with a thread specification.");

                auto const byteRange = makeThreadByteRange(acc, numBytes);
                if(byteRange.isEmpty())
                    return;

                std::memcpy(
                    static_cast<std::uint8_t*>(destPtr) + byteRange.begin,
                    static_cast<std::uint8_t const*>(srcPtr) + byteRange.begin,
                    byteRange.numBytes());
            }
        };

        /** Kernel for a multi-dimensional pitched parallel memcpy
         *
         * A row, the last dimension of the copy, is contiguous in memory but consecutive rows are separated by the
         * pitch of the buffer. The work is distributed over the gap free byte space of the copy,
         * numRows * rowBytes, therefore the byte range of a thread can begin and end in the middle of a row.
         */
        struct ParallelMemcpyPitchedKernel
        {
            /** Execute a multi-dimensional pitched parallel memcpy
             *
             * @tparam T_AccInterface Accelerator interface type
             * @param acc Accelerator interface providing thread/block information
             * @param srcPtr Source buffer pointer
             * @param destPtr Destination buffer pointer
             * @param srcPitchBytes Source pitches in bytes, without the last dimension
             * @param destPitchBytes Destination pitches in bytes, without the last dimension
             * @param rowExtents Extents of the copy without the last dimension, its product is the number of rows
             * @param rowBytes Number of bytes of a single row
             *
             * Example for a 3D [10][20][30] copy of 4 byte elements:
             * - rowExtents = [10][20], 200 rows
             * - rowBytes = 30 * 4 = 120, 24000 bytes in total
             * - with 100 threads each thread copies 240 bytes, spanning two rows
             */
            template<alpaka::onAcc::concepts::Acc<ALPAKA_TYPEOF(api::host)> T_AccInterface>
            void operator()(
                T_AccInterface const& acc,
                void const* srcPtr,
                void* destPtr,
                alpaka::concepts::Vector auto const& srcPitchBytes,
                alpaka::concepts::Vector auto const& destPitchBytes,
                alpaka::concepts::Vector auto const& rowExtents,
                size_t rowBytes) const
            {
                // the grid size is the number of chunks, a frame specification would decouple both
                static_assert(
                    !T_AccInterface::launchedWithFrameSpec(),
                    "The parallel memcpy kernels must be enqueued with a thread specification.");
                static_assert(
                    ALPAKA_TYPEOF(srcPitchBytes)::dim() == ALPAKA_TYPEOF(rowExtents)::dim()
                        && ALPAKA_TYPEOF(destPitchBytes)::dim() == ALPAKA_TYPEOF(rowExtents)::dim(),
                    "The pitches must be given without the last dimension, see Vec::eraseBack().");

                // the row index below divides by the number of bytes of a row
                ALPAKA_ASSERT_ACC(rowBytes > size_t{0u});

                auto const byteRange = makeThreadByteRange(acc, rowExtents.product() * rowBytes);
                if(byteRange.isEmpty())
                    return;

                auto* const destBytes = static_cast<std::uint8_t*>(destPtr);
                auto const* const srcBytes = static_cast<std::uint8_t const*>(srcPtr);

                // the range is not row aligned, the first and the last row can be partial
                for(size_t byteIdx = byteRange.begin; byteIdx < byteRange.end;)
                {
                    size_t const rowIdx = byteIdx / rowBytes;
                    size_t const offsetInRow = byteIdx % rowBytes;
                    size_t const numBytesInRow = std::min(rowBytes - offsetInRow, byteRange.end - byteIdx);

                    alpaka::concepts::Vector<size_t> auto const rowIndices = alpaka::mapToND(rowExtents, rowIdx);

                    std::memcpy(
                        destBytes + (rowIndices * destPitchBytes).sum() + offsetInRow,
                        srcBytes + (rowIndices * srcPitchBytes).sum() + offsetInRow,
                        numBytesInRow);

                    byteIdx += numBytesInRow;
                }
            }
        };

    } // namespace internal::detail
} // namespace alpaka::onHost
