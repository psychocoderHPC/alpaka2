/* Copyright 2026  René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "fn.hpp"

#include <alpaka/alpaka.hpp>

#if __has_include(<thrust/transform.h>) && ALPAKA_LANG_CUDA
#    include <thrust/device_vector.h>
#    include <thrust/transform.h>

namespace vendorExample
{
    /** Cuda function overload for Transform.
     *
     * @{
     */
    constexpr void alpakaFnRegister(Transform::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>)
    {
    }

    constexpr void alpakaFnDispatch(
        Transform::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        alpaka::concepts::IMdSpan auto&& output,
        auto&& binaryOp,
        alpaka::concepts::IMdSpan auto&& input0,
        alpaka::concepts::IMdSpan auto&& input1)
    {
        std::cout << "call thrust::transform" << std::endl;
        // ensure the pointer is non const, capturing the span results into const mdspan within the const lambda
        auto outPtr = output.data();
        queue.enqueueNativeFn(
            [=](auto cudaStream)
            {
                thrust::transform(
                    thrust::cuda::par.on(cudaStream),
                    input0.data(),
                    input0.data() + input0.getExtents().x(),
                    input1.data(),
                    outPtr,
                    binaryOp);
            });
    }

    /** @} */
} // namespace vendorExample
#endif
