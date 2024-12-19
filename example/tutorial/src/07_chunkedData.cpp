/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstdio>
#include <random>

/** @file
 *
 * In the previous example we showed how to handle thread indices by hand to iterate over 1 and 3-dimensional data.
 * There are very seldom cases where you need this explict control over threads and blocks. Very often handling thread
 * indices by hand will result in performance issues at least on CPU devices.
 *
 * This example will show how you can iterate with frames, which can be seen as data index chunks without explicit
 * calculate thread indices by hand. The code is is easy to write and read and will mostly be faster on CPU and GPU
 * devices.
 */

struct VectorAddKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out) const
    {
        // product() returns a scalar therefore we need the explicit Vec1D type
        Vec1D linearNumFrames = acc[alpaka::frame::count].product();
        auto frameExtent = acc[alpaka::frame::extent];
        Vec1D linearFrameExtent = frameExtent.product();

        /* We will use a 1-dimensional shared memory array to load the data from in1 for the corresponding frame into
         * it and use the cached data to compute the final result. Take care, shared memory is never initialized and
         * will contain garbage after the creation.
         *
         * The extents of a shared memory array must be known at compile time, therefore it is required that the frame
         * extent on the host side is of the type CVec.
         */
        auto sharedIn1Data = alpaka::onAcc::declareSharedMdArray<float>(acc, frameExtent);

        /* This kernel is called with 1- dimensional frame extents, nevertheless we will linearize the indices
         * explicitly which allow calling this kernel with M-dimensional frames extents too.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more frames.
         */
        for(auto linearFrameIdx : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::linearBlocksInGrid,
                alpaka::IdxRange{linearNumFrames}))
        {
            // iterate over elements within the frame and use all threads from the thread block
            for(auto linearFrameElem : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{linearFrameExtent}))
            {
                auto const globalDataIdx = linearFrameIdx * frameExtent + linearFrameElem;
                sharedIn1Data[linearFrameElem] = in1[globalDataIdx];
            }

            /* This call is equal to `alpaka::onAcc::syncBlockThreads(acc)`
             *
             * The synchronization is required because we will use for the second loop over frame element indicis a
             * different traversing schema. Therefore, you should not assume any thread and data element relation.
             */
            acc.syncBlockThreads();

            for(auto linearFrameElem : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{linearFrameExtent},
                    alpaka::onAcc::traverse::tiled))
            {
                auto const globalDataIdx = linearFrameIdx * frameExtent + linearFrameElem;
                out[globalDataIdx] = sharedIn1Data[linearFrameElem] + in2[globalDataIdx];
            }

            /* Synchronization is required if the same thread block is calculating more than one frame.
             * If this synchronization is missing threads can begin already to fill the shared memory with the next
             * frame data which results into a data race.
             */
            acc.syncBlockThreads();
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out) const
    {
        auto numFramesMD = acc[alpaka::frame::count];
        auto frameExtentMD = acc[alpaka::frame::extent];

        /* We will use a 3-dimensional shared memory array to load the data from in1 for the corresponding frame into
         * it and use the cached data to compute the final result. Take care, shared memory is never initialized and
         * will contain garbage after the creation.
         *
         * The extents of a shared memory array must be known at compile time, therefore it is required that the frame
         * extent on the host side is of the type CVec.
         */
        auto sharedIn1Data = alpaka::onAcc::declareSharedMdArray<float>(acc, frameExtentMD);

        /* This kernel is called with 1- dimensional frame extents, nevertheless we will linearize the indices
         * explicitly which allow calling this kernel with M-dimensional frames extents too.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more frames.
         */
        for(auto frameIdxMD :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::blocksInGrid, alpaka::IdxRange{numFramesMD}))
        {
            // iterate over elements within the frame and use all threads from the thread block
            for(auto frameElemIdxMD :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{frameExtentMD}))
            {
                auto const globalDataIdxMD = frameIdxMD * frameExtentMD + frameElemIdxMD;
                sharedIn1Data[frameElemIdxMD] = in1[globalDataIdxMD];
            }

            /* This call is equal to `alpaka::onAcc::syncBlockThreads(acc)`
             *
             * The synchronization is required because we will use for the second loop over frame element indicis a
             * different traversing schema. Therefore, you should not assume any thread and data element relation.
             */
            acc.syncBlockThreads();

            for(auto frameElemIdxMD : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{frameExtentMD},
                    alpaka::onAcc::traverse::tiled))
            {
                auto const globalDataIdxMD = frameIdxMD * frameExtentMD + frameElemIdxMD;
                out[globalDataIdxMD] = sharedIn1Data[frameElemIdxMD] + in2[globalDataIdxMD];
            }

            /* Synchronization is required if the same thread block is calculating more than one frame.
             * If this synchronization is missing threads can begin already to fill the shared memory with the next
             * frame data which results into a data race.
             */
            acc.syncBlockThreads();
        }
    }
};

void testVectorAddKernel(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // tolerance
    constexpr float epsilon = 0.000001f;

    // buffer size
    constexpr uint32_t size = 1024 * 1024;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::alloc<float>(host, size);
    auto in2_h = alpaka::onHost::allocMirror(host, in1_h);
    auto out_h = alpaka::onHost::allocMirror(host, in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    // launch the 1-dimensional kernel
    constexpr auto frameExtent = 32u;
    auto numFrames = size / frameExtent;
    // The kernel assumes that the problem size is a multiple of the frame size.
    assert((numFrames * frameExtent) == size);

    auto frameSpec = alpaka::onHost::FrameSpec{numFrames, alpaka::CVec<uint32_t, frameExtent>{}};

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    std::cout << "Testing VectorAddKernel with vector indices with a grid of " << frameSpec << "\n";
    queue
        .enqueue(computeExec, frameSpec, VectorAddKernel1D{}, in1_d.getMdSpan(), in2_d.getMdSpan(), out_d.getMdSpan());

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        float sum = in1_h[i] + in2_h[i];
        assert(out_h[i] < sum + epsilon);
        assert(out_h[i] > sum - epsilon);
    }
    std::cout << "success\n";
}

void testVectorAddKernel3D(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // 3-dimensional and linearised buffer size
    constexpr Vec3D ndsize = {64, 128, 16};
    constexpr uint32_t size = ndsize.product();

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::alloc<float>(host, ndsize);
    auto in2_h = alpaka::onHost::allocMirror(host, in1_h);
    auto out_h = alpaka::onHost::allocMirror(host, in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        in1_h[lIdx] = dist(rand);
        in2_h[lIdx] = dist(rand);
        out_h[lIdx] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);


    // launch the 3-dimensional kernel
    auto frameExtent = alpaka::CVec<uint32_t, 4, 4, 2>{};
    auto numFrames = ndsize / frameExtent;
    // The kernel assumes that the problem size is a multiple of the frame size.
    assert((numFrames * frameExtent).product() == size);

    auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};

    std::cout << "Testing VectorAddKernel3D with vector indices with a grid of " << frameSpec << "\n";

    queue
        .enqueue(computeExec, frameSpec, VectorAddKernel3D{}, in1_d.getMdSpan(), in2_d.getMdSpan(), out_d.getMdSpan());

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        float sum = in1_h[lIdx] + in2_h[lIdx];
        assert(out_h[lIdx] < sum + epsilon);
        assert(out_h[lIdx] > sum - epsilon);
    }
    std::cout << "success\n";
}

int example(auto const cfg)
{
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];

    // initialise the accelerator platform
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    // require at least one device
    std::size_t n = alpaka::onHost::getDeviceCount(platform);

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use the single host device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);
    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // use the first device
    alpaka::onHost::Device device = platform.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    testVectorAddKernel(host, device, computeExec);
    testVectorAddKernel3D(host, device, computeExec);

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
