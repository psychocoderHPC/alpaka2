/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cstdio>
#include <random>

struct VectorAddKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, uint32_t size) const
    {
        for(auto [index] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, Vec1D size) const
    {
        for(auto ndindex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[ndindex] = in1[ndindex] + in2[ndindex];
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, Vec3D size) const
    {
        for(auto ndindex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[ndindex] = in1[ndindex] + in2[ndindex];
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

    // launch the 1-dimensional kernel with scalar size
    auto frameSpec = alpaka::onHost::FrameSpec{32u, 32u};

    std::cout << "Testing VectorAddKernel with scalar indices with a grid of " << frameSpec << "\n";
    queue.enqueue(
        computeExec,
        frameSpec,
        VectorAddKernel{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        size);

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        float sum = in1_h[i] + in2_h[i];
        verify(out_h[i] < sum + epsilon);
        verify(out_h[i] > sum - epsilon);
    }
    std::cout << "success\n";

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);


    std::cout << "Testing VectorAddKernel with vector indices with a grid of " << frameSpec << "\n";
    queue.enqueue(
        computeExec,
        frameSpec,
        VectorAddKernel1D{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        Vec1D{size});

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        float sum = in1_h[i] + in2_h[i];
        verify(out_h[i] < sum + epsilon);
        verify(out_h[i] > sum - epsilon);
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
    constexpr Vec3D ndsize = {50, 125, 16};
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
    auto frameSpec = alpaka::onHost::FrameSpec{Vec3D{5, 5, 1}, Vec3D{4, 4, 4}};
    std::cout << "Testing VectorAddKernel3D with vector indices with a grid of " << frameSpec << "\n";

    queue.enqueue(
        computeExec,
        frameSpec,
        VectorAddKernel3D{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        ndsize);

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        float sum = in1_h[lIdx] + in2_h[lIdx];
        verify(out_h[lIdx] < sum + epsilon);
        verify(out_h[lIdx] > sum - epsilon);
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
