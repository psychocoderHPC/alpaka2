/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

using namespace alpaka;

#if 0
TEST_CASE("enqueue task", "")
{
    std::cout << "enqueue" << std::endl;

    Platform platform = makePlatform(api::cpu);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    concepts::Queue auto queue = Queue{device.makeQueue()};
    constexpr auto fn = [](auto const& acc, auto x)
    {
        std::cout << "blockIdx = " << acc[layer::block].idx() << " threadIdx = " << acc[layer::thread].idx()
                  << " value = " << x << std::endl;
    };

    queue.enqueue(mapping::cpuBlockSerialThreadOne, Vec{3, 3}, Vec{1, 1}, fn, 42);
    // queue.enqueue(mapping::cpuBlockOmpThreadOne, Vec{3, 3}, Vec{1, 1}, KernelBundle{fn, 43});
    // queue.enqueue(mapping::cpuBlockOmpThreadOne, Kernel{fn}.config(Vec{3, 3}, Vec{1, 1})(23));

     enqueue(queue, mapping::cpuBlockOmpThreadOne, Vec{3, 3}, Vec{1, 1}, KernelBundle{fn, 43});
}
#endif

struct PrintIdx
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto x) const
    {
        for(auto i : IndependentGridThreadIter{acc})
        {
#if ALPAKA_LANG_CUDA && __CUDA_ARCH__
            printf(
                "blockIdx = %u threadIdx = %u globalIdx = %u\n",
                acc[layer::block].idx().x(),
                acc[layer::thread].idx().x(),
                i.x());
#else
            std::cout << "blockIdx = " << acc[layer::block].idx() << " threadIdx = " << acc[layer::thread].idx()
                      << " value = " << i << std::endl;
#endif
        }
    }
};

void runPlatformCreationTest(auto mapping, auto queue)
{
    std::cout << "mapping=" << core::demangledName(mapping) << std::endl;
    std::cout << "start enqueue" << std::endl;
    queue.enqueue(mapping, ThreadBlocking{Vec{3, 3}, Vec{1, 1}}, PrintIdx{}, 42);
    wait(queue);
    // queue.enqueue(mapping::cpuBlockOmpThreadOne, Vec{3, 3}, Vec{1, 1}, KernelBundle{fn, 43});
    // queue.enqueue(mapping::cpuBlockOmpThreadOne, Kernel{fn}.config(Vec{3, 3}, Vec{1, 1})(23));

    enqueue(queue, mapping, DataBlocking{Vec{3, 3}, Vec{2, 2}}, KernelBundle{PrintIdx{}, 43});
    wait(queue);
}

TEST_CASE("enqueue hallo idx", "")
{
    executeForEachNoReturn(
        [](auto api)
        {
            std::cout << api.getName() << std::endl;

            Platform platform = makePlatform(api);
            Device device = platform.makeDevice(0);

            std::cout << getName(platform) << " " << device.getName() << std::endl;

            Queue queue = device.makeQueue();

            std::cout << "all mappings" << std::endl;
            auto possibleMappings = supportedMappings(device);
            executeForEachNoReturn([&](auto mapping) { runPlatformCreationTest(mapping, queue); }, possibleMappings);
        },
        enabledApis);
}

struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, uint32_t outSize) const
    {
#if 0
        auto globalIdx = (acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx()).x();
        auto numThreads = (acc[layer::thread].count() * acc[layer::block].count()).x();
        for(uint32_t i = globalIdx; i < outSize; i += numThreads)
            out[i] = i;
#else
        for(auto i : IndependentDataIter{acc})
        {
            out[i.x()] = i.x();
        }
#endif
    }
};

void runIota(auto mapping, auto device)
{
    Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{128u};
    std::cout << "mapping=" << core::demangledName(mapping) << std::endl;
    auto dBuff = alpaka::alloc<uint32_t>(device, extent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = alpaka::alloc<uint32_t>(cpuDevice, extent);

    wait(queue);
    constexpr auto frameSize = 4u;
    alpaka::enqueue(
        queue,
        mapping,
        alpaka::DataBlocking{extent / frameSize, Vec{frameSize}},
        KernelBundle{IotaKernel{}, dBuff.getMdSpan(), extent.x()});
    alpaka::memcpy(queue, hBuff, dBuff);
    wait(queue);
    auto* ptr = alpaka::data(hBuff);
    for(uint32_t i = 0u; i < extent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}

TEST_CASE("iota", "")
{
    executeForEachNoReturn(
        [](auto api)
        {
            std::cout << api.getName() << std::endl;

            Platform platform = makePlatform(api);
            Device device = platform.makeDevice(0);

            std::cout << getName(platform) << " " << device.getName() << std::endl;

            std::cout << "all mappings" << std::endl;
            auto possibleMappings = supportedMappings(device);
            executeForEachNoReturn([&](auto mapping) { runIota(mapping, device); }, possibleMappings);
        },
        enabledApis);
}
