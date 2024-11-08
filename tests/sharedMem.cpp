/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
#if 1
#    include <alpaka/alpaka.hpp>
#    include <alpaka/example/executeForEach.hpp>

#    include <catch2/catch_test_macros.hpp>

#    include <chrono>
#    include <functional>
#    include <iostream>
#    include <thread>

using namespace alpaka;

template<uint32_t T_blockSize>
struct SharedBlockIotaKernel
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numBlocks) const
    {
        // auto& shared = acc[layer::shared].template allocVar<uint32_t[T_blockSize]>();
        // auto& shared = acc.template allocVar<uint32_t[T_blockSize]>();
        auto& shared = declareSharedVar<uint32_t[T_blockSize]>(acc);

        for(auto blockIdx : DataBlockIter{acc, numBlocks})
        {
            auto const numDataElemInBlock = acc[frame::extent];
            auto blockOffset = blockIdx * numDataElemInBlock;
            for(auto inBlockOffset : DataFrameIter{acc})
            {
                uint32_t id = (T_blockSize - 1u - inBlockOffset).x();
                shared[id] = id;
            }
            // acc.sync();
            syncBlockThreads(acc);
            for(auto inBlockOffset : DataFrameIter{acc})
            {
                out[blockOffset + inBlockOffset] = (blockOffset + shared[inBlockOffset.x()]).x();
            }
        }
    }
};

void runSharedBlockIota(auto mapping, auto device)
{
    Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{2u};
    constexpr Vec blockExtent = Vec{128u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    std::cout << "block shared iota exec=" << core::demangledName(mapping) << std::endl;
    auto dBuff = alpaka::alloc<uint32_t>(device, dataExtent);

    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    auto hBuff = alpaka::alloc<uint32_t>(cpuDevice, dataExtent);
    wait(queue);

    alpaka::enqueue(
        queue,
        mapping,
        alpaka::DataBlocking{numBlocks / 2u, blockExtent},
        KernelBundle{SharedBlockIotaKernel<blockExtent.x()>{}, dBuff.getMdSpan(), numBlocks.x()});
    alpaka::memcpy(queue, hBuff, dBuff);
    wait(queue);

    auto* ptr = alpaka::data(hBuff);
    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}

TEST_CASE("block shared iota", "")
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
            executeForEachNoReturn([&](auto mapping) { runSharedBlockIota(mapping, device); }, possibleMappings);
        },
        enabledApis);
}

#endif
