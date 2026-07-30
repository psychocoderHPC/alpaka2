/* Copyright 2026 Alexis Laplanche (SiPearl)
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

TEMPLATE_LIST_TEST_CASE("tutorial parallel memcpy", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict());
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    // BEGIN-TUTORIAL-parallelMemcpy
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    if constexpr(onHost::config::hasMemcpyConfig<ALPAKA_TYPEOF(*queue.get())>)
    {
        auto& configMemcpy = queue.get()->getMemcpyConfig();
        configMemcpy.setMode(onHost::config::MemcpyMode::parallel);
        configMemcpy.setNumCores(0); // Use all available core
        configMemcpy.setMinSizeForParallel(512 * 1024); // 512KB
        std::cout << "Queue memcpy configuration : " << configMemcpy.getConfigString() << std::endl;
    }
    // END-TUTORIAL-parallelMemcpy

    constexpr std::size_t NumElements = 1u << 20; // 1,048,576 elements

    // Host data: fill with a constant value
    std::vector<int> hostIn(NumElements);
    std::vector<int> hostOut(NumElements, -1); // initialize with a sentinel
    int const fillValue = 42;
    std::fill(hostIn.begin(), hostIn.end(), fillValue);

    // Device allocation matching the host container
    auto devBuffer = onHost::allocLike(device, hostIn);

    // Copy host -> device
    onHost::memcpy(queue, devBuffer, hostIn);
    onHost::wait(queue);

    // Ensure hostOut is different before the round-trip
    std::fill(hostOut.begin(), hostOut.end(), -1);

    // Copy device -> host
    onHost::memcpy(queue, hostOut, devBuffer);
    onHost::wait(queue);

    // Check some values (begin, middle, end)
    CHECK(hostOut[0] == fillValue);
    CHECK(hostOut[NumElements / 2] == fillValue);
    CHECK(hostOut[NumElements - 1] == fillValue);
}
