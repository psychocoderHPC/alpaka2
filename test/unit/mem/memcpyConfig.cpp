/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */


#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

/** @file test if memcpyConfig environment variable is working
 * Test:
 * ALPAKA_MEMCPY_MODE
 * ALPAKA_MEMCPY_NUM_CORES
 * ALPAKA_MEMCPY_MIN_SIZE
 */

using namespace alpaka;
using DeviceSpecs = std::decay_t<decltype(onHost::getDeviceSpecsFor(onHost::enabledApis))>;

TEMPLATE_LIST_TEST_CASE("memcopy test", "", DeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    SECTION("envVar memcpy Mode")
    {
        setenv("ALPAKA_MEMCPY_MODE", "parallel", 1);
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        if(onHost::config::hasMemcpyConfig<ALPAKA_TYPEOF(*queue.get())>)
        {
            auto& configMemcpy = queue.get()->getMemcpyConfig();
            REQUIRE(configMemcpy.getMode() == onHost::config::MemcpyMode::parallel);
        }

        setenv("ALPAKA_MEMCPY_MODE", "sequential", 1);
        queue = device.makeQueue(queueKind::blocking);

        if(onHost::config::hasMemcpyConfig<ALPAKA_TYPEOF(*queue.get())>)
        {
            auto& configMemcpy = queue.get()->getMemcpyConfig();
            REQUIRE(configMemcpy.getMode() == onHost::config::MemcpyMode::sequential);
        }
    }

    SECTION("envVar memcpy numCores")
    {
        setenv("ALPAKA_MEMCPY_NUM_CORES", "123", 1);
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        if(onHost::config::hasMemcpyConfig<ALPAKA_TYPEOF(*queue.get())>)
        {
            auto& configMemcpy = queue.get()->getMemcpyConfig();
            REQUIRE(configMemcpy.getNumCores() == 123);
        }
    }

    SECTION("envVar memcpy numCores")
    {
        setenv("ALPAKA_MEMCPY_MIN_SIZE", "789", 1);
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        if(onHost::config::hasMemcpyConfig<ALPAKA_TYPEOF(*queue.get())>)
        {
            auto& configMemcpy = queue.get()->getMemcpyConfig();
            REQUIRE(configMemcpy.getMinSizeForParallel() == 789);
        }
    }
}
