/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>

using namespace alpaka;

struct TimedKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const&) const
    {
    }
};

TEMPLATE_LIST_TEST_CASE("tutorial event timing", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict());
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);

    // BEGIN-TUTORIAL-eventTiming
    // Timing is an explicit queue capability and is disabled by default.
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto start = device.makeEvent(timing::enabled);
    auto end = device.makeEvent(timing::enabled);

    auto const frameSpec = onHost::getFrameSpec(device, alpaka::getExecutor(TestType::makeDict()), Vec{1u});

    queue.enqueue(start);
    queue.enqueue(frameSpec, KernelBundle{TimedKernel{}});
    queue.enqueue(end);

    // This waits for the end marker and returns a duration expressed in seconds.
    auto const elapsed = onHost::getElapsedTime(start, end);
    // END-TUTORIAL-eventTiming

    CHECK(elapsed >= std::chrono::duration<double>::zero());
}
