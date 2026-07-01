/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;

using TestApis = ALPAKA_TYPEOF(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));

struct GetExecutorKernel
{
    template<onAcc::concepts::Acc T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto result) const
    {
        auto executor = alpaka::getExecutor(acc);

        static_assert(concepts::Executor<ALPAKA_TYPEOF(executor)>);
        static_assert(std::is_same_v<ALPAKA_TYPEOF(executor), ALPAKA_TYPEOF(acc.getExecutor())>);

        result[0] = executor == acc.getExecutor();
    }
};

TEMPLATE_LIST_TEST_CASE("get executor", "[interface][executor]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto executor = alpaka::getExecutor(cfg);

    static_assert(concepts::HasExecutor<ALPAKA_TYPEOF(cfg)>);
    static_assert(concepts::Executor<ALPAKA_TYPEOF(executor)>);
    CHECK((executor == cfg[object::exec]));

    onHost::Device device = test::getDeviceOrSkipTest(cfg);
    onHost::Queue queue = device.makeQueue();

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});
    auto hBuff = onHost::allocHostLike(dBuff);

    queue.enqueue(onHost::FrameSpec{Vec{1u}, Vec{1u}, executor}, KernelBundle{GetExecutorKernel{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    CHECK(hBuff[0]);
}

TEMPLATE_LIST_TEST_CASE("get device kind", "[interface][deviceKind]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceKind = alpaka::getDeviceKind(cfg);

    static_assert(concepts::DeviceKind<ALPAKA_TYPEOF(deviceKind)>);
    CHECK((deviceKind == cfg[object::deviceSpec].getDeviceKind()));

    onHost::Device device = test::getDeviceOrSkipTest(cfg);
    onHost::Queue queue = device.makeQueue();

    auto deviceDeviceKind = alpaka::getDeviceKind(device);
    auto queueDeviceKind = alpaka::getDeviceKind(queue);

    static_assert(concepts::DeviceKind<ALPAKA_TYPEOF(deviceDeviceKind)>);
    static_assert(concepts::DeviceKind<ALPAKA_TYPEOF(queueDeviceKind)>);
    static_assert(std::is_same_v<ALPAKA_TYPEOF(deviceDeviceKind), ALPAKA_TYPEOF(device.getDeviceKind())>);
    static_assert(std::is_same_v<ALPAKA_TYPEOF(queueDeviceKind), ALPAKA_TYPEOF(device.getDeviceKind())>);

    CHECK((deviceDeviceKind == device.getDeviceKind()));
    CHECK((queueDeviceKind == device.getDeviceKind()));
}
