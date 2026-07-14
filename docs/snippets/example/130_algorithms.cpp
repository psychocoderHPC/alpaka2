/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */


#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <functional>

using namespace alpaka;

// BEGIN-TUTORIAL-transformFunctor
struct SquareValue
{
    ALPAKA_FN_ACC auto operator()(int const& value) const -> int
    {
        return value * value;
    }
};

// END-TUTORIAL-transformFunctor

// BEGIN-TUTORIAL-transformReduceFunctor
struct MultiplyValues
{
    ALPAKA_FN_ACC auto operator()(concepts::Simd auto const& a, concepts::Simd auto const& b) const
    {
        return a * b;
    }
};

// END-TUTORIAL-transformReduceFunctor

// BEGIN-TUTORIAL-generatorFunctor
struct AddLinearIdx
{
    ALPAKA_FN_ACC auto operator()(int const& value, size_t const& linearIdx) const -> int
    {
        return value + static_cast<int>(linearIdx);
    }
};

// END-TUTORIAL-generatorFunctor

// BEGIN-TUTORIAL-concurrentFunctor
struct AddInPlace
{
    ALPAKA_FN_ACC void operator()(concepts::SimdPtr auto a, concepts::SimdPtr auto const& b) const
    {
        a = a.load() + b.load();
    }
};

// END-TUTORIAL-concurrentFunctor

// BEGIN-TUTORIAL-stencilFunctor
struct CrossStencil
{
    ALPAKA_FN_ACC auto operator()(concepts::SimdPtr auto const& in) const
    {
        using SimdPtrType = ALPAKA_TYPEOF(in);
        using VecIdxType = typename SimdPtrType::IdxType;

        concepts::Simd auto result = in.load();

        using SimdType = ALPAKA_TYPEOF(result);
        concepts::Simd auto const value = SimdType::fill(42);
        result += value;

        for(uint32_t d = 0u; d < VecIdxType::dim(); ++d)
        {
            concepts::Vector auto negative = VecIdxType::fill(0);
            concepts::Vector auto positive = VecIdxType::fill(0);
            negative[d] = -1;
            positive[d] = 1;

            result += in[negative].load() * 3;
            result += in[positive].load() * 5;
        }

        return result;
    }
};

// END-TUTORIAL-stencilFunctor

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    std::array<int, 8u> hostInput{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<int, 8u> hostIota{};
    std::array<int, 8u> hostTransform{};
    std::array<int, 8u> hostScan{};
    std::array<int, 8u> hostGenerator{};
    std::array<int, 8u> hostConcurrent{};


    auto inputBuffer = onHost::alloc<int>(device, onHost::getExtents(hostInput));
    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-iota
    auto iotaBuffer = onHost::alloc<int>(device, onHost::getExtents(hostInput));
    onHost::iota<int>(queue, exec, 10, iotaBuffer);
    // END-TUTORIAL-iota

    // BEGIN-TUTORIAL-transformCall
    auto transformBuffer = onHost::alloc<int>(device, onHost::getExtents(hostInput));
    onHost::transform(queue, exec, transformBuffer, ScalarFunc{SquareValue{}}, inputBuffer);
    // END-TUTORIAL-transformCall

    // BEGIN-TUTORIAL-transformStencilCall
    // Stencil operations need proper halo setup; see unit tests for a complete example.
    // END-TUTORIAL-transformStencilCall

    // BEGIN-TUTORIAL-reduce
    auto reduceBuffer = onHost::alloc<int>(device, Vec{1u});
    onHost::reduce(queue, exec, 0, reduceBuffer, std::plus{}, inputBuffer);
    // END-TUTORIAL-reduce
    auto reduceHost = onHost::allocHostLike(reduceBuffer);

    // BEGIN-TUTORIAL-scan
    auto scanBuffer = onHost::alloc<int>(device, onHost::getExtents(hostInput));
    auto tmpBuffer = onHost::alloc<std::byte>(device, onHost::getScanBufferSize<int>(inputBuffer.getExtents()));
    onHost::inclusiveScan(queue, exec, tmpBuffer, scanBuffer, inputBuffer);
    // END-TUTORIAL-scan

    // BEGIN-TUTORIAL-transformReduceCall
    auto transformReduceBuffer = onHost::alloc<int>(device, Vec{1u});
    onHost::transformReduce(
        queue,
        exec,
        0,
        transformReduceBuffer,
        std::plus{},
        MultiplyValues{},
        inputBuffer,
        inputBuffer);
    // END-TUTORIAL-transformReduceCall
    auto transformReduceHost = onHost::allocHostLike(transformReduceBuffer);

    // BEGIN-TUTORIAL-transformReduceStencilCall
    // Stencil operations need proper halo setup; see unit tests for a complete example.
    // END-TUTORIAL-transformReduceStencilCall

    // BEGIN-TUTORIAL-transformReduceScalarCall
    // ScalarFunc in transformReduce requires matching number of inputs; see unit tests for examples.
    // END-TUTORIAL-transformReduceScalarCall

    // BEGIN-TUTORIAL-generatorCall
    auto generator = LinearizedIdxGenerator{inputBuffer.getExtents()};
    auto generatorBuffer = onHost::alloc<int>(device, onHost::getExtents(hostInput));
    onHost::transform(queue, exec, generatorBuffer, ScalarFunc{AddLinearIdx{}}, inputBuffer, generator);
    // END-TUTORIAL-generatorCall

    // BEGIN-TUTORIAL-concurrentCall
    auto concurrentBuffer = onHost::alloc<int>(device, onHost::getExtents(hostInput));
    onHost::memcpy(queue, concurrentBuffer, inputBuffer);
    onHost::concurrent<int>(queue, exec, inputBuffer.getExtents(), AddInPlace{}, concurrentBuffer, inputBuffer);
    // END-TUTORIAL-concurrentCall

    // Copy data to the host instance to verify the results.
    onHost::memcpy(queue, hostIota, iotaBuffer);
    onHost::memcpy(queue, hostTransform, transformBuffer);
    onHost::memcpy(queue, hostScan, scanBuffer);
    onHost::memcpy(queue, hostGenerator, generatorBuffer);
    onHost::memcpy(queue, hostConcurrent, concurrentBuffer);
    onHost::memcpy(queue, reduceHost, reduceBuffer);
    onHost::memcpy(queue, transformReduceHost, transformReduceBuffer);
    onHost::wait(queue);

    int offset = 10;
    for(int const& v : hostIota)
        CHECK(v == offset++);

    CHECK(hostTransform[0] == 1);
    CHECK(hostTransform[1] == 4);
    CHECK(hostTransform[2] == 9);
    CHECK(hostTransform[3] == 16);
    CHECK(hostTransform[4] == 25);
    CHECK(hostTransform[5] == 36);
    CHECK(hostTransform[6] == 49);
    CHECK(hostTransform[7] == 64);

    CHECK(reduceHost[0] == 36);

    CHECK(hostScan[0] == 1);
    CHECK(hostScan[1] == 3);
    CHECK(hostScan[2] == 6);
    CHECK(hostScan[3] == 10);
    CHECK(hostScan[4] == 15);
    CHECK(hostScan[5] == 21);
    CHECK(hostScan[6] == 28);
    CHECK(hostScan[7] == 36);


    int idx = 0;
    for(int const& v : hostGenerator)
    {
        CHECK(v == (hostInput[idx] + static_cast<int>(generator[idx])));
        ++idx;
    }

    idx = 0;
    for(int const& v : hostConcurrent)
    {
        CHECK(v == hostInput[idx] * 2);
        ++idx;
    }

    CHECK(transformReduceHost[0] == 204);
}
