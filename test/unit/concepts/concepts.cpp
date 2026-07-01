/* Copyright 2025 Anton Reinhard, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

using namespace alpaka;

using Data = std::size_t;
using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;
using DictWithoutDeviceSpecOrExec = Dict<DictEntry<object::Api, api::Host>>;

template<typename TIdx>
void testVector()
{
    STATIC_CHECK(concepts::Vector<Vec<TIdx, TIdx{0}>>); // only disallowed at runtime
    STATIC_CHECK(concepts::Vector<Vec<Data, TIdx{1}>>);
    STATIC_CHECK(concepts::Vector<Vec<Data, TIdx{2}>>);
    STATIC_CHECK(concepts::Vector<Vec<Data, TIdx{3}>>);
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<TIdx, TIdx{0}>>); // only disallowed at runtime
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<Data, TIdx{1}>>);
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<Data, TIdx{2}>>);
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<Data, TIdx{3}>>);

    STATIC_CHECK(concepts::Vector<alpaka::Vec<Data, TIdx{1}>, Data>);
    STATIC_CHECK_FALSE(concepts::Vector<alpaka::Vec<double, TIdx{1}>, float>);

    STATIC_CHECK(concepts::Vector<alpaka::CVec<Data, TIdx{1}, Data{0}>>);
    STATIC_CHECK(concepts::Vector<alpaka::CVec<Data, TIdx{2}, Data{0}, Data{0}>>);
    STATIC_CHECK(concepts::Vector<alpaka::CVec<Data, TIdx{3}, Data{0}, Data{0}, Data{0}>>);
    STATIC_CHECK(concepts::CVector<alpaka::CVec<Data, TIdx{1}, Data{0}>>);
    STATIC_CHECK(concepts::CVector<alpaka::CVec<Data, TIdx{2}, Data{0}, Data{0}>>);
    STATIC_CHECK(concepts::CVector<alpaka::CVec<Data, TIdx{3}, Data{0}, Data{0}, Data{0}>>);

    STATIC_CHECK_FALSE(concepts::Vector<Data[]>);
    STATIC_CHECK_FALSE(concepts::CVector<Data[]>);
    STATIC_CHECK_FALSE(concepts::Vector<Data[], Data>);
    STATIC_CHECK_FALSE(concepts::CVector<Data[], Data>);
}

TEST_CASE("vector concepts", "[concepts][vector]")
{
    testVector<std::uint16_t>();
    testVector<std::uint32_t>();
    testVector<std::uint64_t>();
    testVector<std::int16_t>();
    testVector<std::int32_t>();
    testVector<std::int64_t>();
}

TEST_CASE("frame spec concepts", "[concepts][framespec]")
{
    using namespace alpaka::onHost;

    using TestVec1 = Vec<size_t, 1>;
    using TestVec2 = Vec<size_t, 2>;
    using TestVec3 = Vec<size_t, 3>;

    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec1, TestVec1>>);
    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec2, TestVec2>, size_t>);
    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec3, TestVec3>, size_t, 3>);
    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec2, TestVec2>, size_t, 2>);

    STATIC_CHECK_FALSE(onHost::concepts::FrameSpec<FrameSpec<TestVec1, TestVec1>, int>);
    STATIC_CHECK_FALSE(onHost::concepts::FrameSpec<FrameSpec<TestVec2, TestVec2>, size_t, 3>);
    STATIC_CHECK_FALSE(onHost::concepts::FrameSpec<FrameSpec<TestVec3, TestVec3>, int, 2>);
}

TEST_CASE("thread spec concepts", "[concepts][threadspec]")
{
    using namespace alpaka::onHost;

    using TestVec1 = Vec<size_t, 1>;
    using TestVec2 = Vec<size_t, 2>;

    STATIC_CHECK(onHost::concepts::ThreadSpec<ThreadSpec<TestVec1, TestVec1>>);
    STATIC_CHECK(onHost::concepts::ThreadSpec<ThreadSpec<TestVec2, TestVec2>, size_t>);
    STATIC_CHECK(onHost::concepts::ThreadSpec<ThreadSpec<TestVec1, TestVec1>, size_t, 1>);

    STATIC_CHECK_FALSE(onHost::concepts::ThreadSpec<ThreadSpec<TestVec2, TestVec2>, int>);
    STATIC_CHECK_FALSE(onHost::concepts::ThreadSpec<ThreadSpec<TestVec1, TestVec1>, size_t, 2>);
}

TEMPLATE_LIST_TEST_CASE("backend concept", "[concepts][backend]", TestBackends)
{
    auto backend = TestType::makeDict();

    STATIC_CHECK(alpaka::concepts::Backend<TestType>);
    STATIC_CHECK_FALSE(alpaka::concepts::Backend<DictWithoutDeviceSpecOrExec>);
    STATIC_CHECK_FALSE(alpaka::concepts::Backend<decltype(backend[alpaka::object::deviceSpec])>);
    CHECK(alpaka::getApi(backend) == alpaka::getApi(backend[alpaka::object::deviceSpec]));
    CHECK(alpaka::getDeviceKind(backend) == alpaka::getDeviceKind(backend[alpaka::object::deviceSpec]));
}
