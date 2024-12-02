/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <cstdint>

// index type
using Idx = uint32_t;
// vectors
template<uint32_t T_dim>
using Vec = alpaka::Vec<uint32_t, T_dim>;
// zero dimension aka scalar is currently not supported
// using Scalar = Vec<Dim0D>;
using Vec1D = Vec<1u>;
using Vec2D = Vec<2u>;
using Vec3D = Vec<3u>;

// remove NDEBUG to activate asserts
#ifdef NDEBUG
#    undef NDEBUG
#endif
