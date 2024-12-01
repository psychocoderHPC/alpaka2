/* Copyright 2024 Andrea Bocci
* SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>

#include "config.h"
#include "WorkDiv.hpp"

struct VectorAddKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in1,
                                T const* __restrict__ in2,
                                T* __restrict__ out,
                                uint32_t size) const {
    for (auto index : alpaka::uniformElements(acc, size)) {
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel1D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in1,
                                T const* __restrict__ in2,
                                T* __restrict__ out,
                                Vec1D size) const {
    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
      auto index = ndindex[0];
      out[index] = in1[index] + in2[index];
    }
  }
};

struct VectorAddKernel3D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in1,
                                T const* __restrict__ in2,
                                T* __restrict__ out,
                                Vec3D size) const {
    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
      auto index = (ndindex[0] * size[1] + ndindex[1]) * size[2] + ndindex[2];
      out[index] = in1[index] + in2[index];
    }
  }
};

void testVectorAddKernel(Host host, Platform platform, Device device) {
  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0.f, 1.f};

  // tolerance
  constexpr float epsilon = 0.000001f;

  // buffer size
  constexpr uint32_t size = 1024 * 1024;

  // allocate input and output host buffers in pinned memory accessible by the Platform devices
  auto in1_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  auto in2_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  auto out_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);

  // fill the input buffers with random data, and the output buffer with zeros
  for (uint32_t i = 0; i < size; ++i) {
    in1_h[i] = dist(rand);
    in2_h[i] = dist(rand);
    out_h[i] = 0.;
  }

  // run the test the given device
  auto queue = Queue{device};

  // allocate input and output buffers on the device
  auto in1_d = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);
  auto in2_d = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);
  auto out_d = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);

  // copy the input data to the device; the size is known from the buffer objects
  alpaka::memcpy(queue, in1_d, in1_h);
  alpaka::memcpy(queue, in2_d, in2_h);

  // fill the output buffer with zeros; the size is known from the buffer objects
  alpaka::memset(queue, out_d, 0x00);

  // launch the 1-dimensional kernel with scalar size
  auto div = makeWorkDiv<Acc1D>(32, 32);
  std::cout << "Testing VectorAddKernel with scalar indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc1D>(
      queue, div, VectorAddKernel{}, in1_d.data(), in2_d.data(), out_d.data(), size);

  // copy the results from the device to the host
  alpaka::memcpy(queue, out_h, out_d);

  // wait for all the operations to complete
  alpaka::wait(queue);

  // check the results
  for (uint32_t i = 0; i < size; ++i) {
    float sum = in1_h[i] + in2_h[i];
    assert(out_h[i] < sum + epsilon);
    assert(out_h[i] > sum - epsilon);
  }
  std::cout << "success\n";

  // reset the output buffer on the device to all zeros
  alpaka::memset(queue, out_d, 0x00);

  // launch the 1-dimensional kernel with vector size
  std::cout << "Testing VectorAddKernel1D with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc1D>(
      queue, div, VectorAddKernel1D{}, in1_d.data(), in2_d.data(), out_d.data(), size);

  // copy the results from the device to the host
  alpaka::memcpy(queue, out_h, out_d);

  // wait for all the operations to complete
  alpaka::wait(queue);

  // check the results
  for (uint32_t i = 0; i < size; ++i) {
    float sum = in1_h[i] + in2_h[i];
    assert(out_h[i] < sum + epsilon);
    assert(out_h[i] > sum - epsilon);
  }
  std::cout << "success\n";
}

void testVectorAddKernel3D(Host host, Platform platform, Device device) {
  // random number generator with a gaussian distribution
  std::random_device rd{};
  std::default_random_engine rand{rd()};
  std::normal_distribution<float> dist{0., 1.};

  // tolerance
  constexpr float epsilon = 0.000001;

  // 3-dimensional and linearised buffer size
  constexpr Vec3D ndsize = {50, 125, 16};
  constexpr uint32_t size = ndsize.prod();

  // allocate input and output host buffers in pinned memory accessible by the Platform devices
  auto in1_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  auto in2_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  auto out_h = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);

  // fill the input buffers with random data, and the output buffer with zeros
  for (uint32_t i = 0; i < size; ++i) {
    in1_h[i] = dist(rand);
    in2_h[i] = dist(rand);
    out_h[i] = 0.;
  }

  // run the test the given device
  auto queue = Queue{device};

  // allocate input and output buffers on the device
  auto in1_d = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);
  auto in2_d = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);
  auto out_d = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);

  // copy the input data to the device; the size is known from the buffer objects
  alpaka::memcpy(queue, in1_d, in1_h);
  alpaka::memcpy(queue, in2_d, in2_h);

  // fill the output buffer with zeros; the size is known from the buffer objects
  alpaka::memset(queue, out_d, 0x00);

  // launch the 3-dimensional kernel
  auto div = makeWorkDiv<Acc3D>({5, 5, 1}, {4, 4, 4});
  std::cout << "Testing VectorAddKernel3D with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc3D>(
      queue, div, VectorAddKernel3D{}, in1_d.data(), in2_d.data(), out_d.data(), ndsize);

  // copy the results from the device to the host
  alpaka::memcpy(queue, out_h, out_d);

  // wait for all the operations to complete
  alpaka::wait(queue);

  // check the results
  for (uint32_t i = 0; i < size; ++i) {
    float sum = in1_h[i] + in2_h[i];
    assert(out_h[i] < sum + epsilon);
    assert(out_h[i] > sum - epsilon);
  }
  std::cout << "success\n";
}

int main() {
  // initialise the accelerator platform
  Platform platform;

  // require at least one device
  std::uint32_t n = alpaka::getDevCount(platform);
  if (n == 0) {
    exit(EXIT_FAILURE);
  }

  // use the single host device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);
  std::cout << "Host:   " << alpaka::getName(host) << '\n';

  // use the first device
  Device device = alpaka::getDevByIdx(platform, 0u);
  std::cout << "Device: " << alpaka::getName(device) << '\n';

  testVectorAddKernel(host, platform, device);
  testVectorAddKernel3D(host, platform, device);
}
