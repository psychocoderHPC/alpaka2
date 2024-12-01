/* Copyright 2024 Andrea Bocci
* SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "config.h"

int main() {
  // initialise the accelerator platform
  Platform platform;

  // require at least one device
  std::size_t n = alpaka::getDevCount(platform);
  if (n == 0) {
    exit(EXIT_FAILURE);
  }

  // use the single host device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);
  std::cout << "Host:   " << alpaka::getName(host) << '\n';

  // allocate a buffer of floats in host memory, mapped to be efficiently copied to/from the device
  uint32_t size = 42;
  auto host_buffer = alpaka::allocMappedBuf<float, uint32_t>(host, platform, size);
  std::cout << "pinned host memory buffer at " << std::data(host_buffer) << "\n\n";

  // fill the host buffers with values
  for (uint32_t i = 0; i < size; ++i) {
    host_buffer[i] = i;
  }

  // use the first device
  Device device = alpaka::getDevByIdx(platform, 0u);
  std::cout << "Device: " << alpaka::getName(device) << '\n';

  // create a work queue
  Queue queue{device};

  {
    // allocate a buffer of floats in global device memory, asynchronously
    auto device_buffer = alpaka::allocAsyncBuf<float, uint32_t>(queue, size);
    std::cout << "memory buffer on " << alpaka::getName(alpaka::getDev(device_buffer))
              << " at " << std::data(device_buffer) << "\n\n";

    // set the device memory to all zeros (byte-wise, not element-wise)
    alpaka::memset(queue, device_buffer, 0x00);

    // copy the contents of the device buffer to the host buffer
    alpaka::memcpy(queue, host_buffer, device_buffer);

    // the device buffer goes out of scope, but the memory is freed only
    // once all enqueued operations have completed
  }

  // wait for all operations to complete
  alpaka::wait(queue);

  // read the content of the host buffer
  for (uint32_t i = 0; i < size; ++i) {
    std::cout << host_buffer[i] << ' ';
  }
  std::cout << '\n';

  std::cout << "All work has completed\n";
}
