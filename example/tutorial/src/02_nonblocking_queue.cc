/* Copyright 2024 Andrea Bocci
* SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>
#include <thread>

#include <alpaka/alpaka.hpp>

#include "config.h"

int main() {
  // the host platform always has a single device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);

  std::cout << "Host platform: " << alpaka::core::demangled<HostPlatform> << '\n';
  std::cout << "Found 1 device:\n";
  std::cout << "  - " << alpaka::getName(host) << "\n\n";

  // create a non-blocking host queue and submit some work to it
  alpaka::Queue<Host, alpaka::NonBlocking> queue{host};

  std::cout << "Enqueue some work\n";
  alpaka::enqueue(queue, []() noexcept {
    std::cout << "  - host task running...\n";
    std::this_thread::sleep_for(std::chrono::seconds(5u));
    std::cout << "  - host task complete\n";
  });

  // wait for the work to complete
  std::cout << "Wait for the enqueue work to complete...\n";
  alpaka::wait(queue);
  std::cout << "All work has completed\n";
}
