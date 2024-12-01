/* Copyright 2024 Andrea Bocci
* SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "config.h"

int main() {
  // the host platform always has a single device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);

  std::cout << "Host platform: " << alpaka::core::demangled<HostPlatform> << '\n';
  std::cout << "Found 1 device:\n";
  std::cout << "  - " << alpaka::getName(host) << "\n\n";

  // get all the devices on the accelerator platform
  Platform platform;
  std::vector<Device> devices = alpaka::getDevs(platform);

  std::cout << "Accelerator platform: " << alpaka::core::demangled<Platform> << '\n';
  std::cout << "Found " << devices.size() << " device(s):\n";
  for (auto const& device : devices)
    std::cout << "  - " << alpaka::getName(device) << '\n';
  std::cout << '\n';
}
