/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

int example(auto const deviceApi)
{
    // initialise the accelerator platform
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    // require at least one device
    std::size_t n = alpaka::onHost::getDeviceCount(platform);

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use the single host device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);

    std::cout << "Host platform: " << alpaka::onHost::getName(host_platform) << '\n';
    std::cout << "Found 1 device:\n";
    std::cout << "  - " << alpaka::onHost::getName(host) << "\n\n";

    // allocate a buffer of floats in host memory
    uint32_t size = 42;
    auto host_buffer = alpaka::onHost::alloc<float>(host, alpaka::Vec{size});
    std::cout << "host memory buffer at " << std::data(host_buffer) << "\n\n";

    // fill the host buffers with values
    for(uint32_t i = 0; i < size; ++i)
    {
        host_buffer[i] = i;
    }

    // use the first device
    alpaka::onHost::Device device = platform.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << '\n';

    // create a work queue
    alpaka::onHost::Queue queue = device.makeQueue();
    {
        // allocate a buffer of floats in global device memory
        auto device_buffer = alpaka::onHost::alloc<float>(device, Vec1D{size});
        std::cout << "memory buffer on " << alpaka::onHost::getStaticName(alpaka::onHost::getApi(device_buffer))
                  << " at " << std::data(device_buffer) << "\n\n";

         // set the device memory to all zeros (byte-wise, not element-wise)
        alpaka::onHost::memset(queue, device_buffer, 0x00);

        // create a view to the device data
        auto view = alpaka::onHost::View(device_buffer);

        // copy the contents of the device buffer to the host buffer
        alpaka::onHost::memcpy(queue, host_buffer, view);

        // the device buffer goes out of scope, but the memory is freed only
        // once all enqueued operations have completed
    }

    // wait for all operations to complete
    alpaka::onHost::wait(queue);

    // read the content of the host buffer
    for(uint32_t i = 0; i < size; ++i)
    {
        std::cout << host_buffer[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "All work has completed\n";

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach([=](auto const& tag) { return example(tag); }, onHost::enabledApis);
}
