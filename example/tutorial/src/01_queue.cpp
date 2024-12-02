/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main()
{
    // the cpu api always has a single device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);

    std::cout << "Host platform: " << alpaka::onHost::getName(host_platform) << '\n';
    std::cout << "Found 1 device:\n";
    std::cout << "  - " << alpaka::onHost::getName(host) << "\n\n";

    // create a blocking host queue and submit some work to it
    alpaka::onHost::Queue queue = host.makeQueue();

    std::cout << "Enqueue some work\n";

    alpaka::onHost::enqueue(
        queue,
        []() noexcept
        {
            std::cout << "  - host task running...\n";
            std::this_thread::sleep_for(std::chrono::seconds(5u));
            std::cout << "  - host task complete\n";
        });

    // wait for the work to complete
    std::cout << "Wait for the enqueue work to complete...\n";
    alpaka::onHost::wait(queue);
    std::cout << "All work has completed\n";

    return EXIT_SUCCESS;
}
