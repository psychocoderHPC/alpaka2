/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int example(auto const apit)
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
#if 0
    // host task enqueue is currently not implemented
    alpaka::enqueue(
        queue,
        []() noexcept
        {
            std::cout << "  - host task running...\n";
            std::this_thread::sleep_for(std::chrono::seconds(5u));
            std::cout << "  - host task complete\n";
        });
#endif
    // wait for the work to complete
    std::cout << "Wait for the enqueue work to complete...\n";
    alpaka::onHost::wait(queue);
    std::cout << "All work has completed\n";

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach([=](auto const& tag) { return example(tag); }, onHost::enabledApis);
}
