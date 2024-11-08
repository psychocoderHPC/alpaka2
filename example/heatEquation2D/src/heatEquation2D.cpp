/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "StencilKernel.hpp"
#include "analyticalSolution.hpp"

#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
//! selected accelerator only. If you use the example as the starting point for your project, you can rename the
//! example() function to main() and move the accelerator tag to the function body.
template<typename T_Cfg>
auto example(T_Cfg const& cfg) -> int
{
    using namespace alpaka;

    using Idx = uint32_t;
    using IdxVec = alpaka::Vec<Idx, 2u>;

    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << api.getName() << std::endl;

    // Select specific devices
    Platform platformHost = makePlatform(api::cpu);
    Device devHost = platformHost.makeDevice(0);

    Platform platformAcc = makePlatform(api);
    Device devAcc = platformAcc.makeDevice(0);

    // simulation defines
    // {Y, X}
    constexpr IdxVec numNodes{64, 64};
    constexpr IdxVec haloSize{2, 2};
    constexpr IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000;
    constexpr double tMax = 0.1;

    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    double r = 2 * dt / ((dx * dx * dy * dy) / (dx * dx + dy * dy));
    if(r > 1.)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Initialize host-buffer
    // This buffer will hold the current values (used for the next step)
    auto uBufHost = alpaka::alloc<double>(devHost, extent);

    // Accelerator buffer
    auto uCurrBufAcc = alpaka::alloc<double>(devAcc, extent);
    auto uNextBufAcc = alpaka::alloc<double>(devAcc, extent);

    auto const pitchCurrAcc{uCurrBufAcc.getPitches()};
    auto const pitchNextAcc{uNextBufAcc.getPitches()};

    // Set buffer to initial conditions
    initalizeBuffer(uBufHost, dx, dy);

    // Select queue
    Queue dumpQueue = devAcc.makeQueue();
    Queue computeQueue = devAcc.makeQueue();

    // Copy host -> device
    alpaka::memcpy(computeQueue, uCurrBufAcc, uBufHost);
    alpaka::wait(computeQueue);

    // Appropriate chunk size to split your problem for your Acc
    constexpr Idx xSize = 16u;
    constexpr Idx ySize = 16u;
    constexpr Idx halo = 2u;
    constexpr IdxVec chunkSize{ySize, xSize};
    constexpr auto sharedMemSize = (ySize + halo) * (xSize + halo);

    constexpr IdxVec numChunks{
        alpaka::core::divCeil(numNodes[0], chunkSize[0]),
        alpaka::core::divCeil(numNodes[1], chunkSize[1]),
    };

    assert(
        numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
        && "Domain must be divisible by chunk size");

    StencilKernel<sharedMemSize> stencilKernel;
    BoundaryKernel boundaryKernel;

    auto dataBlocking = alpaka::DataBlocking{numChunks, chunkSize};

    // Simulate
    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        alpaka::enqueue(
            computeQueue,
            exec,
            dataBlocking,
            KernelBundle{
                stencilKernel,
                uCurrBufAcc.data(),
                uNextBufAcc.data(),
                chunkSize,
                pitchCurrAcc,
                pitchNextAcc,
                dx,
                dy,
                dt});
        // Apply boundaries
        alpaka::enqueue(
            computeQueue,
            exec,
            dataBlocking,
            KernelBundle{boundaryKernel, uNextBufAcc.data(), chunkSize, pitchNextAcc, step, dx, dy, dt});

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0)
        {
            alpaka::wait(computeQueue);
            alpaka::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
            alpaka::wait(dumpQueue);
            writeImage(step - 1, uBufHost);
        }
#endif

        // So we just swap next and curr (shallow copy)
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    // Copy device -> host
    alpaka::wait(computeQueue);
    alpaka::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    alpaka::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost, extent, dx, dy, tMax);

    if(resultIsCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return executeForEach([=](auto const& tag) { return example(tag); }, allExecutorsAndApis(enabledApis));
}
