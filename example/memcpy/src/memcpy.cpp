/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <typeinfo>

void hostVerify(
    auto const& bufHost1D,
    auto const& bufHost2D,
    auto const& bufHost3D,
    auto const& bufVerif1D,
    auto const& bufVerif2D,
    auto const& bufVerif3D,
    auto& hostVerif)
{
    auto const* p1 = std::data(bufHost1D);
    auto const* p2 = std::data(bufHost2D);
    auto const* p3 = std::data(bufHost3D);
    auto const* pRef1 = std::data(bufVerif1D);
    auto const* pRef2 = std::data(bufVerif2D);
    auto const* pRef3 = std::data(bufVerif3D);

    // Number of elements (== number of bytes, since Data == uint8_t)
    size_t const n1 = bufHost1D.getExtents().product();
    size_t const n2 = bufHost2D.getExtents().product();
    size_t const n3 = bufHost3D.getExtents().product();

    hostVerif[0] = (std::memcmp(p1, pRef1, n1 * sizeof(*p1)) == 0);
    hostVerif[1] = (std::memcmp(p2, pRef2, n2 * sizeof(*p2)) == 0);
    hostVerif[2] = (std::memcmp(p3, pRef3, n3 * sizeof(*p3)) == 0);
}

inline void fillCyclicPattern(uint8_t* dst, size_t n)
{
    if(n == 0)
        return;

    size_t filled = std::min(n, size_t{256});

    for(size_t i = 0; i < filled; ++i)
    {
        dst[i] = static_cast<uint8_t>(i);
    }

    while(filled < n)
    {
        size_t const copy = std::min(filled, n - filled);
        std::memcpy(dst + filled, dst, copy);
        filled += copy;
    }
}

class devVerify
{
public:
    //! The kernel entry point
    //!
    //! \param acc The accelerator to be executed on
    //! \param bufHost1D The 1D buffer to verify
    //! \param bufHost2D The 2D buffer to verify
    //! \param bufHost3D The 3D buffer to verify
    //! \param extent2D The extent of the 2D buffer
    //! \param extent3D The extent of the 3D buffer
    //! \param resultsBuffer Device buffer to store results (size 3)
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const buf1D,
        alpaka::concepts::IMdSpan auto const buf2D,
        alpaka::concepts::IMdSpan auto const buf3D,
        auto const& extent2D,
        auto const& extent3D,
        alpaka::concepts::IMdSpan auto resultsBuffer) const
    {
        using namespace alpaka;

        // Get buffer sizes
        uint32_t numElements = static_cast<uint32_t>(buf1D.getExtents().product());
        uint32_t numElements2D = static_cast<uint32_t>(buf2D.getExtents().product());
        uint32_t numElements3D = static_cast<uint32_t>(buf3D.getExtents().product());

        // Each thread verifies a portion of the 1D buffer
        uint32_t result1D = 1u;
        for(auto idxVec : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numElements}))
        {
            size_t i = static_cast<size_t>(idxVec[0]);
            result1D = buf1D[i] == static_cast<uint8_t>(i) ? 1u : 0u;
        }

        // Each thread verifies a portion of the 2D buffer
        uint32_t result2D = 1u;
        for(auto idxVec : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numElements2D}))
        {
            size_t i = static_cast<size_t>(idxVec[0]);
            size_t y = i / extent2D.x();
            size_t x = i % extent2D.x();

            result2D = buf2D[Vec{y, x}] == static_cast<uint8_t>(i) ? 1u : 0u;
        }

        // Each thread verifies a portion of the 3D buffer
        uint32_t result3D = 1u;
        for(auto idxVec : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numElements3D}))
        {
            size_t i = static_cast<size_t>(idxVec[0]);
            uint8_t expected = static_cast<uint8_t>(i);

            size_t x = i % extent3D.x();
            size_t y = (i / extent3D.x()) % extent3D.y();
            size_t z = i / (extent3D.x() * extent3D.y());
            result3D = buf3D[Vec{z, y, x}] == expected ? 1u : 0u;
        }

        // Use atomic operations to combine results from all threads
        onAcc::atomicAnd(acc, &resultsBuffer[0], result1D);
        onAcc::atomicAnd(acc, &resultsBuffer[1], result2D);
        onAcc::atomicAnd(acc, &resultsBuffer[2], result3D);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
auto example(auto const deviceSpec, auto const exec, size_t numElements, size_t numberOfRuns) -> int
{
    using namespace alpaka;
    using IdxVec1D = Vec<std::size_t, 1u>;
    using IdxVec2D = Vec<std::size_t, 2u>;
    using IdxVec3D = Vec<std::size_t, 3u>;

    // Define problem size
    IdxVec1D const extent1D(numElements);

    size_t exponent = std::countr_zero(numElements);
    size_t Exp = exponent / 2;
    size_t X_2d = size_t(1) << Exp;
    size_t Y_2d = size_t(1) << Exp;
    IdxVec2D const extent2D{Y_2d, X_2d};

    exponent = std::countr_zero(numElements);
    size_t xExp = exponent / 3;
    size_t yExp = (exponent - xExp) / 2;
    size_t zExp = exponent - xExp - yExp;
    size_t X_3d = size_t(1) << xExp;
    size_t Y_3d = size_t(1) << yExp;
    size_t Z_3d = size_t(1) << zExp;
    IdxVec3D const extent3D{Z_3d, Y_3d, X_3d};

    // Define the buffer element type
    using Data = uint8_t;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << onHost::demangledName<Data>() << std::endl;
    std::cout << "Number of runs: " << numberOfRuns << std::endl;

    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();
    if constexpr(onHost::config::hasMemcpyConfig<ALPAKA_TYPEOF(*queue.get())>)
    {
        auto& optConfig = queue.get()->getMemcpyConfig();
        std::cout << optConfig.getConfigString() << std::endl;
    }

    auto bufHost1D = onHost::allocHost<Data>(extent1D);
    auto bufHost2D = onHost::allocHost<Data>(extent2D);
    auto bufHost3D = onHost::allocHost<Data>(extent3D);
    auto hostVerif = onHost::allocHost<uint32_t>(Vec{3u});

    hostVerif[0] = 0u;
    hostVerif[1] = 0u;
    hostVerif[2] = 0u;

    uint8_t* p1 = bufHost1D.data();
    uint8_t* p2 = bufHost2D.data();
    uint8_t* p3 = bufHost3D.data();

    fillCyclicPattern(p1, numElements); // build the pattern once
    std::memcpy(p2, p1, numElements); // 2D buffer: identical linear layout
    std::memcpy(p3, p1, numElements); // 3D buffer: identical linear layout

    // auto bufAcc1D = onHost::allocLike(devAcc, bufHost1D);
    auto bufAcc1D = onHost::alloc<Data>(devAcc, extent1D);
    auto bufAcc2D = onHost::allocLike(devAcc, bufHost2D);
    auto bufAcc3D = onHost::allocLike(devAcc, bufHost3D);
    auto accVerif = onHost::allocLike(devAcc, hostVerif);

    double CopyRuntime1D = 0.0;
    double CopyRuntime2D = 0.0;
    double CopyRuntime3D = 0.0;

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        auto beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufAcc1D, bufHost1D);
        onHost::wait(queue);
        auto endCopyT = std::chrono::high_resolution_clock::now();
        double copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        CopyRuntime1D += copyRuntime;

        beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufAcc2D, bufHost2D);
        onHost::wait(queue);
        endCopyT = std::chrono::high_resolution_clock::now();
        copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        CopyRuntime2D += copyRuntime;

        beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufAcc3D, bufHost3D);
        onHost::wait(queue);
        endCopyT = std::chrono::high_resolution_clock::now();
        copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        CopyRuntime3D += copyRuntime;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Host to Device memcpy: \n"
              << std::setprecision(6) << "\t 1D:\t" << CopyRuntime1D << "s\t" << std::fixed << std::setprecision(2)
              << numElements * numberOfRuns / CopyRuntime1D / 1E6 << " MB/s\n"
              << std::setprecision(6) << "\t 2D:\t" << CopyRuntime2D << "s\t" << std::fixed << std::setprecision(2)
              << numElements * numberOfRuns / CopyRuntime2D / 1E6 << " MB/s\n"
              << std::setprecision(6) << "\t 3D:\t" << CopyRuntime3D << "s\t" << std::fixed << std::setprecision(2)
              << numElements * numberOfRuns / CopyRuntime3D / 1E6 << " MB/s\n"
              << std::defaultfloat << std::setprecision(6);

    Vec<size_t, 1u> chunkSize = 64u;
    auto range = onHost::FrameSpec{divCeil(extent1D, chunkSize), chunkSize, exec};

    queue.enqueue(range, devVerify{}, bufAcc1D, bufAcc2D, bufAcc3D, extent2D, extent3D, accVerif);
    onHost::wait(queue);

    onHost::memcpy(queue, hostVerif, accVerif);
    onHost::wait(queue);

    uint8_t fail = 0;
    if(!hostVerif[0])
    {
        std::cerr << "Memcpy Host to Device 1D failed!" << std::endl;
        fail++;
    }
    if(!hostVerif[1])
    {
        std::cerr << "Memcpy Host to Device 2D failed!" << std::endl;
        fail++;
    }
    if(!hostVerif[2])
    {
        std::cerr << "Memcpy Host to Device 3D failed!" << std::endl;
        fail++;
    }

    auto bufVerif1D = bufHost1D;
    auto bufVerif2D = bufHost2D;
    auto bufVerif3D = bufHost3D;

    bufHost1D = onHost::allocHost<Data>(extent1D);
    bufHost2D = onHost::allocHost<Data>(extent2D);
    bufHost3D = onHost::allocHost<Data>(extent3D);

    CopyRuntime1D = 0.0;
    CopyRuntime2D = 0.0;
    CopyRuntime3D = 0.0;

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        auto beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHost1D, bufAcc1D);
        onHost::wait(queue);
        auto endCopyT = std::chrono::high_resolution_clock::now();
        double copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        CopyRuntime1D += copyRuntime;

        beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHost2D, bufAcc2D);
        onHost::wait(queue);
        endCopyT = std::chrono::high_resolution_clock::now();
        copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        CopyRuntime2D += copyRuntime;

        beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHost3D, bufAcc3D);
        onHost::wait(queue);
        endCopyT = std::chrono::high_resolution_clock::now();
        copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        CopyRuntime3D += copyRuntime;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Device to Host memcpy: \n"
              << std::setprecision(6) << "\t 1D:\t" << CopyRuntime1D << "s\t" << std::fixed << std::setprecision(2)
              << numElements * numberOfRuns / CopyRuntime1D / 1E6 << " MB/s\n"
              << std::setprecision(6) << "\t 2D:\t" << CopyRuntime2D << "s\t" << std::fixed << std::setprecision(2)
              << numElements * numberOfRuns / CopyRuntime2D / 1E6 << " MB/s\n"
              << std::setprecision(6) << "\t 3D:\t" << CopyRuntime3D << "s\t" << std::fixed << std::setprecision(2)
              << numElements * numberOfRuns / CopyRuntime3D / 1E6 << " MB/s\n"
              << std::defaultfloat << std::setprecision(6);

    hostVerif[0] = 0u;
    hostVerif[1] = 0u;
    hostVerif[2] = 0u;

    hostVerify(bufHost1D, bufHost2D, bufHost3D, bufVerif1D, bufVerif2D, bufVerif3D, hostVerif);

    fail = 0;

    if(!hostVerif[0])
    {
        std::cerr << "Memcpy Device to Host 1D failed!" << std::endl;
        fail++;
    }
    if(!hostVerif[1])
    {
        std::cerr << "Memcpy Device to Host 2D failed!" << std::endl;
        fail++;
    }
    if(!hostVerif[2])
    {
        std::cerr << "Memcpy Device to Host 3D failed!" << std::endl;
        fail++;
    }

    if(fail)
    {
        std::cerr << "Error, " << fail << " device2host memcpy failed!\n";
        return EXIT_FAILURE;
    }


    std::cerr << "All tests passed!\n";
    return EXIT_SUCCESS;
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-n  numElements, power of 2] [-r numberOfRuns] [-h]" << std::endl;
}

int main(int argc, char* argv[])
{
    size_t numElements = 2'097'152;
    size_t numberOfRuns = 1;

    int opt;
    while((opt = getopt(argc, argv, "hn:r:")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
                if(!std::has_single_bit(numElements))
                {
                    std::cerr << "Error: value '" << optarg << "' is not a power of 2\n";
                    return EXIT_FAILURE;
                }
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'r':
            try
            {
                numberOfRuns = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid number of runs '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: number of runs '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            if(numberOfRuns == 0)
            {
                std::cerr << "Error: number of runs must be greater than zero.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }
    using namespace alpaka;

    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return example(deviceSpec, executor, numElements);
     *  @endcode
     *
     * Some examples for device specifications (depending on the active dependencies).
     *
     *   onHost::DeviceSpec{api::host, deviceKind::cpu}
     *   onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}
     *   onHost::DeviceSpec{api::hip, deviceKind::amdGpu}
     *   onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu}
     *
     * A list of api's and device kinds can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     * A list of executors can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */

    return onHost::executeForEach(
        [=](alpaka::concepts::BackendSpec auto const& backend)
        {
            auto selector = onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
            if(!selector.isAvailable())
                return EXIT_SUCCESS;
            return example(
                alpaka::onHost::DeviceSpec{backend},
                alpaka::getExecutor(backend),
                numElements,
                numberOfRuns);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
